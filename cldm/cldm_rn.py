import einops
import torch
import torch as th
import torch.nn as nn

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from tqlt.prior_model.registry_class import NORMAL_PRIOR
from sampler.registry_class import SAMPLER
from cldm.cldm import ControlNet
import pdb
from cldm.attention.mutual_self_attention import ReferenceAttentionControl


class ControlLDMReferenceNet(LatentDiffusion):
    def __init__(self, control_stage_config, control_key, only_mid_control, prior_model=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key

        self.ref_writer = ReferenceAttentionControl(self.control_model, 'write', fusion_blocks='midup')
        self.ref_reader= ReferenceAttentionControl(self.model.diffusion_model, 'read', fusion_blocks='midup')

        if prior_model is not None:
            self.prior = NORMAL_PRIOR.build(dict(type=prior_model.name))
        else:
            self.prior = None

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):

        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        control_posterior = self.encode_first_stage(control)
        control_z = self.get_first_stage_encoding(control_posterior).detach()

        # using prior model to capture x0
        if self.prior is not None:
            # [-1,1] -> [0, 255]
            prior_hint = (batch['hint']+1.) / 2 * 255.
            _, hint_h, hint_w, _ = prior_hint.shape
            size_info = [hint_w, hint_h, 0, 0]
            self.prior.to(self.device)
            # [-1, 1]
            prior_out = self.prior(prior_hint, size_info=size_info)['abs_vals']
            # prior embedding
            prior_posterior = self.encode_first_stage(prior_out)
            prior_z = self.get_first_stage_encoding(prior_posterior).detach()

        return x, dict(c_crossattn=[c], c_concat=[control_z], prior_out=[prior_z])

    @torch.no_grad()
    def get_control(self, control):
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        control_posterior = self.encode_first_stage(control)
        control_z = self.get_first_stage_encoding(control_posterior).detach()


        return control_z


    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)

        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)


        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            hint = torch.cat(cond['c_concat'], 1)
            # reference_net
            ref_t = torch.zeros_like(t)
            self.control_model(x=hint, timesteps=ref_t, context=cond_txt)
            self.ref_reader.update(self.ref_writer)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt)

            # clean bank
            self.ref_reader.clear()
            self.ref_writer.clear()

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)  # bs valid for control.....


        c_cat, c_prior, c = c["c_concat"][0][:N], c["prior_out"][0][:N], c["c_crossattn"][0][:N]

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)

        z = z[:N]
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = self.decode_first_stage(c_cat)
        _,_, img_size_h, img_size_w = log['reconstruction'].shape
        log["conditioning"] = log_txt_as_img((img_size_w, img_size_h), batch[self.cond_stage_key][:N], size=16)

        log_sequence=['control', 'conditioning']

        if self.prior is not None:
            log['prior'] = self.decode_first_stage(c_prior)

            log_sequence.append('prior')

        log_sequence.append('samples')

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], 'c_prior': [c_prior]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], 'c_prior': [c_prior]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

            log_sequence.append(f"samples_cfg_scale_{unconditional_guidance_scale:.2f}")

        log_sequence.append('reconstruction')
        log['visualized'] = torch.cat([log[key].detach().cpu() for key in log_sequence], dim = -2)

        for key in log_sequence:
            log.pop(key)

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):

        ddim_sampler = SAMPLER.build(dict(type=self.sampler_type), model=self)

        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h, w)

        if self.prior is not None:
            c_prior = cond.pop('c_prior')
            kwargs['x0'] = c_prior[0]

        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
