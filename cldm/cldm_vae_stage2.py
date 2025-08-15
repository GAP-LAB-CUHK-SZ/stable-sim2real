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
from sampler.registry_class import SAMPLER
from cldm.cldm import ControlNet
import pdb


class ControlLDMVAE(LatentDiffusion):
    def __init__(self, control_stage_config, control_key, only_mid_control, prior_model=None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13

        if prior_model is not None:
            from tqlt.prior_model.registry_class import NORMAL_PRIOR
            self.prior = NORMAL_PRIOR.build(dict(type=prior_model.name))
        else:
            self.prior = None

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)  # x is actually latent code
        control, control_2 = batch[self.control_key + "_cad"], batch[self.control_key + "_pred"]
        if bs is not None:
            control, control_2 = control[:bs], control_2[:bs]
        control, control_2 = control.to(self.device), control_2.to(self.device)
        control, control_2 = einops.rearrange(control, 'b h w c -> b c h w'), einops.rearrange(control_2, 'b h w c -> b c h w')
        control, control_2 = control.to(memory_format=torch.contiguous_format).float(), control_2.to(memory_format=torch.contiguous_format).float()

        control_posterior, control_posterior_2 = self.encode_first_stage(control), self.encode_first_stage(control_2) # pass to VAE encoder
        control_z = self.get_first_stage_encoding(control_posterior).detach()  # sample from VAE encoder output 
        control_z_2 = self.get_first_stage_encoding(control_posterior_2).detach()  # sample from VAE encoder output 

        # control_z = control_z + control_z_2
        control_z = torch.cat((control_z, control_z_2), dim=1)

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
        else:
            prior_z = None

        weight_mask = batch["weight_map"]

        return x, dict(c_crossattn=[c], c_concat=[control_z], prior_out=[prior_z], w_map=weight_mask)

    @torch.no_grad()
    def get_control(self, control):
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        control_posterior = self.encode_first_stage(control)
        control_z = self.get_first_stage_encoding(control_posterior).detach()

        return control_z


    @torch.no_grad()
    def get_hints(self, image_normalized):
        control = image_normalized
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        control_posterior = self.encode_first_stage(control)
        control_z = self.get_first_stage_encoding(control_posterior).detach()

        return {"c_concat" : [control_z]}


    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)

        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            hint = torch.cat(cond['c_concat'], 1)
            control = self.control_model(x=x_noisy, hint=hint, timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        if 'w_map' in cond:
            return eps, cond['w_map']
        else:
            return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=True, ddim_steps=5, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)  # bs valid for control.....

        c_cat, c = c["c_concat"][0][:N],  c["c_crossattn"][0][:N]

        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)

        z = z[:N]
        log["reconstruction"] = self.decode_first_stage(z)
        # log["control"] = self.decode_first_stage(c_cat)
        _,_, img_size_h, img_size_w = log['reconstruction'].shape
        log["conditioning"] = log_txt_as_img((img_size_w, img_size_h), batch[self.cond_stage_key][:N], size=16)

        log_sequence=['conditioning']

        if self.prior is not None:
            c_prior = c["prior_out"][0][:N]
            log['prior'] = self.decode_first_stage(c_prior)

            log_sequence.append('prior')
        else:
            c_prior = None
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
            # get denoise row (get the denoised lated code given noise and condition)
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)  # decode
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
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


class ControlNetVAE(ControlNet):
    def forward(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        outs = []

        h = hint.type(self.dtype)  # B, 4 or 8, 64, 64

        if h.shape[1] == 8:  # means concat
            h = self.input_new_blocks(h, emb, context)
        
        for module in self.input_blocks:
            h = module(h, emb, context)
            outs.append(h)
        h = self.middle_block(h, emb, context)
        outs.append(h)

        return outs

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            # training all paramters
            params += list(self.model.diffusion_model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
