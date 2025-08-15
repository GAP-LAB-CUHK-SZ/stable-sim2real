# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py
from typing import Any, Dict, Optional

import torch
from einops import rearrange
from ldm.modules.attention import BasicTransformerBlock
import pdb


def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class ReferenceAttentionControl:
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        reference_attn=True,
        fusion_blocks="midup",
        batch_size=1,
    ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            reference_attn,
            batch_size=batch_size,
            fusion_blocks=fusion_blocks,
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        reference_attn,
        dtype=torch.float16,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cpu"),
        fusion_blocks="midup",
    ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        reference_attn = reference_attn
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt


        # Not Use
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )

        def hacked_basic_transformer_inner_forward(
            self,
            x: torch.FloatTensor,
            context=None,
        ):
            '''
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x)) + x
            '''
            norm_hidden_states = self.norm1(x)

            # 1. Self-Attention
            # self.only_cross_attention = False
            if MODE == "write":
                self.bank.append(norm_hidden_states.clone())
                attn_output = self.attn1(
                    norm_hidden_states,
                    context=context if self.disable_self_attn else None
                )

            if MODE == "read":
                bank_fea = [
                    d for d in self.bank  # fix bug
                ]

                modify_norm_hidden_states = torch.cat(
                    [norm_hidden_states] + bank_fea, dim=1
                )

                hidden_states_uc = self.attn1(
                        norm_hidden_states,
                        context=context if self.disable_self_attn else modify_norm_hidden_states
                    )

                if do_classifier_free_guidance:
                    '''
                    hidden_states_c = hidden_states_uc.clone()
                    _uc_mask = uc_mask.clone()
                    if hidden_states.shape[0] != _uc_mask.shape[0]:
                        _uc_mask = (
                            torch.Tensor(
                                [1] * (hidden_states.shape[0] // 2)
                                + [0] * (hidden_states.shape[0] // 2)
                            )
                            .to(device)
                            .bool()
                        )
                    hidden_states_c[_uc_mask] = (
                        self.attn1(
                            norm_hidden_states[_uc_mask],
                            encoder_hidden_states=norm_hidden_states[_uc_mask],
                            attention_mask=attention_mask,
                        )
                        + hidden_states[_uc_mask]
                    )
                    hidden_states = hidden_states_c.clone()
                    '''
                    raise NotImplementedError
                else:
                    attn_output = hidden_states_uc

            x = attn_output + x
            # cross attn
            x = self.attn2(self.norm2(x), context=context) + x
            x = self.ff(self.norm3(x)) + x

            return x

        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.middle_block) + torch_dfs(self.unet.output_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )


            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, BasicTransformerBlock):
                    module._forward = hacked_basic_transformer_inner_forward.__get__(
                        module, BasicTransformerBlock
                    )

                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))

    def update(self, writer):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.middle_block) + torch_dfs(self.unet.output_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in (
                        torch_dfs(writer.unet.middle_block)
                        + torch_dfs(writer.unet.output_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in torch_dfs(writer.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            writer_attn_modules = sorted(
                writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                r.bank = [v.clone() for v in w.bank]
                # w.bank.clear()

    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.middle_block) + torch_dfs(self.unet.output_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r in reader_attn_modules:
                r.bank = []
