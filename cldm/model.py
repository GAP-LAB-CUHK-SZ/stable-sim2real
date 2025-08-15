import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import pdb
from os.path import join


def get_state_dict(d):

    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu', need_save=True):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        name, extension = os.path.splitext(os.path.basename(ckpt_path))
        main_path = f'{name}_main{extension}'
        if not os.path.exists(join(os.path.dirname(ckpt_path), os.path.basename(main_path))):
            state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))

            if need_save:
                torch.save(state_dict, join(os.path.dirname(ckpt_path), os.path.basename(main_path)) )
        else:
            state_dict = torch.load(join(os.path.dirname(ckpt_path), os.path.basename(main_path)), map_location=torch.device(location))

    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def load_state_dict_woclip(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        name, extension = os.path.splitext(os.path.basename(ckpt_path))
        main_path = f'{name}_main{extension}'
        if not os.path.exists(join(os.path.dirname(ckpt_path), os.path.basename(main_path))):
            state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
            torch.save(state_dict, join(os.path.dirname(ckpt_path), os.path.basename(main_path)) )
        else:
            state_dict = torch.load(join(os.path.dirname(ckpt_path), os.path.basename(main_path)), map_location=torch.device(location))

        for key in list(state_dict.keys()):
            if 'cond_stage_model.' in key:
                state_dict.pop(key)

    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def create_model(config_path):
    config = OmegaConf.load(config_path)  # a hierarchical dict
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
