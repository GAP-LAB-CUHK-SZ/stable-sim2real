import sys
import config
import einops
import numpy as np
import torch
from tqdm import tqdm
from cldm.model import create_model, load_state_dict
from einops import repeat
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
import tqlt
import random
import pdb


class simple_sd_model(object):
    '''A model warp sd model

    '''
    def __init__(self, pred_time = 401, model_path='./models/hypersim_finetune_from_indoor500k-sd21_768_vae_onestep_from_Gauss-and-Zero_bae_coordinates_plus_RandomIntrinis_plus_discrete_timestamps-step_13149.ckpt', noise=True, prompt=""):
        self.model = create_model('./config/cldm_v21_plus_VAE_onestep_x0.yaml').cpu()
        self.model.load_state_dict(load_state_dict(model_path, location='cpu'))
        self.model.eval()

        self.pred_time = pred_time
        self.noise = noise
        self.prompt = prompt

    def cuda(self):
       self.model.cuda()
       return self

    def cpu(self):
       self.model.cpu()
       return self

    def float(self):
       self.model.float()
       return self

    def to(self, device):
       self.model.to(device)
       return self

    def eval(self):
        self.model.eval()

        return self

    def train(self):
        self.model.train()
        return self

    @torch.no_grad()
    def __call__(self, x, num_samples, H, W):
        control = self.model.get_control(x[None]).detach()
        control = torch.cat([control for _ in range(num_samples)], dim=0)
        cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([self.prompt] * num_samples)]}



        ts = torch.full((num_samples,), self.pred_time, device=self.model.device, dtype=torch.long)

        if self.noise:
            x_gauss = torch.randn([num_samples, 4, H // 8, W // 8], device=self.model.device)
        else:
            x_gauss =  torch.zeros([num_samples, 4, H // 8, W // 8], device=self.model.device)

        x_T = self.model.apply_model(x_gauss, ts, cond)
        x_0 = self.model.apply_model(x_gauss, torch.full((num_samples,), 0, device=self.model.device, dtype=torch.long), cond)

        return  x_0, x_T

    def decode_first_stage(self, sample):
        return self.model.decode_first_stage(sample)

    def __repr__(self):

        return f"model: \n{self.model}"
