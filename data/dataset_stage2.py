import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
import random
import pdb
from glob import glob
import PIL.Image as Image
import torchvision.transforms as T
import cv2
from share import *
import torch
from plyfile import PlyData
import re


class Sim2RealDataset(Dataset):
    def __init__(self, data_dir='data/ARKitScenes_depth', split='train', resolution=192):
        self.data = []

        if split=='train':
            split_file_path = '/data/mutian/sim2real/train_scene.lst'
        else:
            split_file_path = '/data/mutian/sim2real/val_scene.lst'
        
        with open(split_file_path,'r') as f:
            scene_id_list=f.readlines()

        for scene in scene_id_list:
            scene_id = scene.rstrip() 
            scene_dir = os.path.join(data_dir, scene_id)
            pred_list = glob(os.path.join(data_dir, scene_id,"*_pred_*.npy"))

            for pred in pred_list:
                _pred_path = pred
                _rgb_path = _pred_path.replace("pred_depth", "hint")#.replace("cadcond_scene_diff", "cadcond_scene")
                _gt_cad_path = _pred_path.replace("pred_depth", "gt_cad_depth")#.replace("cadcond_scene_diff", "cadcond_scene")
                _gt_scan_path = _pred_path.replace("pred_depth", "gt_scan_depth")#.replace("cadcond_scene_diff", "cadcond_scene")
                
                if os.path.exists(_pred_path) and os.path.exists(_gt_cad_path) and os.path.exists(_gt_scan_path):
                    self.data += [{"pred":_pred_path, "gt_cad":_gt_cad_path, "gt_scan":_gt_scan_path}]

        self.to_tensor = T.ToTensor()
    
    def norm_and_log_scale(self, d_r, d_min=0.05, d_max=5):
        d_ori = (2.0 * np.log(d_r / d_min) / np.log(d_max / d_min)) - 1.0
        return d_ori

    def convert_to_weight(self, path, depth):
        weight_mask = np.zeros((depth.shape[0], depth.shape[1]))
        with open(path, "r") as files:
            for file in files:
                file_name = file.rstrip().replace(".ply", "_position.txt")
                with open(file_name, "r") as pixel_ids:
                    pixel_range = pixel_ids.readline() 
                numbers = [int(num) for num in re.findall(r'\d+', pixel_range)]
                weight_mask[numbers[0]:numbers[1], numbers[2]:numbers[3]] = 1  # set to 1
        weight_mask = cv2.resize(weight_mask, (64, 64), interpolation=cv2.INTER_NEAREST)
        return weight_mask

    def permute_channels(self, *tensors):
        """Permute the channels of each tensor."""
        return [tensor.permute(1, 2, 0) for tensor in tensors]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        # item_id = item['item_id']
        rgb_image = np.load(item['conditioning_image'])  # actually useless
        scan_depth = np.load(item['gt_scan'])
        cad_depth = np.load(item['gt_cad'])
        pred_depth = np.load(item['pred']).transpose(1, 2, 0)
        
        # get the txt file of diff ply path (if exist) and read them then project to get the mask
        diff_path = item['pred'].replace("pred_depth.npy", "diff_id.txt")
        if os.path.exists(diff_path):
            weight_mask = self.convert_to_weight(diff_path, pred_depth)
        else:  # all 0
            weight_mask = np.zeros((64, 64))
        
        rgb_image_ori = rgb_image
        scan_depth_ori = scan_depth
        cad_depth_ori = cad_depth
        pred_depth_ori = pred_depth

        rgb_image = self.to_tensor((rgb_image * 2.0) - 1.0)
        scan_depth = self.to_tensor(self.norm_and_log_scale(scan_depth))
        cad_depth = self.to_tensor(self.norm_and_log_scale(cad_depth))
        pred_depth = self.to_tensor(self.norm_and_log_scale(pred_depth))
        weight_mask = self.to_tensor(weight_mask)

        rgb_image, scan_depth, cad_depth, pred_depth, weight_mask = self.permute_channels(rgb_image, scan_depth, cad_depth, pred_depth, weight_mask)
        
        diff_depth = (scan_depth - cad_depth) / 2
        diff_depth_pred = (pred_depth - cad_depth) / 2
    
        return dict(jpg=diff_depth, txt="", hint="rgb_image", prior_cad=cad_depth, prior_pred=diff_depth_pred, weight_map=weight_mask, ori_rgb=rgb_image_ori, ori_scan=scan_depth_ori, ori_cad=cad_depth_ori, ori_pred=pred_depth_ori)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data_path = "/data3/mutian/generated_depth/diffusion-sim2real_cadcond_scene_diff"
    dataset = Sim2RealDataset(data_dir=data_path, split='train', resolution=512)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=2, shuffle=True)
    for data in dataloader:
        print("!!!")
