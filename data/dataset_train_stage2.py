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

def log_scale(d_r, d_min=0.05, d_max=5, filter_max=4.9):
    """
    Compute log-scaled depth with special handling for out-of-range values using NumPy.
    Invalid values will result in 1 after log scaling.
    
    Parameters:
    d_r (numpy.ndarray): Raw depth values
    d_min (float): Minimum supported depth
    d_max (float): Maximum supported depth for scaling
    filter_max (float): Maximum value for valid range
    
    Returns:
    numpy.ndarray: Normalized log-scaled depth values
    """
    # Create a mask for valid values
    valid_mask = (d_r >= d_min) & (d_r <= filter_max)
    
    # Create output array
    d_log = np.zeros_like(d_r)
    
    # Process valid values
    d_log[valid_mask] = np.log(d_r[valid_mask] / d_min) / np.log(d_max / d_min)
    
    # Replace invalid values with a value that results in 1 after log scaling
    invalid_value = d_max
    d_log[~valid_mask] = np.log(invalid_value / d_min) / np.log(d_max / d_min)
    
    return d_log


class Sim2RealDataset(Dataset):
    def __init__(self, data_dir='data/lasa_depth', stage1_dir='output/stage1', resolution=192):
        self.data = []

        split_file_path = os.path.join(data_dir, 'train.lst') # path to train.lst
        
        with open(split_file_path,'r') as f:
            item_id_list=f.readlines()
        for item in item_id_list:
            item = item.rstrip() 
            _gt_path = os.path.join(data_dir, item+"_gt.png")
            _rgb_path = _gt_path.replace("_gt.png", "_rgb.png")
            _scan_path = _gt_path.replace("_gt.png", "_scan.png")
            _pred_path = os.path.join(stage1_dir, item+"_pred_depth.npy") # Stage-I output path
            if os.path.exists(_pred_path): # only train on depth samples whose Stage-I output exists
                self.data += [{"conditioning_image":_rgb_path, "scan_depth":_scan_path, "cad_depth":_gt_path, "pred":_pred_path, "item_id": item}]
        
        self.to_tensor = T.ToTensor()
        self.resolution = resolution
        
        # In Stage-II, do not use random cropping, need to keep same with Stage-I output
        self.transform = T.Compose([
            T.CenterCrop((192, 192)),  # Crop the image to 192x192 (fit original input image size) without resizing
            T.Resize((resolution, resolution))  # Resize the cropped image to 512x512 (Stable Diffusion input size)
        ])

    def __len__(self):
        return len(self.data)
    
    def load_image(self, path, mode=None):
        """Load an image given its path."""
        image = Image.open(path)
        if mode:
            image = image.convert(mode)
        return self.to_tensor(image)
    
    def load_depth_image(self, path):
        """Load a depth image given its path."""
        return cv2.imread(path, -1).astype(np.float32) / 1000.0
    
    def log_scale_and_expand(self, image):
        """Apply log scale to the image and expand dimensions to match RGB."""
        scaled_image = log_scale(image)
        return np.stack([scaled_image] * 3, axis=-1)
    
    def split_and_process_data(self, data):
        """Split the data into rgb_image, scan_depth, cad_depth, and invalid mask, and process them."""
        rgb_image = data[:3]
        scan_depth = data[3:6]
        cad_depth = data[6:9]
        invalid_mask = data[9:].bool()
       
        # norm to # [-1, 1]
        rgb_image = (rgb_image * 2.0) - 1.0  
        scan_depth = (scan_depth * 2.0) - 1.0
        cad_depth = (cad_depth * 2.0) - 1.0
        return rgb_image, scan_depth, cad_depth, invalid_mask

    def apply_invalid_mask(self, rgb_image, scan_depth, cad_depth, invalid_mask):
        """Set invalid mask regions to 1 in the rgb_image, scan_depth, and cad_depth."""
        scan_depth[invalid_mask] = 1.0
        rgb_image[invalid_mask] = 1.0
        cad_depth[invalid_mask] = 1.0
        return rgb_image, scan_depth, cad_depth

    def permute_channels(self, *tensors):
        """Permute the channels of each tensor."""
        return [tensor.permute(1, 2, 0) for tensor in tensors]
    
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

    def __getitem__(self, idx):
        """process rgb, scan and cad depth, all same with Stage I:"""
        item = self.data[idx]
        rgb_image = self.load_image(item['conditioning_image'], "RGB")
        scan_depth = self.load_depth_image(item['scan_depth'])
        cad_depth = self.load_depth_image(item['cad_depth'])

        cad_depth_mask = cad_depth > 0.0
        diff_mask = np.abs(scan_depth - cad_depth) < 0.5  # filter context (cad has no depth on context)
        invalid_mask = self.to_tensor(~(cad_depth_mask & diff_mask)).repeat(3, 1, 1)
        
        scan_depth_tensor = self.to_tensor(self.log_scale_and_expand(scan_depth))
        cad_depth_tensor = self.to_tensor(self.log_scale_and_expand(cad_depth))
        
        data_stack = torch.cat((rgb_image, scan_depth_tensor, cad_depth_tensor, invalid_mask), 0)
        transformed_data = self.transform(data_stack)
        rgb_image, scan_depth, cad_depth, invalid_mask = self.split_and_process_data(transformed_data)

        # Set invalid mask values to 1 for depth channels
        rgb_image, scan_depth, cad_depth = self.apply_invalid_mask(rgb_image, scan_depth, cad_depth, invalid_mask)

        """process Stage-I output and reweight_mask:"""
        # load Stage-I output
        pred_depth = np.load(item['pred']).transpose(1, 2, 0)
        pred_depth = self.to_tensor(self.norm_and_log_scale(pred_depth))

        # get the txt file of diff ply path (if exist) and read them then project to get the mask
        diff_path = item['pred'].replace("pred_depth.npy", "diff_id.txt")
        if os.path.exists(diff_path):
            weight_mask = self.convert_to_weight(diff_path, pred_depth)
        else:  # all 0
            weight_mask = np.zeros((64, 64))
        weight_mask = self.to_tensor(weight_mask)

        rgb_image, scan_depth, cad_depth, pred_depth, weight_mask = self.permute_channels(rgb_image, scan_depth, cad_depth, pred_depth, weight_mask)
        
        # get residual
        diff_depth = (scan_depth - cad_depth) / 2 # [-2, 2] / 2
        diff_depth_pred = (pred_depth - cad_depth) / 2  # residual of Stage I output
    
        return dict(jpg=diff_depth, txt="", hint=rgb_image, prior_cad=cad_depth, prior_pred=diff_depth_pred, weight_map=weight_mask)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    data_path = "/data/mutian/lasa_depth"
    stage1_dir = "../output/stage1"
    dataset = Sim2RealDataset(data_dir=data_path, stage1_dir=stage1_dir, resolution=512)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=2, shuffle=True)
    for batch_id, _batch in enumerate(dataloader):
        print("!!!")
