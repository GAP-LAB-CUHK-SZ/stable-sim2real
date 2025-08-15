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
    def __init__(self, data_dir='data/ARKitScenes_depth', split='train', resolution=192):
        self.data = []

        if split=='train':
            split_file_path = '/data3/mutian/sim2real/train.lst'
        else:
            split_file_path = '/data3/mutian/sim2real/val.lst'
        
        with open(split_file_path,'r') as f:
            item_id_list=f.readlines()

        for item in item_id_list:
            item = item.rstrip() 
            _gt_path = os.path.join(data_dir, item+"_gt.png")
            _rgb_path = _gt_path.replace("_gt.png", "_rgb.png")
            _scan_path = _gt_path.replace("_gt.png", "_scan.png")
            self.data += [{"conditioning_image":_rgb_path, "scan_depth":_scan_path, "cad_depth":_gt_path, "item_id": item}]

        self.to_tensor = T.ToTensor()
        self.resolution = resolution
        if split == "train":
            self.transform = T.RandomResizedCrop(size=resolution, scale=(0.5,2.0))
        else:
            self.transform = T.Compose([
                T.CenterCrop((192, 192)),  # Crop the image to 192x192 without resizing
                T.Resize((512, 512))  # Resize the cropped image to 512x512
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

        rgb_image = (rgb_image * 2.0) - 1.0  # make it in [-1, 1]
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
    
    def __getitem__(self, idx):
        item = self.data[idx]

        item_id = item['item_id']
        rgb_image = self.load_image(item['conditioning_image'], "RGB")
        scan_depth = self.load_depth_image(item['scan_depth'])
        cad_depth = self.load_depth_image(item['cad_depth'])

        rgb_image_ori = rgb_image
        scan_depth_ori = scan_depth
        cad_depth_ori = cad_depth

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

        rgb_image, scan_depth, cad_depth = self.permute_channels(rgb_image, scan_depth, cad_depth)
        
        diff_depth = (scan_depth - cad_depth) / 2 # [-2, 2]

        return dict(jpg=diff_depth, txt="", hint=rgb_image, prior=cad_depth, ori_rgb=rgb_image_ori, ori_scan=scan_depth_ori, ori_cad=cad_depth_ori, item_id=item_id)