import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
from cldm.model import create_model, load_state_dict
from share import *
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np 
from scipy.stats import wasserstein_distance
from data.dataset_infer_stage2 import Sim2RealDataset
from pytorch_lightning import seed_everything

def unnorm_and_inverse_log_scale(d_log, d_min=0.05, d_max=5):
    """
    Invert the log-scaled depth values.
    
    Parameters:
    d_log (numpy.ndarray): Log-scaled depth values
    d_min (float): Minimum supported depth
    d_max (float): Maximum supported depth for scaling
    
    Returns:
    numpy.ndarray: Inverted depth values
    """
    d_log = (d_log + 1) / 2
    return d_min * np.exp(d_log * np.log(d_max / d_min))


def unnorm_and_inverse_log_scale_diff(d_cad, d_diff, d_min=0.05, d_max=5):
    """
    1. Add the output residual (d_diff) to the cad to get the output depth
    2. Invert the log-scaled depth values.
    
    Parameters:
    d_diff (numpy.ndarray): Log-scaled residual values
    d_min (float): Minimum supported depth
    d_max (float): Maximum supported depth for scaling
    
    Returns:
    numpy.ndarray: Inverted depth values
    """
    d_diff = d_diff * 2
    d_log = d_diff + d_cad.transpose(0,3,1,2)  # [-1, 1]
    d_log = (d_log + 1) / 2
    return d_min * np.exp(d_log * np.log(d_max / d_min))


def inverse_log_scale(d_log, d_min=0.05, d_max=5):
    """
    Invert the log-scaled depth values.
    
    Parameters:
    d_log (numpy.ndarray): Log-scaled depth values
    d_min (float): Minimum supported depth
    d_max (float): Maximum supported depth for scaling
    
    Returns:
    numpy.ndarray: Inverted depth values
    """
    return d_min * np.exp(d_log * np.log(d_max / d_min))


def compute_depth_metrics(gt, pred, mask):
    """
    Compute depth evaluation metrics.
    
    Parameters:
    gt (numpy.ndarray): Ground truth depth map
    pred (numpy.ndarray): Predicted depth map
    mask (numpy.ndarray): Boolean mask for valid areas
    
    Returns:
    dict: Dictionary containing various depth metrics
    """
    gt = gt[mask]
    pred = pred[mask]
    
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    
    return {
        'a1': a1,
        'a2': a2,
        'a3': a3,
        'rmse': rmse,
        'rmse_log': rmse_log,
        'abs_rel': abs_rel,
        'sq_rel': sq_rel
    }
    

def calculate_depth_diff_wasserstein(pred, cad_depth, scan_depth, mask):
    """
    Calculate the Wasserstein distance between depth difference distributions in the valid mask area.
    
    Parameters:
    pred (numpy.ndarray): Predicted depth map
    cad_depth (numpy.ndarray): CAD depth map
    scan_depth (numpy.ndarray): Scanned depth map
    mask (numpy.ndarray): Boolean mask for valid areas
    
    Returns:
    float: Wasserstein distance between the two depth difference distributions
    """
    # Apply mask
    pred_valid = pred[mask]
    cad_valid = cad_depth[mask]
    scan_valid = scan_depth[mask]
    
    # Calculate depth differences
    model_diff = cad_valid - pred_valid
    data_diff = cad_valid - scan_valid
    
    # Calculate Wasserstein distance (Earth Mover's Distance)
    wass_dist = wasserstein_distance(model_diff, data_diff)
    
    return wass_dist

def add_noise(depth_maps, noise_level=0.1):
    depth_maps = np.mean(depth_maps, axis=-1)

    # Generate random noise with the same shape as the input depth maps
    noise = np.random.randn(*depth_maps.shape) * noise_level
    
    # Preserve values of 5 in the depth maps
    noisy_depth_maps = depth_maps.copy()
    noisy_depth_maps = np.where(noisy_depth_maps != 5, noisy_depth_maps + noise, noisy_depth_maps)
    
    return noisy_depth_maps


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_path', default='pretrained_weights/stage2_main.ckpt')
    parser.add_argument('--data_path', default='dataset/lasa_depth') # path to lasa_depth
    parser.add_argument('--stage1_output_path', default='output/stage1') # path to Stage-I output
    parser.add_argument('--save_dir', default='output/stage2') # path to save the Stage-II prediction files, need about 6MB for each depth, need about 50+GB to hold all LASA scene depth for eval Stage-II
    parser.add_argument('--config_file', default='config/eval_stage2.yaml')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=23016)
    return parser.parse_args()

def main(args):
    seed = args.seed
    seed_everything(seed)
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.config_file).cpu()
    model.load_state_dict(load_state_dict(args.resume_path, location='cpu'), strict=False)
    model = model.cuda()
    
    # Prepare evaluation dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    dataset = Sim2RealDataset(data_dir=args.data_path, stage1_dir=args.stage1_output_path, resolution=args.resolution)
    print("Data quantity:", len(dataset))
    dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size, shuffle=False)
    for batch_id, _batch in enumerate(dataloader):
        _scene_id = _batch["scene_id"]
        _frame_id = _batch["frame_id"]
        _result = model.log_images(_batch, N=args.batch_size, ddim_steps=args.ddim_steps, unconditional_guidance_scale=1)
        _gt, _pred = unnorm_and_inverse_log_scale_diff(_batch['prior_cad'].cpu().numpy(), _result['reconstruction'].cpu().numpy()), unnorm_and_inverse_log_scale_diff(_batch['prior_cad'].cpu().numpy(), _result['samples'].cpu().numpy())  # B, 3, H, W
        # _cad_depth, _scan_depth = unnorm_and_inverse_log_scale(_batch['prior_cad'].cpu().numpy()), unnorm_and_inverse_log_scale(_batch['scan'].cpu().numpy())  # B, H, W, 3
        rgb = _batch['hint'].cpu().numpy()
        # save image array for visualization and rgbd fusion
        for i in range(len(_scene_id)):
            scene_id = _scene_id[i]
            sub_dir = os.path.join(args.save_dir, scene_id)
            os.makedirs(sub_dir, exist_ok=True)
            frame_id = _frame_id[i]
            np.savez_compressed(os.path.join(sub_dir, frame_id + '.npz'), 
                                hint=(rgb[i]+1)/2,
                                pred=_pred[i])
            # np.save(os.path.join(sub_dir, frame_id + '_rgb.npy'), ori_rgb[i])
            # np.save(os.path.join(sub_dir, frame_id + '_scan.npy'), ori_scan[i])
            # np.save(os.path.join(sub_dir, frame_id + '_cad.npy'), ori_cad[i])
            # np.save(os.path.join(sub_dir, frame_id + '_hint.npy'), (rgb[i]+1)/2)
            # np.save(os.path.join(sub_dir, frame_id + '_decode_gt_scan_depth.npy'), _gt[i])
            # np.save(os.path.join(sub_dir, frame_id + '_gt_cad_depth.npy'), _cad_depth[i])
            # np.save(os.path.join(sub_dir, frame_id + '_gt_scan_depth.npy'), _scan_depth[i])
            # np.save(os.path.join(sub_dir, frame_id + '_pred_depth.npy'), _pred[i])
            # np.save(os.path.join(sub_dir, frame_id + '_rand_depth.npy'), _rand_depth[i])

if __name__ == '__main__':
    args = get_argparse()
    main(args)
