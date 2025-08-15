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
from dataset_eval import Sim2RealDataset
from pytorch_lightning import seed_everything


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
    # parser.add_argument('--resume_path', default='/data2/chongjie/workspace/code/NormalDiffusion/weights/init-sd21_vae.ckpt')
    parser.add_argument('--resume_path', default='/data3/mutian/sim2real/checkpoints/diffusion-sim2real_cadcond/epoch=19-step=20000.ckpt')
    parser.add_argument('--data_path', default='/data3/mutian/lasa_cad_depth')
    parser.add_argument('--exp_name', default='diffusion-sim2real_cadcond_stage2')
    parser.add_argument('--scene_id', default='45261507')
    parser.add_argument('--config_file', default='config/eval_depth_cadcond.yaml')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=23012)
    return parser.parse_args()

def main(args):
    seed = args.seed
    seed_everything(seed)
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.config_file).cpu()
    model.load_state_dict(load_state_dict(args.resume_path, location='cpu'), strict=False)
    model = model.cuda()
    
    # Prepare evaluation dir
    save_dir = f'./eval_results/{args.exp_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = Sim2RealDataset(data_dir=args.data_path, split='val', scene_id=args.scene_id, resolution=args.resolution)
    print("Data quantity:", len(dataset))
    dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size, shuffle=False)

    all_metrics = []
    for batch_id, _batch in enumerate(dataloader):
        _item_id = _batch["item_id"]
        _result = model.log_images(_batch, N=args.batch_size, ddim_steps=args.ddim_steps, unconditional_guidance_scale=1)
        _gt, _pred = unnorm_and_inverse_log_scale_diff(_batch['prior'].cpu().numpy(), _result['reconstruction'].cpu().numpy()), unnorm_and_inverse_log_scale_diff(_batch['prior'].cpu().numpy(), _result['samples'].cpu().numpy())  # B, 3, H, W
        _cad_depth, _scan_depth = unnorm_and_inverse_log_scale(_batch['prior'].cpu().numpy()),  unnorm_and_inverse_log_scale(_batch['scan'].cpu().numpy())  # B, H, W, 3

        # add random noise to _cad_depth for comparison:
        # _rand_depth = add_noise(_cad_depth, noise_level=0.5)
        
        # save image array for visualization
        # ori_rgb = _batch['ori_rgb']
        # ori_scan = _batch['ori_scan']
        # ori_cad = _batch['ori_cad']
        rgb = _batch['hint'].cpu().numpy()
        for i in range(len(_item_id)):
            sub_dir = os.path.join(save_dir, args.scene_id)
            os.makedirs(sub_dir, exist_ok=True)
            frame_id = _item_id[i]
            # np.save(os.path.join(sub_dir, frame_id + '_rgb.npy'), ori_rgb[i])
            # np.save(os.path.join(sub_dir, frame_id + '_scan.npy'), ori_scan[i])
            # np.save(os.path.join(sub_dir, frame_id + '_cad.npy'), ori_cad[i])
            np.save(os.path.join(sub_dir, frame_id + '_hint.npy'), (rgb[i]+1)/2)
            # np.save(os.path.join(sub_dir, frame_id + '_decode_gt_scan_depth.npy'), _gt[i])
            np.save(os.path.join(sub_dir, frame_id + '_gt_cad_depth.npy'), _cad_depth[i])
            np.save(os.path.join(sub_dir, frame_id + '_gt_scan_depth.npy'), _scan_depth[i])  
            np.save(os.path.join(sub_dir, frame_id + '_pred_depth.npy'), _pred[i])
            # np.save(os.path.join(sub_dir, frame_id + '_rand_depth.npy'), _rand_depth[i])
        """
        _gt = _gt.mean(axis=1)
        _pred = _pred.mean(axis=1)
        _cad_depth = _cad_depth.mean(axis=-1)
        _scan_depth = _scan_depth.mean(axis=-1)
        
        # Create a mask for valid areas
        valid_mask = (_gt > 0) & (_pred > 0) & (_gt < 4.8) & (_pred < 4.8)
        
        # Compute metrics for each sample in the batch
        for i in range(_gt.shape[0]):
            metrics = compute_depth_metrics(_gt[i], _pred[i], valid_mask[i])
            wass_dist = calculate_depth_diff_wasserstein(_pred[i], _cad_depth[i], _scan_depth[i], valid_mask[i])
            metrics['wasserstein_distance'] = wass_dist
            all_metrics.append(metrics)
        
        print(f"Batch metrics: {metrics}")

    # Compute average metrics across all samples
    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
    print(f"Average metrics: {avg_metrics}")
    
    # Save average metrics
    avg_metrics_save_path = os.path.join(save_dir, 'average_metrics.json')
    with open(avg_metrics_save_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    """

if __name__ == '__main__':
    args = get_argparse()
    main(args)
