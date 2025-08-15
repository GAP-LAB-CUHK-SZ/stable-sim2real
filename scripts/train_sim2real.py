import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from cldm.model import create_model, load_state_dict
from share import *
from pytorch_lightning.strategies import DDPStrategy

from dataset import Sim2RealDataset
import torch
torch.multiprocessing.set_sharing_strategy('file_system')


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_path', default='/data3/mutian/sim2real/weights/init-sd21_vae.ckpt')
    parser.add_argument('--data_path', default='/data3/mutian/lasa_cad_depth')
    parser.add_argument('--exp_name', default='diffusion-sim2real_cadcond_test')
    parser.add_argument('--config_file', default='config/generate_depth_cadcond.yaml')
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--logger_freq', type=int, default=500)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--accumulate_grad', type=int, default=1)
    return parser.parse_args()

def main(args):
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.config_file).cpu()
    model.load_state_dict(load_state_dict(args.resume_path, location='cpu'), strict=False)
    model.learning_rate = args.learning_rate

    # Checkpoint callback
    save_dir = f'./checkpoints/{args.exp_name}'
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        every_n_train_steps=10000,
        save_last=False,
        save_top_k=-1, # saving all model
    )

    dataset = Sim2RealDataset(data_dir=args.data_path, split='train', resolution=args.resolution)
    dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=args.logger_freq, save_dir=save_dir)
    trainer = pl.Trainer(accelerator='gpu', precision=16,  devices=args.gpus,  strategy="ddp_find_unused_parameters_true",
                         accumulate_grad_batches=args.accumulate_grad, 
                         callbacks=[logger,checkpoint_callback])
    trainer.fit(model, dataloader)

if __name__ == '__main__':
    args = get_argparse()
    main(args)
