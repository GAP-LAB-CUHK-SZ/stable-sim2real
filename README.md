<!-- PROJECT LOGO -->

<p align="center">
  <img src="https://mutianxu.github.io/stable-sim2real/static/images/icon_final.jpg" alt="" width="150" height="50"/>
  <h1 align="center">Stable-Sim2Real:
Exploring Simulation of Real-Captured 3D Data with Two-Stage Depth Diffusion</h1>
  <p align="center">
    <a href="https://mutianxu.github.io"><strong>Mutian Xu</strong></a>
    ·
    <a href="https://github.com/hugoycj"><strong>Chongjie Ye</strong></a>
    ·
    <a href="https://haolinliu97.github.io/"><strong>Haolin Liu</strong></a>
    ·
    <a href="https://yushuang-wu.github.io/"><strong>Yushuang Wu</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=HA5zLp4AAAAJ&hl=zh-CN"><strong>Jiahao Chang</strong></a>
    ·
    <a href="https://gaplab.cuhk.edu.cn/"><strong>Xiaoguang Han</strong></a>
    <br>
    SSE, CUHKSZ
    ·
    FNii-Shenzhen
    ·
    Guangdong Provincial Key Laboratory of Future Networks of Intelligence, CUHKSZ
    ·
    Tencent Hunyuan3D
    ·
    ByteDance Games

  </p>
  <h2 align="center">ICCV 2025 </font><font color="Tomato">(Highlight)</font></h2>
  <h3 align="center"><a href="https://arxiv.org/abs/2507.23483">Paper</a> | <a href="https://mutianxu.github.io/stable-sim2real/">Project Page</a></h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://mutianxu.github.io/stable-sim2real/static/images/teaser.jpg" alt="Logo" width="60%">
  </a>
</p>

*Stable-Sim2Real* is a two-stage depth diffusion model for simulating **real-captured** 3D data.
<br>

If you find our code or work helpful, please cite:
```bibtex
@inproceedings{xu2025sim2real,
        title={Stable-Sim2Real: Exploring Simulation of Real-Captured 3D Data with Two-Stage Depth Diffusion}, 
        author={Mutian Xu and Chongjie Ye and Haolin Liu and Yushuang Wu and Jiahao Chang and Xiaoguang Han},
        year={2025},
        booktitle = {International Conference on Computer Vision (ICCV)}
  }
```

TABLE OF CONTENTS
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#-news">News</a>
    </li>
    <li>
      <a href="#requirements-and-installation">Requirements and Installation</a>
    </li>
    <li>
      <a href="#data-preparation">Data Preparation</a>
    </li>
    <li>
      <a href="#run-stable-sim2real-on-lasa-validation-set">Run Stable-Sim2Real on LASA validation set</a>
    </li>
    <!-- <li>
      <a href="#segment-your-own-3d-scene">Segment Your Own 3D Scene</a>
    </li> -->
    <li>
      <a href="#todo">TODO</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
  </ol>
</details>

## 📢 News
- The first major revision of code is out. Try the latest code to run Stable-Sim2Real on LASA! 💪💪💪 (Aug.28, 2025 UTC)
- The initial code is released, which is simple scripts. (Aug.15, 2025 UTC)


## Requirements and Installation

### Hardware requirements
For conducting the inference on a single depth image, at least 3GB GPU memory usage is required.

### Software installation
Start by cloning the repo:
```bash
git clone https://github.com/GAP-LAB-CUHK-SZ/stable-sim2real.git
cd stable-sim2real
```

First of all, you have to make sure that you have all dependencies in place.
The simplest way to do so is to use [anaconda](https://www.anaconda.com/). 

You can create an Anaconda environment called `stable-sim2real` and install all the dependencies as below. For linux, you need to install `libopenexr-dev` before creating the environment. Then install all the remaining dependencies:

```bash
sudo apt-get install libopenexr-dev # for linux
conda create -n stable-sim2real python=3.8 (recommended python version >= 3.8)
conda activate stable-sim2real
pip install -r requirements.txt
```

Next, download all the pretrained models from [OneDrive](https://cuhko365-my.sharepoint.com/:f:/g/personal/221019043_link_cuhk_edu_cn/Em-VWJNEVYVMuDEqGO8oUCIBzWJQVE_iISFtZyrSY6128g?e=tQIyJa) or [Sharepoint](https://cuhko365.sharepoint.com/:f:/s/CUHKSZ_SSE_GAP-Lab2/EmLWPs2C-pNBh-yX9mENuz4BxJ7jn9tW895mKixHnKVUrA?e=aGNmP7), and put all of them into `pretrained_weights` folder.

## Data Preparation

Download the training dataset from [OneDrive](https://cuhko365-my.sharepoint.com/:u:/g/personal/221019043_link_cuhk_edu_cn/EVvkUo5jgEZNiTXNGG1hXBkBTVefVVBLjLJxK7sbzBGjVQ?e=ZmoWX3) or [Sharepoint](https://cuhko365.sharepoint.com/:u:/s/CUHKSZ_SSE_GAP-Lab2/ET66yxmFoExInhh4lEAGuqEB53PsFAK3Z-ns1QeMBR3QYQ?e=wAJark), and put `lasa_depth` into `dataset` folder. It contains 40,000+ synthetic-real depth pairs with corresponding RGB images. It is processed from [LASA](https://github.com/GAP-LAB-CUHK-SZ/LASA), a large-scale aligned shape annotation dataset with CAD/real 3D data pairs.

Next, download the camera poses from [OneDrive](https://cuhko365-my.sharepoint.com/:u:/g/personal/221019043_link_cuhk_edu_cn/EbgsJC64R_BKudMyK2hB4icByEo-42goIY1ZaRfOx3znXA?e=ZFvbaL) or [Sharepoint](https://cuhko365.sharepoint.com/:u:/s/CUHKSZ_SSE_GAP-Lab2/Ebh2CjOhyyNHrVEfov8rt-UB_rr1Mbl8oUVEZ_-0qXnR_A?e=2YxiJZ), and put `lasa_pose` into `dataset` folder. This is for later depth fusion.

## Run Stable-Sim2Real on LASA validation set

### Run Stage-I
This involves generating and saving Stage-I outputs into `.npy` files, by simply running (you may also need to specify some path arguments by yourself):
```
CUDA_VISIBLE_DEVICES=x python eval_stage1.py
```

### Run Stage-II
This involves generating and saving Stage-II outputs into `.npz` files, by simply running (you may also need to specify some path arguments by yourself):
```
CUDA_VISIBLE_DEVICES=x python eval_stage2.py
```

### TSDF fusion
Finally, the Stage-II output depth will be fused (you may also need to specify some path arguments by yourself):
```
cd rgbd_fusion
python lasa_eval_fusion.py
```

After finishing this, the visualization result of the final 3D simulated mesh will be *automatically* 😊 saved as `xxx_mesh.obj` file.

## 🚩 TODO
- [ ] Provide demo code (will also support HuggingFace) to simulate and fuse your own CAD object/scene depth (including render, simulate, and fusion).
- [ ] Complete training code with training and inference of 3D-Aware Local Discriminating.
- [ ] Provide some sample data from OOD datasets (e.g., ShapeNet) for inference.

## Contact
You are welcome to submit issues, send pull requests, or share some ideas with us. If you have any other questions, please contact Mutian Xu (mutianxu@link.cuhk.edu.cn).

## Acknowledgement
Our code base is partially borrowed or adapted from [ControlNet](https://github.com/lllyasviel/ControlNet), [Stable-Normal](https://github.com/Stable-X/StableNormal), and [LASA](https://github.com/GAP-LAB-CUHK-SZ/LASA).
