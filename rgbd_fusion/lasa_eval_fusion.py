import open3d as o3d
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="true"
import glob
    
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--depth_dir",default='./output/stage2',type=str) # path to Stage-II output
parser.add_argument("--save_dir",default='./output/stage2',type=str) # path to save the fused mesh
parser.add_argument("--save_dir",default='./output/stage1',type=str) # path to save the fused mesh
args=parser.parse_args()

input_dir=os.path.join("./dataset/lasa_pose") # path to lasa_pose
scene_id_list = os.listdir(args.depth_dir)
# scene_id_list = ["47334202"]
out_depth = "pred" # "pred", "rand"

for scene_id in scene_id_list:
    print("TSDF fusing scene_{}".format(scene_id))
    image_path_list=glob.glob(os.path.join(args.depth_dir,scene_id,"*.npz"))
    image_id_list=[os.path.basename(item).split(".")[0] for item in image_path_list]
    pose_dir=os.path.join(args.pose_dir, scene_id)
    image_list=[]
    depth_list=[]
    pose_list=[]
    for image_id in image_id_list:
        pose_path=os.path.join(pose_dir, image_id+".txt")
        frame_path=os.path.join(args.depth_dir,scene_id,image_id+".npz")
        frame_data=np.load(frame_path, allow_pickle=True)
        rgb=frame_data['hint'] # H,W,3
        depth=frame_data[out_depth]
        if out_depth == "pred":
            depth=depth.transpose(1, 2, 0)
        depth = np.mean(depth, axis=-1)
        depth[depth<0]=0.0
        depth[depth>4.8]=0.0
        pose=np.loadtxt(pose_path)

        image_list.append(rgb)
        depth_list.append(depth)
        pose_list.append(pose)

    volume=o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    for frame_id in range(len(image_list)):
        color_image=image_list[frame_id]
        depth_im=depth_list[frame_id]
        pose=pose_list[frame_id]

        color_image = np.ascontiguousarray(color_image)
        color_image = o3d.geometry.Image((color_image*255).astype(np.uint8))
        depth_im = o3d.geometry.Image(depth_im.astype(np.float32))
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image,depth_im,depth_scale=1,depth_trunc=10.0,convert_rgb_to_intensity=False,
        )
        volume.integrate(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                width=512, height=512, fx=211.9489*(512/192), fy=211.9489*(512/192), cx=256, cy=256
            ),
            np.linalg.inv(pose)
        )
    mesh = volume.extract_triangle_mesh()
    # mesh.compute_vertex_normals()

    save_folder=os.path.join(args.save_dir,scene_id)
    if os.path.exists(save_folder)==False:
        os.makedirs(save_folder)
    mesh_path=os.path.join(save_folder,"{}_{}_mesh.obj".format(scene_id, out_depth))
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print("finished!")
