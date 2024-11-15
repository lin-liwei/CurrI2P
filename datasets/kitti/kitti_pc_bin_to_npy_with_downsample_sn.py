import numpy as np
import open3d
import os
from scipy.spatial import cKDTree
import struct
from multiprocessing import Process

import matplotlib
matplotlib.use('TkAgg')
from data.kitti_helper import *


def read_velodyne_bin(path):
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2], point[3]])
    return np.asarray(pc_list, dtype=np.float32).T


def process_kitti(input_root_path,
                  output_root_path,
                  seq_list,
                  downsample_voxel_size,
                  sn_radius,
                  sn_max_nn):
    for seq in seq_list:
        input_folder = os.path.join(input_root_path, '%02d' % seq, 'velodyne')
        output_folder = os.path.join(output_root_path, '%02d' % seq, 'voxel%.1f-SNr%.1f' % (downsample_voxel_size, sn_radius))
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        sample_num = round(len(os.listdir(input_folder)))
        for i in range(sample_num):

            print('sequence %d: %d/%d' % (seq, i, sample_num))

            data_np = read_velodyne_bin(os.path.join(input_folder, '%06d.bin' % i))
            pc_np = data_np[0:3, :]
            intensity_np = data_np[3:, :]

            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(pc_np.T)
            downpcd = open3d.geometry.voxel_down_sample(pcd, voxel_size=downsample_voxel_size)

            open3d.geometry.estimate_normals(downpcd,
                                             search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=sn_radius,
                                                                                                  max_nn=sn_max_nn))
            open3d.geometry.orient_normals_to_align_with_direction(downpcd, [0,0,1])

            pc_down_np = np.asarray(downpcd.points).T
            pc_down_sn_np = np.asarray(downpcd.normals).T

            kdtree = cKDTree(pc_np.T)
            D, I = kdtree.query(pc_down_np.T, k=1)
            intensity_down_np = intensity_np[:, I]

            output_np = np.concatenate((pc_down_np, intensity_down_np, pc_down_sn_np), axis=0).astype(np.float32)
            np.save(os.path.join(output_folder, '%06d.npy' % i), output_np)



if __name__ == '__main__':
    input_root_path = '/data_odometry_velodyne/dataset/sequences'
    output_root_path = '/kitti/data_odometry_velodyne/sequences'
    downsample_voxel_size = 0.1
    sn_radius = 0.6
    sn_max_nn = 30
    seq_list = list(range(22))

    thread_num = 22  # One thread for one folder
    kitti_threads = []
    for i in range(thread_num):
        thread_seq_list = [i]
        kitti_threads.append(Process(target=process_kitti,
                                     args=(input_root_path,
                                           output_root_path,
                                           thread_seq_list,
                                           downsample_voxel_size,
                                           sn_radius,
                                           sn_max_nn)))

    for thread in kitti_threads:
        thread.start()

    for thread in kitti_threads:
        thread.join()


