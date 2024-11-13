import os
import torch
import argparse
from models.network import CurrI2P
from models.kitti_dataset import kitti_pc_img_dataset
import numpy as np
import math
import utils.options as options
import cv2
from scipy.spatial.transform import Rotation

def get_P_diff(P_pred_np, P_gt_np):
    P_diff = np.dot(np.linalg.inv(P_pred_np), P_gt_np)
    t_diff = np.linalg.norm(P_diff[0:3, 3])
    r_diff = P_diff[0:3, 0:3]
    R_diff = Rotation.from_matrix(r_diff)
    angles_diff = np.sum(np.abs(R_diff.as_euler('xzy', degrees=True)))
    return t_diff, angles_diff

def test_acc_trans(model, testdataloader, args):
    t_diff_set = []
    angles_diff_set = []

    for step, data in enumerate(testdataloader):

        if step % 1 == 0:
            model.eval()
            img = data['img'].cuda()
            pc_all = data['pc'].cuda()
            intensity = data['intensity'].cuda()
            sn = data['sn'].cuda()
            K_all = data['K'].cuda()
            P_all = data['P'].cuda()

            img_features_all, pc_features_all, img_score_all, pc_score_all = model(pc_all, intensity, sn,
                                                                                   img)

            bs = img_score_all.shape[0]

            for i in range(bs):
                img_score = img_score_all[i]
                pc_score = pc_score_all[i]
                img_feature = img_features_all[i]
                pc_feature = pc_features_all[i]
                pc = pc_all[i]
                P = P_all[i].data.cpu().numpy()
                K = K_all[i].data.cpu().numpy()

                img_x = np.linspace(0, np.shape(img_feature)[-1] - 1, np.shape(img_feature)[-1]).reshape(1, -1).repeat(
                    np.shape(img_feature)[-2], 0).reshape(1, np.shape(img_score)[-2], np.shape(img_score)[-1])
                img_y = np.linspace(0, np.shape(img_feature)[-2] - 1, np.shape(img_feature)[-2]).reshape(-1, 1).repeat(
                    np.shape(img_feature)[-1], 1).reshape(1, np.shape(img_score)[-2], np.shape(img_score)[-1])

                img_xy = np.concatenate((img_x, img_y), axis=0)
                img_xy = torch.tensor(img_xy).cuda()

                img_xy_flatten = img_xy.reshape(2, -1)
                img_feature_flatten = img_feature.reshape(np.shape(img_feature)[0], -1)
                img_score_flatten = img_score.squeeze().reshape(-1)

                img_index = (img_score_flatten > args.img_thres)

                img_xy_flatten_sel = img_xy_flatten[:, img_index]
                img_feature_flatten_sel = img_feature_flatten[:, img_index]

                pc_index = (pc_score.squeeze() > args.pc_thres)

                pc_sel = pc[:, pc_index]
                pc_feature_sel = pc_feature[:, pc_index]

                dist = 1 - torch.sum(img_feature_flatten_sel.unsqueeze(2) * pc_feature_sel.unsqueeze(1), dim=0)
                sel_index = torch.argsort(dist, dim=1)[:, 0]

                pc_sel = pc_sel[:, sel_index].detach().cpu().numpy()
                img_xy_pc = img_xy_flatten_sel.detach().cpu().numpy()

                is_success, R, t, inliers = cv2.solvePnPRansac(pc_sel.T, img_xy_pc.T, K, useExtrinsicGuess=False,
                                                               iterationsCount=500,
                                                               reprojectionError=args.dist_thres,
                                                               flags=cv2.SOLVEPNP_EPNP,
                                                               distCoeffs=None)

                R, _ = cv2.Rodrigues(R)
                T_pred = np.eye(4)
                T_pred[0:3, 0:3] = R
                T_pred[0:3, 3:] = t
                t_diff, angles_diff = get_P_diff(T_pred, P)
                t_diff_set.append(t_diff)
                angles_diff_set.append(angles_diff)

            if step % 100 == 0:
                t_diff_set_np = np.array(t_diff_set)
                angles_diff_set_np = np.array(angles_diff_set)

                index = (angles_diff_set_np < 5) & (t_diff_set_np < 2)
                print('step:', step, '---', 'Good rate : ', t_diff_set_np[index].shape, '/', t_diff_set_np.shape)
                print('RTE mean', np.mean(t_diff_set_np), 'std', np.std(t_diff_set_np))
                print('RRE mean', np.mean(angles_diff_set_np), 'std', np.std(angles_diff_set_np))

    t_diff_set = np.array(t_diff_set)
    angles_diff_set = np.array(angles_diff_set)
    print('RTE mean', np.mean(t_diff_set), 'std', np.std(t_diff_set))
    print('RRE mean', np.mean(angles_diff_set), 'std', np.std(angles_diff_set))

    return t_diff_set, angles_diff_set

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CurrI2P')
    parser.add_argument('--epoch', type=int, default=25, metavar='epoch',
                        help='number of epoch to train')
    parser.add_argument('--train_batch_size', type=int, default=12, metavar='train_batch_size',
                        help='Size of train batch')
    parser.add_argument('--val_batch_size', type=int, default=8, metavar='val_batch_size',
                        help='Size of val batch')
    parser.add_argument('--data_path', type=str, default='/data/kitti/', metavar='data_path',
                        help='train and test data path')
    parser.add_argument('--num_workers', type=int, default=8, metavar='num_workers',
                        help='num of CPUs')
    parser.add_argument('--input_pt_num', type=int, default=40960, metavar='input_pt_num',
                        help='input_pt_num')
    parser.add_argument('--P_tx_amplitude', type=float, default=10, metavar='P_tx_amplitude',
                        help='')
    parser.add_argument('--P_ty_amplitude', type=float, default=0, metavar='P_ty_amplitude',
                        help='')
    parser.add_argument('--P_tz_amplitude', type=float, default=10, metavar='P_tz_amplitude',
                        help='')
    parser.add_argument('--P_Rx_amplitude', type=float, default=2 * math.pi * 0, metavar='P_Rx_amplitude',
                        help='')
    parser.add_argument('--P_Ry_amplitude', type=float, default=2 * math.pi, metavar='P_Ry_amplitude',
                        help='')
    parser.add_argument('--P_Rz_amplitude', type=float, default=2 * math.pi * 0, metavar='P_Rz_amplitude',
                        help='')

    parser.add_argument('--save_path', type=str, default='./outs', metavar='save_path',
                        help='path to save log and model')

    parser.add_argument('--exp_name', type=str, default='test', metavar='save_path',
                        help='path to save log and model')

    parser.add_argument('--num_kpt', type=int, default=512, metavar='num_kpt',
                        help='')
    parser.add_argument('--dist_thres', type=float, default=1, metavar='num_kpt',
                        help='')

    parser.add_argument('--img_thres', type=float, default=0.95, metavar='img_thres',
                        help='')
    parser.add_argument('--pc_thres', type=float, default=0.95, metavar='pc_thres',
                        help='')

    parser.add_argument('--load_ckpt', type=str, default='none', metavar='save_path',
                        help='path to save log and model')

    parser.add_argument('--mode', type=str, default='none', metavar='save_path',
                        help='path to save log and model')

    args = parser.parse_args()

    opt = options.Options()
    opt.input_pt_num = args.input_pt_num

    test_dataset = kitti_pc_img_dataset(args.data_path, 'val', args.input_pt_num,
                                        P_tx_amplitude=args.P_tx_amplitude,
                                        P_ty_amplitude=args.P_ty_amplitude,
                                        P_tz_amplitude=args.P_tz_amplitude,
                                        P_Rx_amplitude=args.P_Rx_amplitude,
                                        P_Ry_amplitude=args.P_Ry_amplitude,
                                        P_Rz_amplitude=args.P_Rz_amplitude, num_kpt=args.num_kpt, is_front=False)

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False,
                                             drop_last=True, num_workers=args.num_workers)
    model = CurrI2P(opt)

    model = model.cuda()
    model.load_state_dict(torch.load(args.load_ckpt))

    t_diff, r_diff = test_acc_trans(model, testloader, args)