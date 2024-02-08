import os
import h5py
import itertools
import logging
import numpy as np
from time import time

from tools.utils import io

"""对于一个对象，获取一个3d边界框"""
def get_3d_bbox(scale, shift=np.zeros(3)):
    """
    Input:
        scale: [3]
        shift: [3]
    Return
        bbox_3d: [3, N]

    """

    bbox_3d = (
        #定义八个顶点，这八个顶点是以原点为中心，尺寸scale为边长的正方体
        np.array(
            [
                [scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
            ]
        )
        #加上偏移量
        + shift
    )
    return bbox_3d

"""
#检查一组点是否在给定的3d边界框内
"""
def pts_inside_box(pts, bbox):
    # pts: N x 3
    # bbox: 8 x 3 (-1, 1, 1), (1, 1, 1), (1, -1, 1), (-1, -1, 1), (-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1)
    #获得边界框的三个向量
    u1 = bbox[5, :] - bbox[4, :]
    u2 = bbox[7, :] - bbox[4, :]
    u3 = bbox[0, :] - bbox[4, :]
    #计算点到边界框的一个顶点的向量up
    up = pts - np.reshape(bbox[4, :], (1, 3))
    #计算向量up与边界框向量的点乘
    p1 = np.matmul(up, u1.reshape((3, 1)))
    p2 = np.matmul(up, u2.reshape((3, 1)))
    p3 = np.matmul(up, u3.reshape((3, 1)))
    p1 = np.logical_and(p1 > 0, p1 < np.dot(u1, u1))
    p2 = np.logical_and(p2 > 0, p2 < np.dot(u2, u2))
    p3 = np.logical_and(p3 > 0, p3 < np.dot(u3, u3))
    #返回一个bool数组，表示每个点是否在边界框内
    return np.logical_and(np.logical_and(p1, p2), p3)

"""
计算两个3d边界框的平均交并比，通过蒙特卡洛方法的思想计算
"""
def iou_3d(bbox1, bbox2, nres=50):
    #确定随机生成点的范围
    bmin = np.min(np.concatenate((bbox1, bbox2), 0), 0)
    bmax = np.max(np.concatenate((bbox1, bbox2), 0), 0)
    xs = np.linspace(bmin[0], bmax[0], nres)
    ys = np.linspace(bmin[1], bmax[1], nres)
    zs = np.linspace(bmin[2], bmax[2], nres)
    pts = np.array([x for x in itertools.product(xs, ys, zs)])
    ##计算在两个边界框范围内的点的数量，调用上面定义的pts_inside_box函数
    flag1 = pts_inside_box(pts, bbox1)
    flag2 = pts_inside_box(pts, bbox2)
    #计算同时在两个边界框范围内的点的数量
    intersect = np.sum(np.logical_and(flag1, flag2))
    # 计算至少在一个边界框范围内的点的数量
    union = np.sum(np.logical_or(flag1, flag2))
    if union == 0:
        return 1
    else:
        return intersect / float(union)

"""
计算两个向量之间的夹角
"""
def axis_diff_degree(v1, v2):
    v1 = v1.reshape(-1)
    v2 = v2.reshape(-1)
    r_diff = (
        np.arccos(np.clip(np.sum(v1 * v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), a_min=-1.0, a_max=1.0))
        * 180
        / np.pi
    )
    return min(r_diff, 180 - r_diff)

"""计算两条直线之间的距离，传入的参数分别是两个点以及两个方向向量"""
def dist_between_3d_lines(p1, e1, p2, e2):
    p1 = p1.reshape(-1)
    p2 = p2.reshape(-1)
    e1 = e1.reshape(-1)
    e2 = e2.reshape(-1)
    orth_vect = np.cross(e1, e2)
    p = p1 - p2

    if np.linalg.norm(orth_vect) == 0:
        dist = np.linalg.norm(np.cross(p, e1)) / np.linalg.norm(e1)
    else:
        dist = np.linalg.norm(np.dot(orth_vect, p)) / np.linalg.norm(orth_vect)

    return dist

"""ancsh效果评估"""
class ANCSHEvaluator:
    def __init__(self, cfg, combined_results_path, num_parts):
        start = time()
        self.cfg = cfg
        self.log = logging.getLogger('evaluator')
        self.log.info("Loading the data from results hdf5 file")
        self.f_combined = h5py.File(combined_results_path, "r+")
        self.instances = sorted(self.f_combined.keys())
        self.num_parts = num_parts
        self.results = {}
        self.log.info(f"Load the data: {time()-start} seconds")

        # err_rotation = []
        # err_translation = []

        # test_filter = ["0042", "0014"]
        # for ins in self.instances:
        #     if self.f_combined[ins]["is_valid"][0] == True:
        #         if ins.split('_')[1] in test_filter:
        #             continue
        #         if np.array(self.f_combined[ins]["err_rotation"][:]).sum() > 180:
        #             print(ins)
        #         err_rotation.append(self.f_combined[ins]["err_rotation"][:])
        #         err_translation.append(self.f_combined[ins]["err_translation"][:])
        # err_rotation = np.array(err_rotation)
        # err_translation = np.array(err_translation)

        # import pdb
        # pdb.set_trace()

    def process_ANCSH(self, gt=False, do_eval=True):
        self.do_eval = do_eval
        self.results = []
        # flag = False
        #遍历所有的实例
        for instance in self.instances:
            start = time()
            # if instance == "eyeglasses_0042_9_23":
            #     flag = True
            ins_combined = self.f_combined[instance]
            # if ins_combined["is_valid"][0] == True and not "12587" in instance:
            #如果组合结果是有效的，就从中获取组合之后的结果
            if ins_combined["is_valid"][0] == True:
                # Get the useful information from the combined_results
                prefix = 'pred_' if not gt else 'gt_'

                #逐点分割
                pred_seg_per_point = ins_combined[f"{prefix}seg_per_point"][:]
                #每个点的npcs坐标与naocs坐标
                pred_npcs_per_point = ins_combined[f"{prefix}npcs_per_point"][:]
                pred_naocs_per_point = ins_combined[f"{prefix}naocs_per_point"][:]

                #if do_eval表示如果是在评估模式下
                #获取真实的naocs坐标
                if do_eval:
                    gt_naocs_per_point = ins_combined["gt_naocs_per_point"][:]

                #获取一些指标
                pred_unitvec_per_point = ins_combined[f"{prefix}unitvec_per_point"][:]
                pred_heatmap_per_point = ins_combined[f"{prefix}heatmap_per_point"][:]
                pred_axis_per_point = ins_combined[f"{prefix}axis_per_point"][:]
                pred_joint_cls_per_point = ins_combined[f"{prefix}joint_cls_per_point"][:]

                #如果是在评估模式下，同时获得真实的指标
                if do_eval:
                    gt_unitvec_per_point = ins_combined["gt_unitvec_per_point"][:]
                    gt_heatmap_per_point = ins_combined["gt_heatmap_per_point"][:]
                    gt_axis_per_point = ins_combined["gt_axis_per_point"][:]
                    gt_joint_cls_per_point = ins_combined["gt_joint_cls_per_point"][:]

                    gt_npcs_scale = ins_combined["gt_npcs2cam_scale"][:]
                    gt_npcs_rt = ins_combined["gt_npcs2cam_rt"][:]
                    gt_naocs_scale = ins_combined["gt_naocs2cam_scale"][:]
                    gt_naocs_rt = ins_combined["gt_naocs2cam_rt"][:]

                #获取npcs缩放尺度以及npcs旋转矩阵
                pred_npcs_scale = ins_combined[f"{prefix}npcs2cam_scale"][:]
                pred_npcs_rt = ins_combined[f"{prefix}npcs2cam_rt"][:]

                if do_eval:
                    gt_jointIndex_per_point = gt_joint_cls_per_point

                    # Get the norm factors and corners used to calculate NPCS to calculate the 3dbbx
                    gt_norm_factors = ins_combined["gt_norm_factors"]
                    gt_corners = ins_combined["gt_norm_corners"]

                #创建存储评估结果的字典
                result = {
                    "joint_is_valid": [True],
                    "err_pose_scale": [],
                    "err_pose_volume": [],
                    "iou_cam_3dbbx": [],
                    "gt_cam_3dbbx": [],
                    "pred_cam_3dbbx": [],
                    "pred_joint_axis_naocs": [],
                    "pred_joint_pt_naocs": [],
                    "gt_joint_axis_naocs": [],
                    "gt_joint_pt_naocs": [],
                    "pred_joint_axis_cam": [],
                    "pred_joint_pt_cam": [],
                    "gt_joint_axis_cam": [],
                    "gt_joint_pt_cam": [],
                    "err_joint_axis": [],
                    "err_joint_line": [],
                }
                #遍历每一个部件
                for partIndex in range(self.num_parts):
                    #如果是在评估模式下
                    if do_eval:
                        #通过真实的npcs角点来计算scale，用于创建3d边界框
                        norm_factor = gt_norm_factors[partIndex]
                        corner = gt_corners[partIndex]
                        npcs_corner = np.zeros_like(corner)
                        # Calculatet the corners in npcs
                        npcs_corner[0] = (
                            np.array([0.5, 0.5, 0.5])
                            - 0.5 * (corner[1] - corner[0]) * norm_factor
                        )
                        npcs_corner[1] = (
                            np.array([0.5, 0.5, 0.5])
                            + 0.5 * (corner[1] - corner[0]) * norm_factor
                        )
                        # Calculate the gt bbx
                        gt_scale = npcs_corner[1] - npcs_corner[0]
                        #创建一个3d边界框，将真实的每一个部件框起来
                        gt_3dbbx = get_3d_bbox(gt_scale, shift=np.array([0.5, 0.5, 0.5]))
                    # Calculate the pred bbx
                    #得到每一个点所属部件的索引，np.where返回符合条件的数组下标
                    pred_part_points_index = np.where(
                        pred_seg_per_point == partIndex
                    )[0]
                    #得到与每一个部件索引相符的npcs坐标的点
                    centered_npcs = (
                        pred_npcs_per_point[
                            pred_part_points_index
                        ]
                        - 0.5
                    )
                    #对于这些点，其最大绝对值的二倍为尺度
                    pred_scale = 2 * np.max(abs(centered_npcs), axis=0)
                    #得到预测的每一个部件的3d边界框
                    pred_3dbbx = get_3d_bbox(pred_scale, np.array([0.5, 0.5, 0.5]))

                    #如果是在评估模式下
                    if do_eval:
                        # Record the pose scale and volume error
                        #记录预测的尺度误差，误差是通过计算预测与真实之间尺度乘以npcs尺度的差的l2范数得到
                        result["err_pose_scale"].append(
                            np.linalg.norm(
                                pred_scale * pred_npcs_scale[partIndex]
                                - gt_scale * gt_npcs_scale[partIndex]
                            )
                        )
                        # todo: whethere to take if it's smaller than 1, then it needs to consider the ratio
                        ratio_pose_volume = pred_scale[0] * pred_scale[1] * pred_scale[2] * pred_npcs_scale[partIndex]**3 / (
                                gt_scale[0] * gt_scale[1] * gt_scale[2] * gt_npcs_scale[partIndex]**3
                            )

                        # if ratio_pose_volume == 0:
                        #     import pdb
                        #     pdb.set_trace()
                        
                        #计算预测的体积误差
                        if ratio_pose_volume > 1:
                            result["err_pose_volume"].append(
                                ratio_pose_volume - 1
                            )
                        else:
                            result["err_pose_volume"].append(
                                1 / ratio_pose_volume - 1
                            )

                        # Calcualte the mean relative error（MRE，平均相对误差） for the parts
                        # This evaluation metric seems wierd, don't code it
                        # https://github.com/dragonlong/articulated-pose/blob/master/evaluation/eval_pose_err.py#L263
                        # https://github.com/dragonlong/articulated-pose/blob/master/evaluation/eval_pose_err.py#L320

                        # Calculatet the 3diou for each part
                        #计算真实3d边界框经尺度放缩后的3d边界框
                        gt_scaled_3dbbx = gt_3dbbx * gt_npcs_scale[partIndex]
                    #计算预测3d边界框经尺度放缩后的3d边界框
                    pred_scaled_3dbbx = pred_3dbbx * pred_npcs_scale[partIndex]
                    #如果是在评估模式下
                    if do_eval:
                        #计算真实npcs变换矩阵与真实3d边界框的点乘，得到真实的相机坐标下的3d边界框
                        gt_cam_3dbbx = (
                            np.dot(gt_npcs_rt[partIndex].reshape((4, 4), order='F')[:3, :3], gt_scaled_3dbbx.T).T
                            + gt_npcs_rt[partIndex].reshape((4, 4), order='F')[:3, 3].T
                        )
                    #计算预测npcs变换矩阵与预测3d边界框的点乘，得到真实的相机坐标下的3d边界框
                    pred_cam_3dbbx = (
                        np.dot(pred_npcs_rt[partIndex].reshape((4, 4), order='F')[:3, :3], pred_scaled_3dbbx.T).T
                        + pred_npcs_rt[partIndex].reshape((4, 4), order='F')[:3, 3].T
                    )
                    #如果在评估模式下，则计算真实与预测的相机坐标下的3d边界框的平均交并比并记录保存
                    if do_eval:
                        iou_cam_3dbbx = iou_3d(gt_cam_3dbbx, pred_cam_3dbbx)
                        result["gt_cam_3dbbx"].append(gt_cam_3dbbx)
                    result["pred_cam_3dbbx"].append(pred_cam_3dbbx)
                    if do_eval:
                        result["iou_cam_3dbbx"].append(iou_cam_3dbbx)

                    # Calculate the evaluation metric for the joints
                    # Calculate the scale and translation from naocs to npcs
                    pred_npcs = pred_npcs_per_point[
                        pred_part_points_index
                    ]
                    pred_naocs = pred_naocs_per_point[
                        pred_part_points_index
                    ]

                    #如果部件的索引为0
                    if partIndex == 0:
                        self.naocs_npcs_scale = np.std(np.mean(pred_npcs, axis=1)) / np.std(
                            np.mean(pred_naocs, axis=1)
                        )
                        self.naocs_npcs_translation = np.mean(
                            pred_npcs - self.naocs_npcs_scale * pred_naocs, axis=0
                        )

                    if partIndex >= 1:
                        # joint 0 is meaningless, the joint index starts from 1
                        thres_r = self.cfg.evaluation.thres_r
                        # Calculate the predicted joint info
                        #计算每个点的一个偏移量，用每个点的单位方向向量乘以预测的每个点的heatmap
                        pred_offset = (
                            pred_unitvec_per_point
                            * (1 - pred_heatmap_per_point.reshape(-1, 1))
                            * thres_r
                        )
                        #得到预测的关节点
                        pred_joint_pts = pred_naocs_per_point + pred_offset
                        #得到预测的点的关节索引，即返回每一个点预测的关节分类等于当前部件索引的下标
                        pred_joint_points_index = np.where(
                            pred_joint_cls_per_point == partIndex
                        )[0]
                        #求出所有预测的关节点相同索引的平均值，作为最终预测关节轴方向
                        pred_joint_axis = np.median(
                            pred_axis_per_point[pred_joint_points_index], axis=0
                        )
                        #求出所有预测的关节点的平均值，作为最终预测的关节点
                        pred_joint_pt = np.median(
                            pred_joint_pts[pred_joint_points_index], axis=0
                        )
                        #得到的两个结果分别是预测的关节轴方向以及关节点
                        result["pred_joint_axis_naocs"].append(pred_joint_axis)
                        result["pred_joint_pt_naocs"].append(pred_joint_pt)

                        # Convert the pred joint into camera coordinate from naocs -> npcs -> camera
                        #计算npcs坐标下的预测的关节点
                        temp_joint_pt_npcs = (
                            pred_joint_pt * self.naocs_npcs_scale
                            + self.naocs_npcs_translation
                        )
                        #计算相机坐标系下预测的关节点
                        pred_joint_pt_cam = (
                            np.dot(
                                pred_npcs_rt[0].reshape((4, 4), order='F')[:3, :3], pred_npcs_scale[0] * temp_joint_pt_npcs.T
                            ).T
                            + pred_npcs_rt[0].reshape((4, 4), order='F')[:3, 3]
                        )
                        #计算相机坐标系下预测的关节轴方向
                        pred_joint_axis_cam = np.dot(
                            pred_npcs_rt[partIndex].reshape((4, 4), order='F')[:3, :3], pred_joint_axis.T
                        ).T
                        #将结果保存
                        result["pred_joint_axis_cam"].append(pred_joint_axis_cam)
                        result["pred_joint_pt_cam"].append(pred_joint_pt_cam)
                        #如果是在评估模式下
                        if do_eval:
                            # Calculate the gt joint info
                            #计算真实偏移值
                            gt_offset = (
                                gt_unitvec_per_point
                                * (1 - gt_heatmap_per_point.reshape(-1, 1))
                                * thres_r
                            )
                            #计算真实关节点
                            gt_joint_pts = gt_naocs_per_point + gt_offset
                            #得到真实的点的关节索引
                            gt_joint_points_index = np.where(
                                gt_jointIndex_per_point == partIndex
                            )[0]
                            #如果索引长度为0，则警告这个关节实例是无效的
                            if len(gt_joint_points_index) == 0:
                                self.log.warning(f"Invalid JOINT instance {instance}")
                                result = {"joint_is_valid": [False]}
                                break
                            #计算真实的关节轴，为所有符合关节索引的点的均值
                            gt_joint_axis = np.median(
                                gt_axis_per_point[gt_joint_points_index], axis=0
                            )
                            #计算真实的关节点，为所有符合关节索引的关节点的均值
                            gt_joint_pt = np.median(gt_joint_pts[gt_joint_points_index], axis=0)
                            result["gt_joint_axis_naocs"].append(gt_joint_axis)
                            result["gt_joint_pt_naocs"].append(gt_joint_pt)
                            # Conver the gt joint into camera coordinate using the naocs pose, naocs -> camera
                            #计算相机坐标系下真实的关节点
                            gt_joint_pt_cam = (
                                np.dot(gt_naocs_rt.reshape((4, 4), order='F')[:3, :3], gt_naocs_scale * gt_joint_pt.T).T
                                + gt_naocs_rt.reshape((4, 4), order='F')[:3, 3]
                            )
                            #计算相机坐标系下真实的关节轴方向
                            gt_joint_axis_cam = np.dot(gt_naocs_rt.reshape((4, 4), order='F')[:3, :3], gt_joint_axis.T).T
                            result["gt_joint_axis_cam"].append(gt_joint_axis_cam)
                            result["gt_joint_pt_cam"].append(gt_joint_pt_cam)

                            # Calculate the error between the gt joints and pred joints in the camera coordinate
                            #调用axis_diff_degree函数，计算预测的关节轴方向和真实的关节轴方向之间的角度误差
                            err_joint_axis = axis_diff_degree(
                                gt_joint_axis_cam, pred_joint_axis_cam
                            )
                            """
                            调用dist_between_3d_lines函数，计算最终预测得到的关节轴（由关节点和轴方向确定一条直线）
                            和真实的关节轴之间的距离误差
                            """
                            err_joint_line = dist_between_3d_lines(
                                gt_joint_pt_cam,
                                gt_joint_axis_cam,
                                pred_joint_pt_cam,
                                pred_joint_axis_cam,
                            )
                            #记录所得到的这两个误差
                            result["err_joint_axis"].append(err_joint_axis)
                            result["err_joint_line"].append(err_joint_line)
                #将所有计算得到的结果储存起来
                self.results.append(result)
                # self.log.info(f"Ins {instance} is done in {time() - start} seconds")
            else:
                self.log.warning(f"Invalid POSE instance {instance}")
                self.results.append({})
        #保存并输出
        self.print_and_save()

    #保存并输出模型评估结果
    def print_and_save(self):
        # Print the mean errors for scale, volumeerr_pose_scale
        if self.do_eval:
            err_pose_scale = []
            err_pose_volume = []
            iou_cam_3dbbx =  []
            err_joint_axis = []
            err_joint_line = [] 
            for result in self.results:
                if not result == {}:
                    if result["joint_is_valid"][0] == True:
                        err_pose_scale.append(result["err_pose_scale"]) 
                        err_pose_volume.append(result["err_pose_volume"])
                        iou_cam_3dbbx.append(result["iou_cam_3dbbx"])
                        err_joint_axis.append(result["err_joint_axis"])
                        err_joint_line.append(result["err_joint_line"])
            mean_err_pose_scale = np.mean(err_pose_scale, axis=0)
            mean_err_pose_volume = np.mean(err_pose_volume, axis=0)
            self.log.info(f"Mean Error for pose scale: {mean_err_pose_scale}")
            self.log.info(f"Mean Error for pose volume: {mean_err_pose_volume}")

            # Print the mean iou for different parts
            mean_iou_cam_3dbbx = np.mean(iou_cam_3dbbx, axis=0)
            self.log.info(f"Mean iou for different parts is: {mean_iou_cam_3dbbx}")

            # Print the mean error for joints in the camera coordinate
            mean_err_joint_axis = np.mean(err_joint_axis, axis=0)
            mean_err_joint_line = np.mean(err_joint_line, axis=0)
            self.log.info(f"Mean joint axis error in camera coordinate (degree): {mean_err_joint_axis}")
            self.log.info(f"Mean joint axis line distance in camera coordinate (m): {mean_err_joint_line}")

        io.ensure_dir_exists(self.cfg.paths.evaluation.output_dir)
        f = h5py.File(
            os.path.join(self.cfg.paths.evaluation.output_dir, self.cfg.paths.evaluation.prediction_filename),
            "w"
        )
        for k, v in self.f_combined.attrs.items():
            f.attrs[k] = v
        if self.do_eval:
            f.attrs["err_pose_scale"] = mean_err_pose_scale
            f.attrs["err_pose_volume"] = mean_err_pose_volume
            f.attrs["iou_cam_3dbbx"] = mean_iou_cam_3dbbx
            f.attrs["err_joint_axis"] = mean_err_joint_axis
            f.attrs["err_joint_line"] = mean_err_joint_line

        self.log.info(f"Storing the evaluation results")
        for i, ins in enumerate(self.instances):
            # self.log.info(f"ins {ins} is storing")
            result = self.results[i]
            group = f.create_group(ins)
            for k, v in self.f_combined[ins].items():
                group.create_dataset(k, data=v, compression="gzip")
            for k, v in result.items():
                group.create_dataset(k, data=v, compression="gzip")
