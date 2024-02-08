import torch.nn.functional as F
import torch

#计算部件分割的loss，实际为预测与真实的平均交并比
def compute_miou_loss(pred_seg_per_point, gt_seg_onehot):
    #计算部件分割向量与独热编码的点乘，并将结果相加，返回给dot
    dot = torch.sum(pred_seg_per_point * gt_seg_onehot, axis=1)
    #计算两个向量各元素之和与dot的差值
    denominator = torch.sum(pred_seg_per_point, axis=1) + torch.sum(gt_seg_onehot, axis=1) - dot
    #计算平均交并比
    mIoU = dot / (denominator + 1e-10)
    #返回loss，如果交并比很高，则预测很准确，则loss极低
    return torch.mean(1.0 - mIoU)

"""
#计算坐标的loss，用于计算ancsh和npcs坐标的loss
#接受四个参数，分别是预测坐标与真实坐标、部件数以及分割独热编码
"""
def compute_coorindate_loss(pred_coordinate_per_point, gt_coordinate_per_point, num_parts, gt_seg_onehot):
    loss_coordinate = 0.0
    """
    将预测坐标沿着第三个维度（dim=2）分割成多个部分，每个部分的大小为3。例如，如果
    pred_coordinate_per_point的形状是(B, N, 3K)，那么coordinate_splits就是一个包含K个
    形状为(B, N, 3)的张量的列表。
    """
    coordinate_splits = torch.split(pred_coordinate_per_point, split_size_or_sections=3, dim=2)
    #将分割独热编码沿着第三个维度分割，每个部分的大小为1
    mask_splits = torch.split(gt_seg_onehot, split_size_or_sections=1, dim=2)
    for i in range(num_parts):
        #计算张量coordinate_splits[i] - gt_coordinate_per_point的l2范数
        diff_l2 = torch.norm(coordinate_splits[i] - gt_coordinate_per_point, dim=2)
        #将独热编码与l2范数进行点乘后求平均数，并更新到坐标损失中
        loss_coordinate += torch.mean(mask_splits[i][:, :, 0] * diff_l2, axis = 1)
    return torch.mean(loss_coordinate, axis=0)

"""
计算向量的loss，用于各种预测指标，比如旋转轴、平移轴等
接受三个参数，预测向量、真实向量以及
"""
def compute_vect_loss(pred_vect_per_point, gt_vect_per_point, mask):
    #检查向量的第三个维度是否为1
    if pred_vect_per_point.shape[2] == 1:
        #将向量压缩为二维，即舍弃掉第三个维度
        pred_vect_per_point = torch.squeeze(pred_vect_per_point, dim=2)
        #计算预测坐标与真实坐标的绝对差，并乘以掩码
        diff_l2 = torch.abs(pred_vect_per_point - gt_vect_per_point) * mask
    else:
        #计算向量pred_vect_per_point - gt_vect_per_point的l2范数，并乘以掩码
        diff_l2 = torch.norm(pred_vect_per_point - gt_vect_per_point, dim=2) * mask
    #返回所有损失的平均值
    return torch.mean(torch.mean(diff_l2, axis=1), axis=0)