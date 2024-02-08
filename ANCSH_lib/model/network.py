import torch
import torch.nn as nn
import torch.nn.functional as F
from ANCSH_lib.model.backbone import PointNet2
from ANCSH_lib.model import loss
from ANCSH_lib.utils import NetworkType

#定义了ANCSH模型的网络结构
"""
对于使用pytorch来定义网络结构的每一个类，都需要将torch.nn.module作为基类继承
"""
class ANCSH(nn.Module):
    def __init__(self, network_type, num_parts):
        #继承基类的构造函数
        super().__init__()
        #获取当前的网络类型ANCSH/NPCS
        self.network_type = NetworkType[network_type] if isinstance(network_type, str) else network_type

        # Define the shared PN++
        #使用pointnet++网络
        self.backbone = PointNet2()
        #若为ANCSH
        if self.network_type == NetworkType.ANCSH:
            """
            论文中的第一个module，用于部件分割
            """
            # segmentation branch
            #分割部分网络结构
            #一维卷积层，输入通道128，输出通道为部件数，在eyeglasses数据集中为3，卷积核大小为1，不填充
            self.seg_layer = nn.Conv1d(128, num_parts, kernel_size=1, padding=0)
            # NPCS branch
            #定义NPCS层网络结构，这里的nn.sequential是一个时序容器，允许网络结构按照传入模块的先后排序
            self.npcs_layer = nn.Sequential(
                #定义了两层一维卷积层
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.Conv1d(128, 3 * num_parts, kernel_size=1, padding=0),
            )
            """
            论文中的第二个module，将NPCS坐标投射到NAOCS坐标
            """
            # NAOCS scale and translation
            #NAOCS尺度变换层，为一维卷积层
            self.scale_layer = nn.Conv1d(
                128, 1 * num_parts, kernel_size=1, padding=0
            )
            #NAOCS平移变换层，为一维卷积层
            self.trans_layer = nn.Conv1d(
                128, 3 * num_parts, kernel_size=1, padding=0
            )
            """
            论文中的第三部分，预测关节体的联合参数
            我们关心的是两种关节体，即旋转关节和平移关节
            前者需要预测旋转轴和旋转点，后者需要预测的是平移轴
            """
            #定义参数预测的网络结构
            # Joint parameters
            #参数预测的特征提取层
            self.joint_feature_layer = nn.Sequential(
                #一维卷积层
                nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
                #对上一个卷积层作标准化batchnormal操作
                nn.BatchNorm1d(128),
                #非线性激活层，参数为是否进行覆盖运算
                nn.ReLU(True),
                #随机丢弃层，按照传入的概率将张量中的元素置为0
                nn.Dropout(0.5),
                nn.Conv1d(128, 128, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Dropout(0.5),
            )
            # Joint UNitVec, heatmap, joint_cls
            #预测旋转轴
            self.axis_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)
            #预测平移轴？
            self.unitvec_layer = nn.Conv1d(128, 3, kernel_size=1, padding=0)
            #预测heatmap
            self.heatmap_layer = nn.Conv1d(128, 1, kernel_size=1, padding=0)
            #预测关节分类
            self.joint_cls_layer = nn.Conv1d(
                128, num_parts, kernel_size=1, padding=0
            )
        #如果是NPCS，只需要作分割和预测npcs坐标
        elif self.network_type == NetworkType.NPCS:
            # segmentation branch
            self.seg_layer = nn.Conv1d(128, num_parts, kernel_size=1, padding=0)
            # NPCS branch
            self.npcs_layer = nn.Conv1d(
                128, 3 * num_parts, kernel_size=1, padding=0
            )
        #如果输入为其它则报错
        else:
            raise ValueError(f"No implementation for the network type {self.network_type.value}")

    #定义前向传播的结构，以及计算出npcs坐标变换后的naocs坐标
    def forward(self, input):
        #加载pointnet++对输入特征进行处理
        features = self.backbone(input)
        #将得到的特征作为输入，输入分割预测层，并将结果转置，得到预测的分割坐标
        pred_seg_per_point = self.seg_layer(features).transpose(1, 2)
        #将得到的特征作为输入，输入npcs坐标预测层，并将结果转置，得到预测的npcs坐标
        pred_npcs_per_point = self.npcs_layer(features).transpose(1, 2)
        #如果是ANCSH
        if self.network_type == NetworkType.ANCSH:
            #得到预测的尺度变换
            pred_scale_per_point = self.scale_layer(features).transpose(1, 2)
            #得到逐点平移变换
            pred_trans_per_point = self.trans_layer(features).transpose(1, 2)
            #得到提取的特征
            joint_features = self.joint_feature_layer(features)
            #以下为分别得到的旋转轴、平行轴、热图预测
            pred_axis_per_point = self.axis_layer(joint_features).transpose(1, 2)
            pred_unitvec_per_point = self.unitvec_layer(joint_features).transpose(1, 2)
            pred_heatmap_per_point = self.heatmap_layer(joint_features).transpose(1, 2)
            pred_joint_cls_per_point = self.joint_cls_layer(joint_features).transpose(1, 2)

        # Process the predicted things
        #分割预测之后再接一个softmax非线性激活层
        pred_seg_per_point = F.softmax(pred_seg_per_point, dim=2)
        #逐点平移变换之后再接一个sigmoid非线性激活层
        pred_npcs_per_point = F.sigmoid(pred_npcs_per_point)

        if self.network_type == NetworkType.ANCSH:
            #在之前预测的结果之后接一堆各种各样非线性激活层
            pred_scale_per_point = F.sigmoid(pred_scale_per_point)
            pred_trans_per_point = F.tanh(pred_trans_per_point)

            pred_heatmap_per_point = F.sigmoid(pred_heatmap_per_point)
            pred_unitvec_per_point = F.tanh(pred_unitvec_per_point)
            pred_axis_per_point = F.tanh(pred_axis_per_point)
            pred_joint_cls_per_point = F.softmax(pred_joint_cls_per_point, dim=2)

            # Calculate the NAOCS per point
            #将scale扩展为3×3尺寸，以便与npcs_per_point和trans_per_point进行乘法和加法运算
            pred_scale_per_point_repeat = pred_scale_per_point.repeat(1, 1, 3)
            #将npcs坐标按尺度缩放并平移后得到naocs坐标
            pred_naocs_per_point = (
                    pred_npcs_per_point * pred_scale_per_point_repeat + pred_trans_per_point
            )
        #预测结果为分割以及npcs坐标
        pred = {
            "seg_per_point": pred_seg_per_point,
            "npcs_per_point": pred_npcs_per_point,
        }

        #如果网络类型为ANCSH，则将预测结果添加如下的选项
        if self.network_type == NetworkType.ANCSH:
            pred.update(
                {
                    "heatmap_per_point": pred_heatmap_per_point,
                    "unitvec_per_point": pred_unitvec_per_point,
                    "axis_per_point": pred_axis_per_point,
                    "joint_cls_per_point": pred_joint_cls_per_point,
                    "scale_per_point": pred_scale_per_point,
                    "trans_per_point": pred_trans_per_point,
                    "naocs_per_point": pred_naocs_per_point,
                }
            )

        return pred

    #计算损失
    def losses(self, pred, gt):
        # The returned loss is a value
        #获取部件的数量
        num_parts = pred["seg_per_point"].shape[2]
        # Convert the gt['seg_per_point'] into gt_seg_onehot B*N*K
        #将gt_seg_per_point转换为独热编码，维度为部件数
        gt_seg_onehot = F.one_hot(gt["seg_per_point"].long(), num_classes=num_parts)
        # pred['seg_per_point']: B*N*K, gt_seg_onehot: B*N*K
        #以下计算分割与npcs坐标的loss，损失函数定义在loss.py
        #计算部件分割的loss
        seg_loss = loss.compute_miou_loss(pred["seg_per_point"], gt_seg_onehot)
        # pred['npcs_per_point']: B*N*3K, gt['npcs_per_point']: B*N*3, gt_seg_onehot: B*N*K
        #计算预测npcs坐标的loss
        npcs_loss = loss.compute_coorindate_loss(
            pred["npcs_per_point"],
            gt["npcs_per_point"],
            num_parts=num_parts,
            gt_seg_onehot=gt_seg_onehot,
        )
        #如果网络类型为ancsh
        if self.network_type == NetworkType.ANCSH:
            # pred['naocs_per_point']: B*N*3K, gt['naocs_per_point']: B*N*3, gt_seg_onehot: B*N*K
            #计算预测的ancsh坐标的loss
            naocs_loss = loss.compute_coorindate_loss(
                pred["naocs_per_point"],
                gt["naocs_per_point"],
                num_parts=num_parts,
                gt_seg_onehot=gt_seg_onehot,
            )

            # Get the useful joint mask, gt['joint_cls_per_point'] == 0 means that
            # the point doesn't have a corresponding joint
            # B*N
            #创建真实数据的关节掩码
            gt_joint_mask = (gt["joint_cls_per_point"] > 0).float()
            # Get the heatmap and unitvec map, the loss should only be calculated for revolute joint
            #创建一个全为0的向量来作为revolute掩码
            gt_revolute_mask = torch.zeros_like(gt["joint_cls_per_point"]) == 1
            #找出gt["joint_type"]中第一行中等于1的元素的索引
            revolute_index = torch.where(gt["joint_type"][0] == 1)[0]
            #检查gt["joint_type"]第一列是否全为-1，不是的话会抛出异常
            assert (gt["joint_type"][:, 0] == -1).all() == True
            #这个循环遍历所有旋转关节的索引，然后更新gt_revolute_mask。如果gt["joint_cls_per_point"]
            #等于当前索引，那么对应的gt_revolute_mask会被设置为True。
            for i in revolute_index:
                gt_revolute_mask = torch.logical_or(gt_revolute_mask, (gt["joint_cls_per_point"] == i))
            #转换为浮点类型
            gt_revolute_mask = gt_revolute_mask.float()
            # pred['heatmap_per_point']: B*N*1, gt['heatmap_per_point']: B*N, gt_revolute_mask: B*N
            #使用掩码来计算热图，也就是旋转点的loss
            heatmap_loss = loss.compute_vect_loss(
                pred["heatmap_per_point"], gt["heatmap_per_point"], mask=gt_revolute_mask
            )
            # pred['unitvec_per_point']: B*N*3, gt['unitvec_per_point']: B*N*3, gt_revolute_mask: B*N
            #使用掩码来计算旋转关节转轴方向的loss
            unitvec_loss = loss.compute_vect_loss(
                pred["unitvec_per_point"], gt["unitvec_per_point"], mask=gt_revolute_mask
            )
            # pred['axis_per_point]: B*N*3, gt['axis_per_point']: B*N*3, gt_joint_mask: B*N
            #计算平移关节的平移轴的loss
            axis_loss = loss.compute_vect_loss(
                pred["axis_per_point"], gt["axis_per_point"], mask=gt_joint_mask
            )

            # Conver the gt['joint_cls_per_point'] into gt_joint_cls_onehot B*N*K
            gt_joint_cls_onehot = F.one_hot(
                gt["joint_cls_per_point"].long(), num_classes=num_parts
            )
            #计算关节分类的loss？
            joint_loss = loss.compute_miou_loss(
                pred["joint_cls_per_point"], gt_joint_cls_onehot
            )
        #返回计算的loss
        loss_dict = {
            "seg_loss": seg_loss,
            "npcs_loss": npcs_loss,
        }
        #如果网络类型是ancsh，则多返回计算的几个loss
        if self.network_type == NetworkType.ANCSH:
            loss_dict.update(
                {
                    "naocs_loss": naocs_loss,
                    "heatmap_loss": heatmap_loss,
                    "unitvec_loss": unitvec_loss,
                    "axis_loss": axis_loss,
                    "joint_loss": joint_loss,
                }
            )

        return loss_dict
