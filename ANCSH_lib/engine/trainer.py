import os
import h5py
import logging
import numpy as np
from time import time

import torch
import torch.optim as optim
from torch.optim import optimizer
from torch.utils.tensorboard import SummaryWriter

from ANCSH_lib.model import ANCSH
from ANCSH_lib.data import ANCSHDataset
from ANCSH_lib import utils
from ANCSH_lib.utils import AvgRecorder, NetworkType
from tools.utils import io
# from tools.visualization import ANCSHVisualizer


class ANCSHTrainer:
    def __init__(self, cfg, data_path, network_type, num_parts):
        #加载配置文件
        self.cfg = cfg
        #创建一个日志记录器
        self.log = logging.getLogger("Network")
        # data_path is a dictionary {'train', 'test'}
        #指定训练的设备，如果配置文件中的指定使用CUDA，且CUDA可用，则使用，否则使用CPU
        if cfg.device == "cuda:0" and torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        self.device = device
        self.log.info(f"Using device {self.device}")
        #设置网络类型，由传入的参数指定
        self.network_type = NetworkType[network_type] if isinstance(network_type, str) else network_type

        """
        设置一些重要的超参数，包括：
        1.部件数量，由传入参数指定
        2.最大训练epoch，由配置文件指定
        3.加载训练模型，调用build_model()函数获取
        4.将模型加载到训练的设备CUDA/CPU上
        5.创建一个优化器，使用
        """
        self.num_parts = num_parts
        self.max_epochs = cfg.network.max_epochs
        self.model = self.build_model()
        self.model.to(device)
        self.log.info(f"Below is the network structure:\n {self.model}")
        #创建一个优化器，使用torch中提供的Adam优化算法，使用模型中所有的参数
        #继承的torch.nn.module.parameters会返回包含模型所有参数的一个迭代器
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=cfg.network.lr, betas=(0.9, 0.99)
        )
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.7)
        #设置数据路径
        self.data_path = data_path
        #初始化写入器，用于记录训练信息
        self.writer = None
        #初始化train_loader和test_loader
        self.train_loader = None
        self.test_loader = None
        #调用配置文件中的eval_only方法来初始化data_loader
        self.init_data_loader(self.cfg.eval_only)
        #初始化用于存储测试结果的向量
        self.test_result = None

    #构建模型
    def build_model(self):
        model = ANCSH(self.network_type, self.num_parts)
        return model

    #初始化data_loader
    def init_data_loader(self, eval_only):
        #如果不只是需要评估，先加载训练数据的data_loader
        if not eval_only:
            """
            调用torch.utils.data.DataLoader方法创建一个数据加载器，需要传入参数为
            1.需要加载的数据集，这里调用了dataset.py中创建的ANCSH数据集
            2.部件数量，通过配置文件获得
            3.每一个batch的batchsize
            4.shuffle为训练每一个epoch时是否打乱数据
            5.num_workers为调用子进程的数量
            """
            self.train_loader = torch.utils.data.DataLoader(
                ANCSHDataset(
                    self.data_path["train"], num_points=self.cfg.network.num_points
                ),
                batch_size=self.cfg.network.batch_size,
                shuffle=True,
                num_workers=self.cfg.network.num_workers,
            )
            #记录加载信息
            self.log.info(f'Num {len(self.train_loader)} batches in train loader')

        #加载测试数据的data_loader，与训练数据不同，不需要打乱数据
        self.test_loader = torch.utils.data.DataLoader(
            ANCSHDataset(
                self.data_path["test"], num_points=self.cfg.network.num_points
            ),
            batch_size=self.cfg.network.batch_size,
            shuffle=False,
            num_workers=self.cfg.network.num_workers,
        )
        self.log.info(f'Num {len(self.test_loader)} batches in test loader')

    #训练一个epoch
    def train_epoch(self, epoch):
        self.log.info(f'>>>>>>>>>>>>>>>> Train Epoch {epoch} >>>>>>>>>>>>>>>>')

        #将模型设置为训练模式
        #继承的torch.nn.module.train()，仅仅当模型中有dropout和batchnormal有效
        self.model.train()

        #初始化一些用于记录时间的对象或变量
        #AvgRecorder()类定义在utils.py，是一个更新并记录一些变量的类。
        iter_time = AvgRecorder()
        io_time = AvgRecorder()
        to_gpu_time = AvgRecorder()
        network_time = AvgRecorder()
        start_time = time()
        end_time = time()
        remain_time = ''

        #初始化一个字典，用于记录训练epoch的loss
        epoch_loss = {
            'total_loss': AvgRecorder()
        }

        # if self.train_loader.sampler is not None:
        #     self.train_loader.sampler.set_epoch(epoch)
        #循环遍历每一个batch
        for i, (camcs_per_point, gt_dict, id) in enumerate(self.train_loader):
            io_time.update(time() - end_time)
            # Move the tensors to the device
            s_time = time()
            #将相机坐标系下的逐点坐标加载到指定的设备
            camcs_per_point = camcs_per_point.to(self.device)
            gt = {}
            #将其它真实的逐点坐标写入gt这个列表
            for k, v in gt_dict.items():
                gt[k] = v.to(self.device)
            to_gpu_time.update(time() - s_time)

            #获取损失
            # Get the loss
            s_time = time()
            #传入相机坐标系下的逐点坐标，需要调用network.py中的forward方法得到预测的各个指标
            pred = self.model(camcs_per_point)
            #调用network.py中的losses方法，计算预测与真实之间的损失，返回的字典记录各项损失
            loss_dict = self.model.losses(pred, gt)
            network_time.update(time() - s_time)

            #定义一个损失张量
            loss = torch.tensor(0.0, device=self.device)
            #从配置文件中得到损失权重
            loss_weight = self.cfg.network.loss_weight
            # use different loss weight to calculate the final loss
            #根据给定搞得损失权重计算并更新损失
            for k, v in loss_dict.items():
                if k not in loss_weight:
                    raise ValueError(f"No loss weight for {k}")
                loss += loss_weight[k] * v

            #计算各项损失，并更新总损失，epoch_loss是一个损失字典，定义在126行
            """
            这段代码在做这样一个事，之前定义的epoch_loss只有一个键值对，就是total_loss，通过
            调用network.losses返回的是一个含有各项损失的loss_dict，遍历这个dict，如果epoch_loss
            里面没有对应的键，就将键添加到epoch_loss里，并记录对应的值
            """
            for k, v in loss_dict.items():
                if k not in epoch_loss.keys():
                    epoch_loss[k] = AvgRecorder()
                epoch_loss[k].update(v)
            epoch_loss['total_loss'].update(loss)

            """
            self.optimizer是模型的一个优化器，定义在52行，使用Adam算法进行后向传播更新参数
            1.首先将模型的全部梯度清零，pytorch中，梯度是累加的，对于每一个batch，需要先将之前
            batch的梯度清零。
            2.通过后向传播来计算损失函数关于所有参数的梯度
            3.调用之前创建的优化器，使用Adam算法进行更新参数
            """
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #以下几行代码在更新迭代时间并计算剩余时间
            # time and print
            current_iter = epoch * len(self.train_loader) + i + 1
            max_iter = (self.max_epochs + 1) * len(self.train_loader)
            remain_iter = max_iter - current_iter

            iter_time.update(time() - end_time)
            end_time = time()

            remain_time = remain_iter * iter_time.avg
            remain_time = utils.duration_in_hours(remain_time)

        #将当前学习率（learning rate）添加到tensorboard中
        #self.optimizer.param_groups[0]["lr"]获取当前学习率，epoch为当前的epoch
        self.writer.add_scalar("lr", self.optimizer.param_groups[0]["lr"], epoch)
        # self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], epoch)
        # self.scheduler.step()
        # Add the loss values into the tensorboard
        #将当前的总损失和各项损失添加到tensorboard中
        for k, v in epoch_loss.items():
            if k == "total_loss":
                self.writer.add_scalar(f"{k}", epoch_loss[k].avg, epoch)
            else:
                self.writer.add_scalar(f"loss/{k}", epoch_loss[k].avg, epoch)

        #按照配置文件中给定的记录日志的频率（实际应该是每10个epoch），记录日志信息
        if epoch % self.cfg.train.log_frequency == 0:
            loss_log = ''
            for k, v in epoch_loss.items():
                loss_log += '{}: {:.5f}  '.format(k, v.avg)
            #输出当前epoch训练的各项信息，包括各种耗费时间、剩余时间、各项平均损失等
            self.log.info(
                'Epoch: {}/{} Loss: {} io_time: {:.2f}({:.4f}) to_gpu_time: {:.2f}({:.4f}) network_time: {:.2f}({:.4f}) \
                duration: {:.2f} remain_time: {}'
                    .format(epoch, self.max_epochs, loss_log, io_time.sum, io_time.avg, to_gpu_time.sum,
                            to_gpu_time.avg, network_time.sum, network_time.avg, time() - start_time, remain_time))

    #对模型和优化效果进行评估
    def eval_epoch(self, epoch, save_results=False):
        self.log.info(f'>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        val_error = {
            'total_loss': AvgRecorder()
        }
        #如果需要保存结果
        if save_results:
            io.ensure_dir_exists(self.cfg.paths.network.test.output_dir)
            inference_path = os.path.join(self.cfg.paths.network.test.output_dir,
                                          self.network_type.value + '_' + self.cfg.paths.network.test.inference_result)
            self.test_result = h5py.File(inference_path, "w")
            self.test_result.attrs["network_type"] = self.network_type.value

        # test the model on the val set and write the results into tensorboard
        #将模型设置为评估模式
        self.model.eval()
        #评估模式下不需要计算梯度
        with torch.no_grad():
            start_time = time()
            #从测试数据加载器中加载数据
            for i, (camcs_per_point, gt_dict, id) in enumerate(self.test_loader):
                # 将相机坐标系下逐点坐标加载至设备
                camcs_per_point = camcs_per_point.to(self.device)
                gt = {}
                #将真实的各项数据加载至设备
                for k, v in gt_dict.items():
                    gt[k] = v.to(self.device)
                # 传入相机坐标系下的逐点坐标，需要调用network.py中的forward方法得到预测的各个指标
                pred = self.model(camcs_per_point)
                #如果需要保存数据则将数据保存
                if save_results:
                    self.save_results(pred, camcs_per_point, gt, id)
                #调用network.py中的losses方法，计算预测与真实之间的损失，返回的字典记录各项损失
                loss_dict = self.model.losses(pred, gt)
                #通过配置文件获取损失权重
                loss_weight = self.cfg.network.loss_weight
                loss = torch.tensor(0.0, device=self.device)
                # use different loss weight to calculate the final loss
                #按照给定的损失权重计算各项损失
                for k, v in loss_dict.items():
                    if k not in loss_weight:
                        raise ValueError(f"No loss weight for {k}")
                    loss += loss_weight[k] * v
                """
                这段代码在做这样一个事，之前定义的val_loss只有一个键值对，就是total_loss，通过
                调用network.losses返回的是一个含有各项损失的loss_dict，遍历这个dict，如果val_loss
                里面没有对应的键，就将键添加到val_loss里，并记录对应的值
                """
                # Used to calculate the avg loss
                for k, v in loss_dict.items():
                    if k not in val_error.keys():
                        val_error[k] = AvgRecorder()
                    val_error[k].update(v)
                val_error['total_loss'].update(loss)
        # write the val_error into the tensorboard
        #如果写入器writer不为空，则写入评估的各项损失
        if self.writer is not None:
            for k, v in val_error.items():
                self.writer.add_scalar(f"val_error/{k}", val_error[k].avg, epoch)
        #将各项损失信息拼接为一个字符串并准备输出
        loss_log = ''
        for k, v in val_error.items():
            loss_log += '{}: {:.5f}  '.format(k, v.avg)
        #向日志写入各种信息，包括当前的epoch、各项平均损失以及持续时间
        self.log.info(
            'Eval Epoch: {}/{} Loss: {} duration: {:.2f}'
                .format(epoch, self.max_epochs, loss_log, time() - start_time))
        if save_results:
            self.test_result.close()
        #返回各项损失的字典
        return val_error

    #模型训练过程部分
    def train(self, start_epoch=0):
        #将模型设置为训练模式
        self.model.train()
        #创建一个tensorboard写入器，用于记录训练过程中的信息
        self.writer = SummaryWriter(self.cfg.paths.network.train.output_dir)
        #确保输出目录存在
        io.ensure_dir_exists(self.cfg.paths.network.train.output_dir)
        #初始时，将best_model设置为none
        best_model = None
        #将最好的结果设置为无穷大
        best_result = np.inf
        #逐个训练每一个epoch，在这里是1200个epoch
        for epoch in range(start_epoch, self.max_epochs + 1):
            #调用train_epoch(epoch)函数，训练一个epoch
            self.train_epoch(epoch)
            #如果是配置文件中给定的频率，或是最后一个epoch
            if epoch % self.cfg.train.save_frequency == 0 or epoch == self.max_epochs:
                # Save the model
                #首先保存模型，包括当前模型状态与更新的参数
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    os.path.join(self.cfg.paths.network.train.output_dir,
                                 self.cfg.paths.network.train.model_filename % epoch),
                )
                #调用eval_epoch(epoch)函数，评估当前的各项损失
                val_error = self.eval_epoch(epoch)
                #更新best_model，当总损失小于当前最佳损失时进行更新
                if best_model is None or val_error["total_loss"].avg < best_result:
                    best_model = {
                        #需要更新的信息有epoch数、模型状态、Adam算法更新的所有参数
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    }
                    #更新最佳损失
                    best_result = val_error["total_loss"].avg
                    #保存最优模型
                    torch.save(
                        best_model,
                        os.path.join(self.cfg.paths.network.train.output_dir,
                                     self.cfg.paths.network.train.best_model_filename)
                    )
        self.writer.close()

    #获得最新的模型的路径
    def get_latest_model_path(self, with_best=False):
        train_result_dir = os.path.dirname(self.cfg.paths.network.train.output_dir)
        #调用utils.py的utils.get_latest_file_with_datetime函数，获取模型的文件名与文件夹
        folder, filename = utils.get_latest_file_with_datetime(train_result_dir,
                                                               self.network_type.value + '_', ext='.pth')
        #按照获得的文件名与文件夹获取模型路径
        model_path = os.path.join(train_result_dir, folder, filename)
        if with_best:
            model_path = os.path.join(train_result_dir, folder, self.cfg.paths.network.train.best_model_filename)
        return model_path

    #测试函数
    def test(self, inference_model=None):
        #如果存在已训练好的预训练模型，则加载模型路径
        if not inference_model or not io.file_exist(inference_model):
            inference_model = self.get_latest_model_path(with_best=True)
        if not io.file_exist(inference_model):
            raise IOError(f'Cannot open inference model {inference_model}')
        # Load the model
        self.log.info(f"Load model from {inference_model}")
        #按照给定路径加载模型
        checkpoint = torch.load(inference_model, map_location=self.device)
        epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        #评估模型
        self.eval_epoch(epoch, save_results=True)

        # # create visualizations of evaluation results
        # if self.cfg.test.render.render:
        #     export_dir = os.path.join(self.cfg.paths.network.test.output_dir,
        #                               self.cfg.paths.network.test.visualization_folder)
        #     io.ensure_dir_exists(export_dir)
        #     inference_path = os.path.join(self.cfg.paths.network.test.output_dir,
        #                                   self.network_type.value + '_' + self.cfg.paths.network.test.inference_result)
        #     with h5py.File(inference_path, "r") as inference_h5:
        #         visualizer = ANCSHVisualizer(inference_h5, network_type=self.network_type)
        #         visualizer.render(self.cfg.test.render.show, export=export_dir, export_mesh=self.cfg.test.render.export)

    #保存结果
    def save_results(self, pred, camcs_per_point, gt, id):
        # Save the results and gt into hdf5 for further optimization
        batch_size = pred["seg_per_point"].shape[0]
        for b in range(batch_size):
            group = self.test_result.create_group(f"{id[b]}")
            group.create_dataset(
                "camcs_per_point",
                data=camcs_per_point[b].detach().cpu().numpy(),
                compression="gzip",
            )

            # save prediction results
            raw_segmentations = pred['seg_per_point'][b].detach().cpu().numpy()
            raw_npcs_points = pred['npcs_per_point'][b].detach().cpu().numpy()
            segmentations, npcs_points = utils.get_prediction_vertices(raw_segmentations, raw_npcs_points)
            group.create_dataset('pred_seg_per_point', data=segmentations, compression="gzip")
            group.create_dataset('pred_npcs_per_point', data=npcs_points, compression="gzip")
            if self.network_type == NetworkType.ANCSH:
                raw_naocs_points = pred['naocs_per_point'][b].detach().cpu().numpy()
                _, naocs_points = utils.get_prediction_vertices(raw_segmentations, raw_naocs_points)
                raw_joint_associations = pred['joint_cls_per_point'][b].detach().cpu().numpy()
                joint_associations = np.argmax(raw_joint_associations, axis=1)
                joint_axes = pred['axis_per_point'][b].detach().cpu().numpy()
                point_heatmaps = pred['heatmap_per_point'][b].detach().cpu().numpy().flatten()
                unit_vectors = pred['unitvec_per_point'][b].detach().cpu().numpy()

                group.create_dataset('pred_naocs_per_point', data=naocs_points, compression="gzip")
                group.create_dataset('pred_joint_cls_per_point', data=joint_associations, compression="gzip")
                group.create_dataset('pred_axis_per_point', data=joint_axes, compression="gzip")
                group.create_dataset('pred_heatmap_per_point', data=point_heatmaps, compression="gzip")
                group.create_dataset('pred_unitvec_per_point', data=unit_vectors, compression="gzip")

            # Save the gt
            for k, v in gt.items():
                group.create_dataset(
                    f"gt_{k}", data=gt[k][b].detach().cpu().numpy(), compression="gzip"
                )
    #断点续训
    def resume_train(self, model_path=None):
        #如果没有给定训练路径，就调用get_latest_model_path函数搜索最新的模型的路径
        if not model_path or not io.file_exist(model_path):
            model_path = self.get_latest_model_path()
        # Load the model
        #检查按照给定路径找到的文件是否费控
        if io.is_non_zero_file(model_path):
            #若非空则加载当前模型
            checkpoint = torch.load(model_path, map_location=self.device)
            epoch = checkpoint["epoch"]
            self.log.info(f"Continue training with model from {model_path} at epoch {epoch}")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.model.to(self.device)
        else:
            epoch = 0
        #继续训练
        self.train(epoch)
