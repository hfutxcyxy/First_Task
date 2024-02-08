#python目录与文件处理库
import os
os.environ["CUDA_VISIBLE_DEVICES"] = 'gpu_ids'
import h5py
#记录程序运行时的信息
import logging
import torch
from time import time

import hydra
from hydra.utils import get_original_cwd
#处理配置
from omegaconf import DictConfig, OmegaConf

from ANCSH_lib import ANCSHTrainer, utils
from ANCSH_lib.utils import NetworkType
from tools.utils import io

#创建一个名为‘train’的日志记录器，记录程序运行过程中的错误信息、警告信息、调试信息
log = logging.getLogger('train')

"""
装饰器，用于标记下面的主函数，这个装饰器的作用是告诉Hydra库，我们要使用的配置文件的路径是"configs"，
配置文件的名字是"network"。Hydra库会在函数main(cfg: DictConfig)被调用之前，先去读取配置文件，
然后把配置文件的内容以DictConfig对象的形式传递给main(cfg: DictConfig)函数。这样我们就可以
在main(cfg: DictConfig)函数中，通过cfg参数来访问配置文件的内容了。
"""
@hydra.main(config_path="configs", config_name="network")
def main(cfg: DictConfig):
    #更新配置文件中的"paths.result_dir"字段为当前的绝对路径和工作目录路径
    OmegaConf.update(cfg, "paths.result_dir", io.to_abs_path(cfg.paths.result_dir, get_original_cwd()))
    #设置训练数据的路径，如果train.input_data存在则使用，不存在则使用默认的路径
    train_path = cfg.train.input_data if io.file_exist(cfg.train.input_data) else cfg.paths.preprocess.output.train
    #设置测试数据的路径，如果cfg.test.split == 'val'，则使用cfg.paths.preprocess.output.val
    test_path = cfg.paths.preprocess.output.val if cfg.test.split == 'val' else cfg.paths.preprocess.output.test
    #更新测试数据的路径，如果cfg.test.input_data存在则使用，否则就保持不变
    test_path = cfg.test.input_data if io.file_exist(cfg.test.input_data) else test_path
    #创建一个字典类型，用于存储训练数据与测试数据的路径，使得训练key、测试key与对应的路径相符
    data_path = {"train": train_path, "test": test_path}
    #获取对象的部件数
    num_parts = utils.get_num_parts(train_path)
    # test_num_parts = utils.get_num_parts(test_path)
    # assert num_parts == test_num_parts
    log.info(f'Instances in dataset have {num_parts} parts')
    #设置网络类型ANCSH/NPCS，其中NetworkType是一个类，在ANCSH_lib.utils中定义
    network_type = NetworkType[cfg.network.network_type]
    #设置随机种子
    utils.set_random_seed(cfg.random_seed)
    #torch.set_deterministic(True)
    #固定随机源
    torch.backends.cudnn.deterministic = True
    #设置环境变量“CUBLAS_WORKSPACE_CONFIG” 的值，用以配置英伟达的cuBLAS库，提供许多线性代数操作
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #设置一个ANCSH的trainer，需要用到的参数有配置文件、数据的路径、网络类型以及对象中部件的个数
    trainer = ANCSHTrainer(
        cfg=cfg,
        data_path=data_path,
        network_type=network_type,
        num_parts=num_parts,
    )
    """
    以下分支决定是否执行训练或是测试，检查配置文件中的eval_only字段，如果为false，则执行训练，否则
    执行测试。
    """
    if not cfg.eval_only:
        #输出一条信息，包括训练数据路径以及测试数据路径
        log.info(f'Train on {train_path}, validate on {test_path}')
        #检查配置文件的train.continuous字段，若为假，则从头开始训练
        if not cfg.train.continuous:
            trainer.train()
        #否则就从指定的预训练模型开始训练
        else:
            trainer.resume_train(cfg.train.input_model)
        #训练完毕，执行测试
        trainer.test()
    #否则只需要执行测试
    else:
        #输出一条信息，包括测试数据路径以及使用的预训练模型
        log.info(f'Test on {test_path} with inference model {cfg.test.inference_model}')
        #在指定的预训练模型上进行测试
        trainer.test(inference_model=cfg.test.inference_model)

"""
if __name__ == "__main__": 是Python中的一个常见模式。这个条件判断的作用是检查当前的模块
（也就是.py文件）是否被直接运行。
在Python中，每个模块都有一个内置的变量__name__，这个变量的值取决于模块是如何被使用的。
(1)如果模块被直接运行，那么__name__的值就是"__main__"。
(2)如果模块被导入到其他模块中，那么__name__的值就是模块的名字。
所以，if __name__ == "__main__": 这个条件判断的意思就是，如果当前模块被直接运行，那么就执行
接下来的代码块。
在这个代码中，如果当前模块被直接运行，那么就会执行以下操作：
记录开始时间。
调用main()函数。
记录结束时间。
计算并记录整个过程的持续时间
"""
if __name__ == "__main__":
    start = time()
    main()
    stop = time()
    duration_time = utils.duration_in_hours(stop - start)
    log.info(f'Total time duration: {duration_time}')
