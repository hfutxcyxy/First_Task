import os
import random
import torch
import h5py
import numpy as np
from enum import Enum
from datetime import datetime

from tools.utils import io

#这里是定义了main中用到的网络类型，是一个枚举类型，只能够取ANCSH或者NPCS
class NetworkType(Enum):
    ANCSH = 'ANCSH'
    NPCS = 'NPCS'

#设置随机数种子函数
def set_random_seed(seed):
    #调用numpy的随机数库，生成一个随机数种子
    np.random.seed(seed)
    """
    torch.manual_seed(seed)会设置生成随机数的种子，并返回一个torch.Generator对象。
    get_state()函数会返回一个表示当前随机数生成器状态的对象。torch.set_rng_state(state)函数
    会设置随机数生成器的状态
    即将随机数生成器的状态设置为生成当前随机数种子的生成器的状态
    """
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    #调用random库生成随机数种子
    random.seed(seed)

#将时间转换成标准时间格式
def duration_in_hours(duration):
    #调用python的除法取余函数divmod，商给分钟，余数给秒
    t_m, t_s = divmod(duration, 60)
    #同理
    t_h, t_m = divmod(t_m, 60)
    #设置标准输出格式
    duration_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
    return duration_time

#获取预测的分割与坐标
def get_prediction_vertices(pred_segmentation, pred_coordinates):
    #返回预测的分割
    segmentations = np.argmax(pred_segmentation, axis=1)
    #返回预测的坐标，使用两个向量作为索引在pred_coordinates进行搜索
    coordinates = pred_coordinates[
        np.arange(pred_coordinates.shape[0]).reshape(-1, 1),
        np.arange(3) + 3 * np.tile(segmentations.reshape(-1, 1), [1, 3])]
    return segmentations, coordinates


def get_num_parts(h5_file_path):
    if not io.file_exist(h5_file_path):
        raise IOError(f'Cannot open file {h5_file_path}')
    input_h5 = h5py.File(h5_file_path, 'r')
    num_parts = input_h5[list(input_h5.keys())[0]].attrs['numParts']
    bad_groups = []
    #定义一个lambda匿名函数，用于检查每一个数据集中部件数是否等于获取的部件数，若不等于，则将
    #该数据集的名字加入bad_group中
    visit_groups = lambda name, node: bad_groups.append(name) if isinstance(node, h5py.Group) and node.attrs[
        'numParts'] != num_parts else None
    #使用visititem函数遍历文件中的每一个数据集，并对其调用visit_group函数检查
    input_h5.visititems(visit_groups)
    input_h5.close()
    #如果存在数据集是bad_group，则证明数据集有问题，会抛出一个error
    if len(bad_groups) > 0:
        raise ValueError(f'Instances {bad_groups} in {h5_file_path} have different number of parts than {num_parts}')
    #返回读取到的部件数
    return num_parts


def get_latest_file_with_datetime(path, folder_prefix, ext, datetime_pattern='%Y-%m-%d_%H-%M-%S'):
    folders = os.listdir(path)
    #定义要搜索的文件的标准命名格式
    folder_pattern = folder_prefix + datetime_pattern
    #找出所有符合指定前缀与扩展名的文件
    matched_folders = np.asarray([fd for fd in folders if fd.startswith(folder_prefix)
                                  if len(io.get_file_list(os.path.join(path, fd), ext))])
    #如果没有一个匹配，则返回两个空字符
    if len(matched_folders) == 0:
        return '', ''
    #将每个匹配的文件夹的日期时间转换为时间戳（以毫秒为单位）
    timestamps = np.asarray([int(datetime.strptime(fd, folder_pattern).timestamp() * 1000) for fd in matched_folders])
    #将文件按照时间戳进行排序
    sort_idx = np.argsort(timestamps)
    #按照排序好的时间索引对符合的文件进行排序
    matched_folders = matched_folders[sort_idx]
    #返回时间排序最新的文件夹
    latest_folder = matched_folders[-1]
    #列出所有的文件
    files = io.alphanum_ordered_file_list(os.path.join(path, latest_folder), ext=ext)
    #返回时间排序最新的文件
    latest_file = files[-1]
    return latest_folder, latest_file

#更新并记录一些值
class AvgRecorder(object):
    """
    Average and current value recorder
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
