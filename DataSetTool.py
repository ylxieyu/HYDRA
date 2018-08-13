import numpy as np
import os
import Tools.FilesTool as FilesTool
import imblearn.over_sampling as over_sampling


class DataSetTool:
    # 08版的度量补偿
    # Mij in Target = (Mij in Target * Mean(Mj in Source)) / Mean(Mj) in  Target
    @staticmethod
    def metric_compensation(source, target):
        # 遍历每一个度量属性
        for j in range(target.shape[1]):
            # 计算每个度量属性的均值
            metric_mean_source = np.mean(source[:, j])
            metric_mean_target = np.mean(target[:, j])
            # 遍历每一个样例
            for i in range(target.shape[0]):
                target[i, j] = (target[i, j] * metric_mean_source) / metric_mean_target
        return target

    # 17版进行调整的度量补偿
    # Mij in Source = (Mij in Source * Mean(Mj in Target)) / Mean(Mj) in Source
    @staticmethod
    def metric_compensation_adopt(source, target):
        # 遍历每一个度量属性
        for j in range(source.shape[1]):
            # 计算每个度量属性的均值
            metric_mean_source = np.mean(source[:, j])
            metric_mean_target = np.mean(target[:, j])
            # 遍历每一个样例
            for i in range(source.shape[0]):
                source[i, j] = (source[i, j] * metric_mean_target) / metric_mean_source
        return source

    # 读取文件夹下的所有文件，并返回处理好的数据集
    # metrics_num 度量数目（原始数据中除开标签列的列数）
    # is_sample 是否重采样
    # is_normalized 是否数据归一化
    @staticmethod
    def init_data(folder_path, metrics_num, is_sample=True, is_normalized=True):
        # 获取目录下所有原始文件
        files = os.listdir(folder_path)
        data_list, label_list = [], []
        for file in files:
            # 每一个子文件的真实路径
            file_path = folder_path+file
            # txt文件
            if 'txt' == FilesTool.file_type(file) or 'TXT' == FilesTool.file_type(file):
                # 直接读取文件
                data_file = np.loadtxt(file_path, dtype=float, delimiter=',', usecols=range(0, metrics_num+1))
                label_file = np.loadtxt(file_path, dtype=float, delimiter=',', usecols=metrics_num+1)
                if is_normalized:
                    # 数据归一化
                    data_file -= data_file.min()
                    data_file /= data_file.max()
                    label_file -= label_file.min()
                    label_file /= label_file.max()
                # 加入列表
                data_list.append(data_file)
                label_list.append(label_file)
        # 重采样
        if is_sample:
            for index in range(len(data_list)):
                data_list[index], label_list[index] = over_sampling.SMOTE(kind='regular').fit_sample(data_list[index],
                                                                                                     label_list[index])
        return data_list, label_list
