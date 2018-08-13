# Genetic Algorithm
from sklearn.linear_model import LogisticRegression
import random
import numpy as np
from Tools.EvaluationTool import EvaluationTool


class GA:
    def __init__(self, pop_size, max_gen=100, crossover_rate=0.6, mutation_rate=0.1):
        # 初始化
        self.pop_size = pop_size  # 种群数
        self.max_gen = max_gen  # 最大繁殖代数
        self.crossover_rate = crossover_rate  # 交叉率
        self.mutation_rate = mutation_rate  # 变异率
        self.genes_num = None  # 基因数目
        self.clf_list = None
        self.target_data = None
        self.target_label = None
        self.pop_f1_list = []  # 记录种群中每个个体的f1值，避免重复计算

    def fit(self, source_data_list, source_label_list, target_data, target_label):
        # 得到一众基础分类器
        clf_list = self.get_base_clf(source_data_list, source_label_list, target_data, target_label)
        self.clf_list = clf_list
        # 个体->染色体->list of genes
        # 确定基因的个数 = 分类器个数 + 阈值
        genes_num = len(clf_list) + 1
        self.genes_num = genes_num
        # 初始化种群
        pop = np.random.random((self.pop_size, self.genes_num))
        # 最后一列阈值的取值范围由（0,1）调整到(0,genes_num-1)
        pop[:, -1] = pop[:, -1] * (genes_num - 1)
        # 计算初始群体中的最优解
        pre_solution, pre_f1 = self.get_best_from_pop(pop)
        # 繁殖
        cur_gen = 0  # 当前代数为0
        while cur_gen < self.max_gen:
            temp_pop = pop.copy()
            pop_new = self.ga_generation(temp_pop)
            cur_solution, cur_f1 = self.get_best_from_pop(pop_new)
            if cur_f1 > pre_f1:
                pre_f1 = cur_f1
                pre_solution = cur_solution
            cur_gen += 1
        print(pre_f1)
        return pre_solution, pre_f1

    # 训练得到一众基础分类器
    def get_base_clf(self, source_data_list, source_label_list, target_data, target_label):
        # 将target中一部分样例添加到source中，当做训练集
        sdl, sll, td, tl = self.transfer_target_to_source(source_data_list, source_label_list,
                                                          target_data, target_label)
        self.target_data, self.target_label = td, tl
        clf_list = []
        for index in range(len(source_data_list)):
            # 论文中用的是逻辑回归，感觉效果不是很好
            clf = LogisticRegression()
            # 换成C4.5
            # clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=30)
            clf.fit(sdl[index], sll[index])
            clf_list.append(clf)
        # 使用target训练，得到(N+1)th clf
        clf = LogisticRegression()
        clf.fit(td, tl)
        clf_list.append(clf)
        return clf_list

    # 从target中转移部分样例到source中
    # 20180806 将for循环改成列表推导式
    @staticmethod
    def transfer_target_to_source(source_data_list, source_label_list, target_data, target_label):
        # 随机移除10%的样例
        t_sample_num = target_data.shape[0]
        t_remove_num = round(t_sample_num * 0.1)  # 四舍五入取整
        t_remove_index = random.sample(range(0, t_sample_num), t_remove_num)
        t_other_index = [i for i in range(0, t_sample_num) if i not in t_remove_index]
        # 新的target
        target_data_new = np.array([target_data[i, :] for i in t_other_index])
        target_label_new = np.array([target_label[i] for i in t_other_index])
        # 移出的样例以及标签
        temp_data = np.array([target_data[i, :] for i in t_remove_index])
        temp_label = np.array([target_label[i] for i in t_remove_index])
        # 新的source
        temp_data_list = [np.append(source_data_list[i], temp_data, axis=0) for i in range(len(source_data_list))]
        temp_label_list = [np.append(source_label_list[i], temp_label, axis=0) for i in range(len(source_label_list))]
        return temp_data_list, temp_label_list, target_data_new, target_label_new

    # 从种群中选出最优的个体（染色体）
    def get_best_from_pop(self, pop):
        if len(self.pop_f1_list) != 0:
            self.pop_f1_list.clear()
        # 比较每一个个体的F1值
        pre_f1, best_index = 0, 0
        for index in range(pop.shape[0]):
            cur_f1 = self.score(pop[index])
            self.pop_f1_list.append(cur_f1)
            if cur_f1 > pre_f1:
                pre_f1 = cur_f1
                best_index = index
        return pop[best_index], pre_f1

    # 预测目标数据集的标签并计算分数
    def score(self, pop_item):
        # 计算target中每一个样例的得分
        predict_label = []
        for sample in range(self.target_data.shape[0]):
            # comp = from i to N+1 (wight * clf prediction) / loc(sample)
            comp = 0
            for i in range(len(self.clf_list)):
                comp += pop_item[i] * self.clf_list[i].predict(self.target_data[sample].reshape(1, -1))[0]
            # comp /= self.target_data[sample, 0]
            if comp >= pop_item[-1]:
                predict_label.append(1)
            else:
                predict_label.append(0)
        score = EvaluationTool.cal_f1(np.array(predict_label), self.target_label)
        return score

    # 繁殖迭代
    def ga_generation(self, pop):
        # 选择阶段
        pop_select = self.select(pop)
        # 交叉阶段
        pop_cross = self.crossover(pop_select)
        # 变异阶段
        pop_mutation = self.mutation(pop_cross)
        return np.array(pop_mutation)

    # 选择
    # 20180807 只修改到这里 下次从这里开始
    def select(self, pop):
        fit_list, q_list = [], []  # 适应度列表，累积概率列表
        choose_index = set()  # 被选中的个体
        fit_sum = sum(self.pop_f1_list)   # 适应度总和
        p_list_array = np.divide(np.array(self.pop_f1_list), fit_sum)  # 遗传到下一代概率列表
        for i in range(len(p_list_array)):
            # 计算每个个体的累积概率
            q = 0
            for j in range(i+1):
                q += p_list_array[j]
            q_list.append(q)
        # 产生一个随机数列表，用于选择
        rand_list = np.random.rand(pop.shape[0])
        for i in range(len(rand_list)):
            choose_index.add(self.get_index_from_list(rand_list[i], q_list))
        pop_new = [pop[i] for i in choose_index]
        return np.array(pop_new)

    # 从目标列表中，返回传入参数所对应的位置，传入参数应位于两个值之间
    @staticmethod
    def get_index_from_list(num, target_list):
        for i in range(len(target_list)):
            if i == 0 and num <= target_list[0]:
                return 0
            else:
                if target_list[i-1] < num <= target_list[i]:
                    return i

    # 交叉
    def crossover(self, pop):
        son_list = []
        pair_num = int(pop.shape[0]/2)
        for i in range(pair_num):
            rand_num = random.random()
            if rand_num < self.crossover_rate:
                # 随机选取交叉位置
                rand_cross_index = random.randint(0, pop.shape[1]-1)  # ???是否减一
                # 交叉并产生新子代
                parent_a = pop.copy()[i*2, :]
                parent_b = pop.copy()[i*2+1, :]
                temp_parent_a = parent_a.copy()[rand_cross_index:]
                parent_a[rand_cross_index:] = parent_b[rand_cross_index:]  # 新子代a
                parent_b[rand_cross_index:] = temp_parent_a  # 新子代b
                son_list.append(parent_a)
                son_list.append(parent_b)
        if len(son_list) != 0:
            pop_new = np.append(pop, np.array(son_list), axis=0)
        else:
            pop_new = pop
        return pop_new

    # 变异
    def mutation(self, pop):
        son_list = []
        for i in range(pop.shape[0]):
            rand_num = random.random()
            if rand_num < self.mutation_rate:
                # 随机产生变异位置
                rand_mutation_index = random.randint(0, pop.shape[1]-1)  # ???是否减一
                # 变异产生新子代
                parent = pop.copy()[i, :]
                if rand_mutation_index == pop.shape[1]-1:
                    # 最后一列变异
                    r = random.random()*(self.genes_num-1)
                    parent[rand_mutation_index] = r
                else:
                    # 其他列变异
                    r = random.random()
                    parent[rand_mutation_index] = r
                son_list.append(parent)
        if len(son_list) != 0:
            pop_new = np.append(pop, np.array(son_list), axis=0)
        else:
            pop_new = pop
        return pop_new
