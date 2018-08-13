# 工具类
# 用来计算各种评估指标
import csv as csv


class EvaluationTool:
    # 计算recall
    @staticmethod
    def cal_recall(x, y, size):
        tp = 0
        fn = 0
        for i in range(size):
            if y[i] == 1 and x[i] == 1:
                # 此为真反例
                tp += 1
            elif y[i] == 1 and x[i] == 0:
                # 真实情况是正例，预测结果为反例
                fn += 1
        try:
            return tp / (tp + fn)
        except ZeroDivisionError:
            return 0

    # 计算precision
    @staticmethod
    def cal_precision(x, y, size):
        tp = 0
        fp = 0
        for i in range(size):
            if x[i] == 1 and y[i] == 1:
                # 此为真正例
                tp += 1
            elif y[i] == 0 and x[i] == 1:
                # 本身是反例，被预测为正例
                fp += 1
        try:
            return tp / (tp + fp)
        except ZeroDivisionError:
            return 0

    # 计算F1
    @classmethod
    def cal_f1(cls, predictions, target):
        p = cls.cal_precision(predictions, target, len(target))
        r = cls.cal_recall(predictions, target, len(target))
        try:
            return (2 * p * r) / (p + r)
        except ZeroDivisionError:
            return 0

    # 输出最终结果
    @classmethod
    def get_output(cls, predictions, target, index=0):
        out = dict()
        output_list = []
        out['precision'] = cls.cal_precision(predictions, target, len(predictions))  # 求准确率
        out['recall'] = cls.cal_recall(predictions, target, len(predictions))  # 求召回率
        # print(out)
        output_list.append(out['precision'])
        output_list.append(out['recall'])
        # 将记录保存到csv文件中
        with open('D:\\project\\PythonProject\\result\\result_' + str(index) + '.csv', 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(output_list)
        return out
