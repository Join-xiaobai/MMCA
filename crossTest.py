import numpy as np


def cross_validation_metrics_print(accuracys, sensitivitys, specificitys, precisions, recalls, f1s, aurocs, auprcs):
    """
    计算5折交叉验证的平均性能指标和标准差

    参数:
    accuracys, sensitivitys, specificitys, precisions, recalls, f1s, aurocs, auprs (list):
        每一折的对应指标值

    返回:
    str: 包含平均值和标准差的表格格式字符串
    """
    return f"Final scores (mean)\n" \
        f"|   Accuracy  | Sensitivity | Specificity |  Precision  |   recall    |  F1-score   |     AUC     |     AUPR    |\n" \
        f"|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|\n" \
        f"| {np.mean(accuracys) * 100:.2f}%±{np.std(accuracys) * 100:.2f}%| {np.mean(sensitivitys) * 100:.2f}%±{np.std(sensitivitys) * 100:.2f}%| " \
        f"{np.mean(specificitys) * 100:.2f}%±{np.std(specificitys) * 100:.2f}%| {np.mean(precisions) * 100:.2f}%±{np.std(precisions) * 100:.2f}%| " \
        f"{np.mean(recalls) * 100:.2f}%±{np.std(recalls) * 100:.2f}%| {np.mean(f1s) * 100:.2f}%±{np.std(f1s) * 100:.2f}%| " \
        f"{np.mean(aurocs) * 100:.2f}%±{np.std(aurocs) * 100:.2f}%| {np.mean(auprcs) * 100:.2f}%±{np.std(auprcs) * 100:.2f}%| "

def cross_validation_metrics(accuracys, sensitivitys, specificitys, precisions, recalls, f1s, aurocs, auprcs):
    """
    计算5折交叉验证的平均性能指标和标准差

    参数:
    accuracys, sensitivitys, specificitys, precisions, recalls, f1s, aurocs, auprs (list):
        每一折的对应指标值

    返回:
    dict: 包含平均值和标准差的字典
    """
    return {
        'Accuracy': f'{np.mean(accuracys) * 100:.2f}%±{np.std(accuracys) * 100:.2f}%',
        'Sensitivity': f'{np.mean(sensitivitys) * 100:.2f}%±{np.std(sensitivitys) * 100:.2f}%',
        'Specificity': f'{np.mean(specificitys) * 100:.2f}%±{np.std(specificitys) * 100:.2f}%',
        'Precision': f'{np.mean(precisions) * 100:.2f}%±{np.std(precisions) * 100:.2f}%',
        'NPV': f'{np.mean(recalls) * 100:.2f}%±{np.std(recalls) * 100:.2f}%',
        'F1-score': f'{np.mean(f1s) * 100:.2f}%±{np.std(f1s) * 100:.2f}%',
        'MCC-score': f'{np.mean(aurocs) * 100:.2f}%±{np.std(aurocs) * 100:.2f}%',
        'AUC': f'{np.mean(auprcs) * 100:.2f}%±{np.std(auprcs) * 100:.2f}%'
    }

if __name__ == '__main__':
    # 假设已有每一折的指标值
    accuracys = [0.8116543388057009, 0.8130976005773047, 0.8186902399422695, 0.8069637380479885, 0.8118347465271514]
    sensitivitys = [0.9962, 0.9971, 0.9966, 0.9967, 0.9969]
    specificitys = [0.9882, 0.9905, 0.9916, 0.9920, 0.9901]
    precisions = [0.9886, 0.9896, 0.9903, 0.9909, 0.9892]
    recalls = [0.9964, 0.9971, 0.9968, 0.9969, 0.9971]
    f1s = [0.9923, 0.9932, 0.9934, 0.9937, 0.9929]
    aurocs = [0.9825, 0.9855, 0.9868, 0.9876, 0.9846]
    auprcs = [0.9971, 0.9975, 0.9977, 0.9978, 0.9973]

    metrics = cross_validation_metrics(accuracys, sensitivitys, specificitys, precisions, recalls, f1s, aurocs, auprcs)

    print(metrics)

    metrics_print = cross_validation_metrics_print(accuracys, sensitivitys, specificitys, precisions, recalls, f1s, aurocs, auprcs)

    print(metrics_print)