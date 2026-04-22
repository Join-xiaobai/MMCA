import pandas as pd
from torch_geometric.data import HeteroData
import torch
import torch.nn as nn
import torch_geometric.transforms as T
import numpy as np
from torch_geometric.nn import SAGEConv, to_hetero, GATConv
import tqdm
import torch.nn.functional as F
from torch import Tensor
import os
import torch_sparse
import torch_scatter
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import LinkNeighborLoader
from crossTest import cross_validation_metrics_print
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, accuracy_score
import logging



data = HeteroData()
# drug_node = pd.read_csv('./data/drug_feature.csv')
# drug_node_id = drug_node['drug_name'].unique()
# print(drug_node_id)
# protein_node = pd.read_csv('./data/protein_feature.csv')
# protein_node_id = protein_node['protein'].unique()
# print(protein_node_id)

# 读取 drug_feature.csv 文件
drug_node = pd.read_csv('./data/drug_feature.csv')
# 获取 drug_name 列的唯一值并编号
drug_node_id = pd.Series(np.arange(len(drug_node['drug_name'].unique())), 
                        index=drug_node['drug_name'].unique())
# print(drug_node_id)

# 读取 protein_feature.csv 文件 
protein_node = pd.read_csv('./data/protein_feature.csv')
# 获取 protein 列的唯一值并编号
protein_node_id = pd.Series(np.arange(len(protein_node['protein'].unique())),
                           index=protein_node['protein'].unique())
# print(protein_node_id)


# 保存所有类型节点索引
data["drug"].node_id = torch.arange(len(drug_node_id))
# print(data["drug"].node_id)
data["protein"].node_id = torch.arange(len(protein_node_id))
# print(data["protein"].node_id)

# 添加药物节点的特征，特征已经构造好
drug_feature = pd.read_csv('./data/drug_feature.csv')
drug_feature = drug_feature.iloc[:, 1:] # 只保留除第一列以外的其他列
drug_feature = torch.from_numpy(drug_feature.values).to(torch.float)
# print(drug_feature)
data["drug"].x = drug_feature
# print(drug_feature.shape)


# 添加蛋白质节点的特征，特征已经构造好
protein_feature = pd.read_csv('./data/protein_feature.csv')
protein_feature = protein_feature.iloc[:, 1:] # 只保留除第一列以外的其他列
protein_feature = torch.from_numpy(protein_feature.values).to(torch.float)
# print(protein_feature)
data["protein"].x = protein_feature
# print(protein_feature.shape)


# 获取节点之间的关联关系，即边的情况
# 注意这里需要转化为张量形式才可用
# 从DataFrame中提取出需要的数据并转换为numpy数组
# 读取关联边的内容


# 读取 drug_protein.csv 文件
edge_index_drug_to_protein = pd.read_csv('./data/drug_protein.csv')
# print(edge_index_drug_to_protein)

# 将 drug_name 和 protein 列的值替换为对应的编号
edge_index_drug_to_protein['Drug'] = edge_index_drug_to_protein['Drug'].map(drug_node_id)
edge_index_drug_to_protein['Protein'] = edge_index_drug_to_protein['Protein'].map(protein_node_id)

# print(edge_index_drug_to_protein)

# 打印转换后的结果
# print(edge_index_drug_to_protein)
edge_index_drug_to_protein = np.array(edge_index_drug_to_protein[['Drug', 'Protein']])
# print(edge_index_drug_to_protein)
edge_index_drug_to_protein = np.transpose(edge_index_drug_to_protein)
# print(edge_index_drug_to_protein)

data['drug','link','protein'].edge_index = torch.tensor(edge_index_drug_to_protein, dtype=torch.long)
# print(data['drug','link','protein'].edge_index)
# print(data['drug','link','protein'].edge_index.shape)

# 上一步此时只是保存了 a->b的关系 但是我们这个任务是无向图所以也应该保存 b->a的关系
# 即设置为双向关系
data = T.ToUndirected()(data)
# print(data)

# 定义DTI模型
class LTDTIModel(nn.Module):
    def __init__(self, drug_size, protein_size, hidden_sizes, output_size, num_heads=4):
        super(LTDTIModel, self).__init__()

        # 创建Protein特征提取器
        self.protein_encoder = nn.Sequential(
            nn.Linear(protein_size, hidden_sizes[0]),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        # 创建多头注意力层
        self.attention = nn.MultiheadAttention(hidden_sizes[0], num_heads)

        # 添加 nn.MaxPool1d 层
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 创建多个隐藏层
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(32, hidden_sizes[1]))
        for i in range(1, len(hidden_sizes)-1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            self.hidden_layers.append(nn.LeakyReLU())
            self.hidden_layers.append(nn.Dropout(0.2))
            # 添加残差连接
            self.hidden_layers.append(nn.Identity(hidden_sizes[i], hidden_sizes[i + 1]))

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )

    def forward(self, protein_features):
        # 编码protein特征
        protein_emb = self.protein_encoder(protein_features)

        # 应用多头注意力
        # (batch_size, sequence_length, embed_dim)
        attended_protein, _ = self.attention(protein_emb.unsqueeze(0), protein_emb.unsqueeze(0), protein_emb.unsqueeze(0))
        
        # 调整注意力结果的维度
        attended_protein = attended_protein.squeeze(0)

        # 融合 drug_emb 和 attended_drug 结果
        protein_features = 0.5 * protein_emb + 0.5 * attended_protein

        # 应用 nn.MaxPool1d
        protein_features = self.max_pool(protein_features.unsqueeze(0)).squeeze(0)

        # 通过隐藏层
        for layer in self.hidden_layers:
            protein_features = layer(protein_features)

        # 通过输出层
        output = self.output_layer(protein_features)
        return output
    
class SG_GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # 在 GATConv 模块中，当 add_self_loops 设置为 True 时，会在图的邻接矩阵中为每个节点添加自环。
        # 这意味着每个节点都会与自己相连，形成一个自循环的边。添加自环可以帮助节点捕捉到自身的特征，并在消息传递过程中考虑节点自身的影响。
        # 然而，在使用多个边类型的情况下，如您提供的边类型列表 [('disease', 'link', 'event'), ('event', 'rev_link', 'disease')]，
        # 将 add_self_loops 设置为 True 可能会导致错误的消息传递结果。因此，错误信息建议不要将 add_self_loops 设置为 True。
        # 不设置会报错！！！
        self.conv11 = GATConv(in_channels=hidden_channels, out_channels=8, heads=8, dropout=0.6, add_self_loops=False)
        self.conv22 = GATConv(in_channels=hidden_channels, out_channels=hidden_channels, heads=1, concat=False, dropout=0.6, add_self_loops=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # 第一次卷积
        # SAGE
        SAGEConv_x = F.relu(self.conv1(x, edge_index))
        gatingUnit_SAGEConv_x = torch.sigmoid(SAGEConv_x) # 门控单元
        # GAT
        GATConv_x = F.relu(self.conv11(x, edge_index))
        gatingUnit_GATConv_x = torch.sigmoid(GATConv_x)  # 门控单元
        # 权重分配
        SAGEConv_x = SAGEConv_x * gatingUnit_SAGEConv_x
        GATConv_x  = GATConv_x  * gatingUnit_GATConv_x
        # 连接
        x = x + SAGEConv_x + GATConv_x
        
        # 第二次卷积
        # SAGE
        SAGEConv_x = F.relu(self.conv2(x, edge_index))
        gatingUnit_SAGEConv_x = torch.sigmoid(SAGEConv_x) # 门控单元
        # GAT
        GATConv_x = F.relu(self.conv22(x, edge_index))
        gatingUnit_GATConv_x = torch.sigmoid(GATConv_x) # 门控单元
        # 权重分配
        SAGEConv_x = SAGEConv_x * gatingUnit_SAGEConv_x
        GATConv_x  = GATConv_x  * gatingUnit_GATConv_x
        # 连接
        x = x + SAGEConv_x + GATConv_x
        
        return x
    
# 定义分类器
class Classifier(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # MLP解码器
        self.decoder = nn.Sequential(nn.Linear(2*in_channels, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 1))
    def forward(self, x_drug: Tensor, x_protein: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_drug = x_drug[edge_label_index[0]]
        edge_feat_protein = x_protein[edge_label_index[1]]
        edge_input = torch.cat([edge_feat_drug, edge_feat_protein], dim=-1)
        return self.decoder(edge_input).squeeze(-1)
        
        # edge_feat_drug = x_drug[edge_label_index[0]]
        # edge_feat_protein = x_protein[edge_label_index[1]]
        # return (edge_feat_drug * edge_feat_protein).sum(dim=-1)
        
class Model(torch.nn.Module):
    def __init__(self, hidden_channels, drug_dim, protein_dim):  # *_dim为有特征的那个节点的第二个维度数
        super().__init__()
        # 特征处理
        self.drug_lin = torch.nn.Linear(drug_dim, hidden_channels)
        self.protein_lin = torch.nn.Linear(protein_dim, hidden_channels)
        self.drug_emb = torch.nn.Embedding(data["drug"].num_nodes, hidden_channels)
        self.protein_emb = torch.nn.Embedding(data["protein"].num_nodes, hidden_channels)
        # GNN处理药物特征
        self.gnn = SG_GNN(hidden_channels)
        # 转化为异构图网络
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        # LTDTIModel处理蛋白质特征
        self.LTDTIModel = LTDTIModel(hidden_channels, hidden_channels, [64, 256, 128, 64], hidden_channels)
        # 分类器
        self.classifier = Classifier(hidden_channels)

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "drug": self.drug_lin(data["drug"].x) + self.drug_emb(data["drug"].node_id),
            "protein": self.protein_lin(data["protein"].x) + self.protein_emb(data["protein"].node_id),
        }
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_gnn_dict = self.gnn(x_dict, data.edge_index_dict)
        x_LTDTIModel_dict = self.LTDTIModel(x_dict["protein"])
        pred = self.classifier(
            x_gnn_dict["drug"],
            # x_dict["protein"],
            x_LTDTIModel_dict,
            data["drug", "link", "protein"].edge_label_index,
        )
        return pred
    
def test_model(model, data):
    # Define the validation seed edges:
    edge_label_index = data["drug", "link", "protein"].edge_label_index
    edge_label = data["drug", "link", "protein"].edge_label
    data_loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[20, 10],
        edge_label_index=(("drug", "link", "protein"), edge_label_index),
        edge_label=edge_label,
        batch_size=30 * 128,
        shuffle=False,
    )

    preds = []
    ground_truths = []
    for sampled_data in tqdm.tqdm(data_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data))
            ground_truths.append(sampled_data["drug", "link", "protein"].edge_label)

    pred = torch.cat(preds, dim=0).cpu().numpy()
    # 使用阈值0.5将预测输出转换为二分类标签
    pred = (pred > -5.5).astype(float)
    
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

    # Calculate the metrics
    test_accuracy = accuracy_score(ground_truth, np.round(pred))
    precision = precision_score(ground_truth, np.round(pred))
    recall = recall_score(ground_truth, np.round(pred))
    f1 = f1_score(ground_truth, np.round(pred))
    auroc = roc_auc_score(ground_truth, pred)
    auprc = average_precision_score(ground_truth, pred)

    # 计算混淆矩阵
    print(ground_truth)
    print(np.round(pred)) # [ -5.  -2.   6. ...  -9. -11. -10.]
    cm = confusion_matrix(ground_truth, np.round(pred))
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)


    log_message = (
        f'Test Accuracy: {test_accuracy:.2f}, '
        f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}, AUROC: {auroc:.2f}, AUPRC: {auprc:.2f}, '
        f'Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}'
    )
    print(log_message)

    return test_accuracy, precision, recall, f1, auroc, auprc, sensitivity, specificity, log_message





model = Model(hidden_channels=64, drug_dim=64, protein_dim=340)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = torch.nn.BCEWithLogitsLoss().to(device)

# 交叉验证
randomseeds = [10, 20, 30, 40, 50]
models = ['bestModel/best_model_10.pt', 'bestModel/best_model_20.pt', 'bestModel/best_model_30.pt',
          'bestModel/best_model_40.pt', 'bestModel/best_model_50.pt']

# 数据集选择
datasets = './data/drug_protein.csv'

# 保存结果
accuracys = []
sensitivitys = []
specificitys = []
precisions = []
recalls = []
f1s = []
aurocs = []
auprcs = []

# 准备数据集
for randomseed, model_path in zip(randomseeds, models):
    # 设置随机种子
    torch.manual_seed(randomseed)

    transform = T.RandomLinkSplit(
        num_val = 0.1, #设置验证机比例 10%
        num_test = 0.1, #设置测试机比例 30%
        disjoint_train_ratio = 0.3, #设置训练集的边(消息边，监督边)中 监督边（不参与消息传递）占比（可设置为0）
        neg_sampling_ratio = 1.0, #设置负采样比例
        add_negative_train_samples = False, #设置每轮训练随机采负样本而不是一直使用固定的第一次随机初始化的负样本
        edge_types = ('drug','link','protein'), #设置边的类型
        rev_edge_types = ('protein','rev_link','drug'), #设置双向边
    )

    # 因为设置了
    # num_val = 0.1, #设置验证机比例 10%
    # num_test = 0.3, #设置测试机比例 30%
    # 所以优先出来的是val 和 test 数据集
    train_data, val_data, test_data = transform(data)
    
    # 保存最好的模型
    best_model = torch.load(model_path)  # 加载最佳模型
    model.load_state_dict(best_model)  # 将参数加载到模型中

    print('--'*20,'测试开始','--'*20)
    # 创建日志记录器
    logging.basicConfig(filename='test_log.txt', level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    test_accuracy, precision, recall, f1, auroc, auprc, sensitivity, specificity, log_message = test_model(model, test_data)
    logging.info(log_message)
    
    accuracys.append(float(test_accuracy))
    sensitivitys.append(float(sensitivity))
    specificitys.append(float(specificity))
    precisions.append(float(precision))
    recalls.append(float(recall))
    f1s.append(float(f1))
    aurocs.append(float(auroc))
    auprcs.append(float(auprc))

    print('--'*20,'测试结束','--'*20)
print(accuracys, sensitivitys, specificitys, precisions, recalls, f1s, aurocs, auprcs)

metrics_print = cross_validation_metrics_print(accuracys, sensitivitys, specificitys, precisions, recalls, f1s,
                                               aurocs, auprcs)
print(metrics_print)
logging.info(metrics_print)