import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from tqdm import tqdm
import logging
import time
from tabulate import tabulate
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
import os
from torch_geometric.utils import negative_sampling

# 读取数据
drug_embeddings_feature = pd.read_csv('./data/drug_embeddings.csv')
drug_embeddings_feature_index = pd.read_csv('./data/drug_embeddings.csv', index_col=0)
drug_features = pd.read_csv('./data/drug_mol2vec_features.csv')
# 这个文件是按照 \t 的符号分隔保存的
relationship_data = pd.read_csv('./data/all_drug.csv')
# print(relationship_data.columns)

# 将标签从N和Y转化为1和0
# N代表两个药物是没有负相关作用的，所以是表示药物之间有关联作用
relationship_data['Adverse'] = relationship_data['Adverse'].apply(lambda x: 1 if x == 'N' else 0)


# 数据预处理
def create_node_features(drug_embeddings_feature, drug_features):
    # 将数据类型转换为 float32
    drug_embeddings_feature = drug_embeddings_feature.values[:, 1:].astype(np.float32)
    drug_features = drug_features.values[:, 1:].astype(np.float32)

    # 然后转换为 PyTorch Tensor
    drug_embeddings_feature = torch.from_numpy(drug_embeddings_feature)
    drug_features = torch.from_numpy(drug_features)
    # 将两个张量沿着列(第二个维度)进行拼接
    combined_tensor = torch.cat([drug_embeddings_feature, drug_features], dim=1)
    return torch.tensor(combined_tensor, dtype=torch.float)


def create_edge_index(interaction_scores):
    # 将蛋白质名称映射到索引
    protein_to_index = {protein: idx for idx, protein in enumerate(drug_embeddings_feature_index.index)}

    # 将蛋白质名称转换为索引
    edges = interaction_scores[['DrugBankID_InteractorA', 'DrugBankID_InteractorB']].applymap(
        protein_to_index.get).values.T

    return torch.tensor(edges, dtype=torch.long)


def create_labels(interaction_scores):
    return torch.tensor(interaction_scores['Adverse'].values, dtype=torch.float).view(-1)


# 创建图数据
node_features = create_node_features(drug_embeddings_feature, drug_features)

# 数据集重采样
positive_samples = relationship_data[relationship_data['Adverse'] == 1]
negative_samples = relationship_data[relationship_data['Adverse'] == 0]
balanced_interaction_scores = pd.concat([positive_samples, negative_samples])

# 创建图数据对象
edge_index = create_edge_index(balanced_interaction_scores)
labels = create_labels(balanced_interaction_scores)
data = Data(x=node_features, edge_index=edge_index, y=labels)

# 划分数据集
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
transform = T.Compose([
    T.ToDevice(device),
])

test_data = transform(data)

batch_size = 128 * 128 * 8
test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[-1, -1],
    # neg_sampling_ratio=2.0,
    batch_size=batch_size,
    shuffle=True,
)


# 定义模型
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)
        # MLP解码器
        self.decoder = Sequential(Linear(2 * out_channels, hidden_channels),
                                  ReLU(),
                                  Linear(hidden_channels, 1))

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x_res = x
        x = F.relu(self.conv2(x_res, edge_index))
        x_res = x_res + x
        x = F.relu(self.conv3(x_res, edge_index))
        x_res = x_res + x
        x = self.conv4(x_res, edge_index)
        return x

    def decode(self, z, edge_label_index):
        edge_input = torch.cat([z[edge_label_index[0]], z[edge_label_index[1]]],
                               dim=-1)  # Concatenate the embeddings of the source and target nodes
        # print(z[edge_label_index[0]].shape)
        # ------------------------------------------------
        import pandas as pd

        features = []
        for idx in edge_label_index[0]:
            feature = z[idx].cpu().detach().numpy()
            features.append(feature)

        df = pd.DataFrame(features)
        df.to_csv('./p_score.csv', index=False, header=False)
        print('特征处理并保存完毕！')
        # ------------------------------------------------
        exit(1)  # 退出时返回状态码1
        return self.decoder(edge_input).squeeze(-1)
        # return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


# 模型初始化
in_channels = node_features.shape[1]
hidden_channels = 64
out_channels = 64
model = Net(in_channels, hidden_channels, out_channels).to(device)
# 训练准备
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
criterion = torch.nn.BCEWithLogitsLoss()


def train(loader):
    model.train()
    total_loss = 0

    # 创建一个tqdm实例
    progress_bar = tqdm(loader, desc="Training")
    # 使用tqdm包装loader，以显示进度条
    for batch in progress_bar:
        # Data(x=[18463, 340], edge_index=[2, 1863991], y=[1863991], edge_label_index=[2, 131072], n_id=[18463], e_id=[1863991], input_id=[131072])
        # print(batch)
        batch.to(device)
        optimizer.zero_grad()
        z = model.encode(batch.x, batch.edge_index)
        out = model.decode(z, batch.edge_index).view(-1)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_description(f"Training (loss = {total_loss / (progress_bar.n + 1):.4f})")

    return total_loss / len(loader)


@torch.no_grad()
def test(loader):
    model.eval()
    total_score = 0
    # 使用tqdm包装loader，以显示进度条
    for batch in tqdm(loader, desc="Val or Testing"):
        batch.to(device)
        z = model.encode(batch.x, batch.edge_index)
        out = model.decode(z, batch.edge_index).view(-1).sigmoid()
        total_score += roc_auc_score(batch.y.cpu().numpy(), out.cpu().numpy())

    return total_score / len(loader)


print("#" * 20)
# from datetime import datetime
# current_datetime = datetime.now()
# formatted_datetime = current_datetime.strftime("%m_%d_%H:%M")

# ----------------------------创建日志文件----------------------------
log_file_path = './outputs'
if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)

# log_path = os.path.join(log_file_path, f'tonggou_log_{formatted_datetime}.txt')
log_path = os.path.join(log_file_path, f'log.txt')

if not os.path.exists(log_path):
    with open(log_path, 'a') as f:
        f.write('log info:\n')

# ----------------------------创建日志文件----------------------------


# 加载最好的模型
best_model = torch.load('./bestModel/best_model.pt')  # 加载最佳模型
model.load_state_dict(best_model)  # 将参数加载到模型中

print("定义完成，开始训练")

print("#" * 20)
best_val_auc = best_test_auc = 0
best_val_epoch = best_test_epoch = 0
test_auc = test(test_loader)