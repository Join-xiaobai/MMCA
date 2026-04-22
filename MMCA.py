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

# 设置随机种子
torch.manual_seed(50)

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


# print(train_data)
# print(val_data)
# print(test_data)

# print(train_data["drug"].num_nodes)
# print(train_data["protein"].num_nodes)

# 定义随机边:
edge_label_index = train_data['drug','link','protein'].edge_label_index
edge_label = train_data['drug','link','protein'].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[20, 10],
    neg_sampling_ratio=2.0,
    edge_label_index=(('drug','link','protein'), edge_label_index),
    edge_label=edge_label,
    batch_size=128 * 30,
    shuffle=True,
)

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
    
def train_model(model, optimizer, criterion, total_epoch, train_loader, val_data, test_data, log_path, best_model_path):
    max_val_auc = max_test_auc = 0
    max_val_epoch = max_test_epoch = 0
    best_model = None

    for epoch in range(1, total_epoch + 1):
        total_loss = total_examples = 0
        for sampled_data in tqdm.tqdm(train_loader):
            optimizer.zero_grad()
            sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data["drug", "link", "protein"].edge_label
            loss = criterion(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

        # ----------------------------日志文件----------------------------
        with open(log_path, 'a') as f:
            f.write('\n')
            f.write('-----' * 20)
            f.write('\n')
            f.write(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
            f.write('\n')
        # ----------------------------日志文件----------------------------

        val_auc = test_model(model, val_data)
        test_auc = test_model(model, test_data)
        print(f"Val AUC: {val_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")

        # ----------------------------日志文件----------------------------
        with open(log_path, 'a') as f:
            f.write(f"Val AUC: {val_auc:.4f}")
            f.write('\n')
            f.write(f"Test AUC: {test_auc:.4f}")
            f.write('\n')
            f.write('\n')
        # ----------------------------日志文件----------------------------

        if val_auc >= max_val_auc:
            max_val_auc = val_auc
            max_val_epoch = epoch

        if test_auc >= max_test_auc:
            max_test_auc = test_auc
            max_test_epoch = epoch
            best_model = model.state_dict()  # 保存当前最佳模型的参数
            # 保存最佳模型
            torch.save(best_model, best_model_path)

        print(f"Best Val AUC: {max_val_auc:.4f}, Best Epoch: {max_val_epoch:03d}")
        print(f"Best Test AUC: {max_test_auc:.4f}, Best Epoch: {max_test_epoch:03d}")

        # ----------------------------日志文件----------------------------
        with open(log_path, 'a') as f:
            f.write(f"Best Val AUC: {max_val_auc:.4f}, Best Epoch: {max_val_epoch:03d},")
            f.write('\n')
            f.write(f"Best Test AUC: {max_test_auc:.4f}, Best Epoch: {max_test_epoch:03d},")
            f.write('\n')
            f.write('-----' * 20)
            f.write('\n')
            f.write('\n')
        # ----------------------------日志文件----------------------------

    # 保存最佳模型
    torch.save(best_model, best_model_path)


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
    # sampled_data = next(iter(data_loader))
    preds = []
    ground_truths = []
    for sampled_data in tqdm.tqdm(data_loader):
        with torch.no_grad():
            sampled_data.to(device)
            preds.append(model(sampled_data))
            ground_truths.append(sampled_data["drug", "link", "protein"].edge_label)
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    # print(f"Validation AUC: {auc:.4f}")
    return auc

model = Model(hidden_channels=64, drug_dim=64, protein_dim=340)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = torch.nn.BCEWithLogitsLoss().to(device)

# ----------------------------创建日志文件----------------------------
log_file_path = './outputs'
if not os.path.exists(log_file_path):
    os.makedirs(log_file_path)

log_path = os.path.join(log_file_path, 'log.txt')

if not os.path.exists(log_path):
    with open(log_path, 'a') as f:
        f.write('log info:\n')

# ----------------------------创建日志文件----------------------------


# 保存最好的模型
# best_model = torch.load(model_path)  # 加载最佳模型
# model.load_state_dict(best_model)  # 将参数加载到模型中
best_model_path = './bestModel'
if not os.path.exists(best_model_path):
    os.makedirs(best_model_path)
best_model_path = os.path.join(best_model_path, 'best_model.pt')

total_epoch = 300

print('--'*20,'训练开始','--'*20)
train_model(model, optimizer, criterion, total_epoch, train_loader, val_data, test_data, log_path, best_model_path)
print('--'*20,'训练结束','--'*20)
