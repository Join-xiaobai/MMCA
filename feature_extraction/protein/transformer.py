import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate

class PPIPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(PPIPredictionModel, self).__init__()
        self.encoder_layer = TransformerEncoderLayer(input_dim, num_heads,batch_first=True)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Output layer for PPI prediction

    def forward(self, x1, x2):
        x1 = x1.unsqueeze(1)  # Add the sequence length dimension
        x2 = x2.unsqueeze(1)
        encoded1 = self.encoder(x1)

        encoded2 = self.encoder(x2)
        pooled1 = torch.mean(encoded1, dim=1)
        pooled2 = torch.mean(encoded2, dim=1)

        df = pd.DataFrame(pooled1.cpu().detach().numpy())
        df.to_csv('./p_score.csv', index=False, header=False)
        exit(1)
        # print(pooled1.shape)

        combined = (pooled1 + pooled2) / 2
        hidden = F.relu(self.fc1(combined))
        logits =self.fc3(F.relu(self.fc2(hidden)))
        return logits

class PPIPredictionDataset(Dataset):
    def __init__(self, protein_embeddings, protein_features, relationship_file):
        self.protein_embeddings = pd.read_csv(protein_embeddings, index_col=0)
        self.protein_features = pd.read_csv(protein_features, index_col=0)
        self.relationship_data = pd.read_csv(relationship_file)
        # self.targets = (self.relationship_data['Adverse'] == 'N').astype(int)
        self.targets = self.relationship_data['y']

    def __len__(self):
        return len(self.relationship_data)

    def __getitem__(self, idx):
        pro1, pro2, _, y = self.relationship_data.iloc[idx]
        pro1_protein_embeddings_features = torch.tensor(self.protein_embeddings.loc[pro1].values).float()
        pro1_protein_features = torch.tensor(self.protein_features.loc[pro1].values).float()
        # 调整输入特征的维度
        pro1_protein_embeddings_features = pro1_protein_embeddings_features.unsqueeze(0)
        pro1_protein_features = pro1_protein_features.unsqueeze(0)
        # 拼接特征
        pro1_features = torch.cat((pro1_protein_embeddings_features, pro1_protein_features), dim=1)
        
        pro2_protein_embeddings_features = torch.tensor(self.protein_embeddings.loc[pro2].values).float()
        pro2_protein_features = torch.tensor(self.protein_features.loc[pro2].values).float()
        # 调整输入特征的维度
        pro2_protein_embeddings_features = pro2_protein_embeddings_features.unsqueeze(0)
        pro2_protein_features = pro2_protein_features.unsqueeze(0)
        # 拼接特征
        pro2_features = torch.cat((pro2_protein_embeddings_features, pro2_protein_features), dim=1)
        
        # 输入 pro1_features 的形状为 (1, feature_size)  # 将 batch_size 维度压缩掉
        pro1_features = pro1_features.squeeze(0)
        pro2_features = pro2_features.squeeze(0)
        
        # y = torch.tensor(y == 'N').float()
        y = torch.tensor(y).float()
        return pro1_features, pro2_features, y
# Define the training, validation, and testing functions
from sklearn.metrics import roc_auc_score

def train(model, data_loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="Training", leave=False)

    for batch_idx, (x1, x2, y) in progress_bar:
        # print(x1.shape)
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x1, x2)
        loss = criterion(logits, y.float().view(-1, 1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = torch.round(torch.sigmoid(logits))
        correct += (predicted == y.view(-1, 1)).sum().item()
        total += y.size(0)
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(total_loss/total)})
        y_true.extend(y.tolist())
        y_pred.extend(predicted.tolist())

    accuracy = correct / total
    average_loss = total_loss / len(data_loader)
    auc = roc_auc_score(y_true, y_pred)

    return average_loss, accuracy, auc

def test(model, data_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="test", leave=False)
    with torch.no_grad():
        for batch_idx, (x1, x2, y) in progress_bar:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            logits = model(x1, x2)
            loss = criterion(logits, y.float().view(-1, 1))

            total_loss += loss.item()
            predicted = torch.round(torch.sigmoid(logits))
            correct += (predicted == y.view(-1, 1)).sum().item()
            total += y.size(0)
            progress_bar.set_postfix({'test_loss': '{:.3f}'.format(loss.item()/len(x1))})
            y_true.extend(y.tolist())
            y_pred.extend(predicted.tolist())

    accuracy = correct / total
    average_loss = total_loss / len(data_loader)
    auc = roc_auc_score(y_true, y_pred)

    return average_loss, accuracy, auc

def validate(model, data_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc="validate", leave=False)
    with torch.no_grad():
        for batch_idx, (x1, x2, y) in progress_bar:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)
            logits = model(x1, x2)
            loss = criterion(logits, y.float().view(-1, 1))

            total_loss += loss.item()
            predicted = torch.round(torch.sigmoid(logits))
            correct += (predicted == y.view(-1, 1)).sum().item()
            total += y.size(0)
            progress_bar.set_postfix({'validate_loss': '{:.3f}'.format(loss.item()/len(x1))})
            y_true.extend(y.tolist())
            y_pred.extend(predicted.tolist())

    accuracy = correct / total
    average_loss = total_loss / len(data_loader)
    auc = roc_auc_score(y_true, y_pred)

    return average_loss, accuracy, auc

if __name__ == '__main__':
    # 获取蛋白质特征,然后自动结束程序
    from torch.utils.data import DataLoader
    import numpy as np

    # Create dataset
    dataset = PPIPredictionDataset('./anc2vec_Human.csv', './handcrafted_Human.csv', './all_protein.csv')

    # Create data loader
    batch_size = 4096 * 16
    num_worker = 22
    test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_worker, pin_memory=True)

    num_epochs = 300
    learning_rate = 3e-4
    model = PPIPredictionModel(input_dim=340, hidden_dim=256, num_layers=2, num_heads=4)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    model.load_state_dict(torch.load('./transformer_outputs/train_0.9681-val_0.9473-test_0.9477.pth')) # 加载模型
    # best_model_params = model.state_dict()
    best_model_params = None

    test_loss, test_accuracy, test_auc = test(model, test_loader)