# MMCA: Multi-Modal Co-Attention for Drug–Target Interaction Prediction

本目录已按 [Join-xiaobai/MMCA](https://github.com/Join-xiaobai/MMCA) 的 `main` 分支（当前提交 `16e0b9b70d561b5496f3e9c8b1610f0b1dc4c58c`）整理，并补回了本地已有的完整特征数据。`MMCA.py` 是当前入口。

## 先回答你提出的四个问题

### 1. 运行 MMCA 所需的完整数据文件

以下三个文件已经放在 `data/`，可以直接被 `MMCA.py`、`test.py` 读取：

| 文件 | 作用 | 当前内容 |
|---|---|---:|
| `data/drug_feature.csv` | 药物节点特征；第 1 列是 `drug_name` | 11,151 个药物，64 个数值特征 |
| `data/protein_feature.csv` | 蛋白节点特征；第 1 列是 `protein` | 18,465 个蛋白，340 个数值特征 |
| `data/drug_protein.csv` | 已知药物–蛋白相互作用；列为 `Drug,Protein` | 13,856 条边 |

数据文件是本机 `C:\Users\join\Desktop\1` 中的完整版本，已整理成 MMCA 可直接读取的 CSV；上游仓库同时提供了对应的分卷归档。运行校验：

```powershell
cd C:\Users\join\Desktop\1\MMCA
python scripts\validate_data.py
```

校验脚本会检查列名、维度、数值是否有限、ID 是否重复，以及相互作用中的 ID 是否都能在特征表中找到。

### 2. 正确或较新的实现

当前实现来自 GitHub `main`，并做了两项启动修复：

* `MMCA.py` 和 `test.py` 使用合法的 `nn.Identity()` 初始化方式。
* `test.py` 支持五个随机种子 checkpoint 和训练生成的 `bestModel/best_model.pt`。

训练脚本默认使用随机种子 50、300 个 epoch，并把最佳模型保存到 `bestModel/best_model.pt`。`test.py` 和 `crossTest.py` 是仓库中用于评估的脚本。

### 3. 药物和蛋白特征提取过程

可复用的脚本与现有中间文件已整理到 `feature_extraction/`。

#### 药物特征

1. `feature_extraction/drug/data/drug_mol2vec_features.csv` 保存 Mol2vec 特征；`drug_embeddings.csv` 保存图/嵌入输入；`all_drug.csv` 保存药物关系数据。
2. `feature_extraction/drug/gcn.py` 读取这些文件，将嵌入与 Mol2vec 特征拼接，加载 `feature_extraction/drug/bestModel/best_model.pt`，用 GCN 得到 64 维表示。
3. 原脚本将结果临时写到无 ID、无表头的 `p_score.csv` 并退出；将药物 ID 和列名补回后得到本项目的 `data/drug_feature.csv`（`drug_name,0,...,63`）。
4. `convert_xlsx_to_csv.py` 只负责把 Mol2vec 的 XLSX 中间文件转成 CSV。

#### 蛋白特征

1. `feature_extraction/protein/transformer.py` 将 Anc2vec 序列嵌入和 handcrafted 特征拼成 340 维输入。
2. 它需要 `anc2vec_Human.csv`、`handcrafted_Human.csv`、`all_protein.csv`，以及 `transformer_outputs/train_0.9681-val_0.9473-test_0.9477.pth` 这个预训练 Transformer checkpoint。
3. 模型池化后的向量会写成无 ID、无表头的 `p_score.csv`；补回蛋白 ID 和列名后得到本项目的 `data/protein_feature.csv`（`protein,0,...,339`）。

本地目录中已提供最终的 `protein_feature.csv`、`all_protein.csv` 和蛋白侧脚本。若要从蛋白原始序列重新生成特征，请使用 `anc2vec_Human.csv`、`handcrafted_Human.csv` 和对应的 Transformer checkpoint，并按上述流程生成 340 维表示。

### 4. BioSNAP、BindingDB、Human 的环境和复现实验说明

BioSNAP、BindingDB 和 Human 数据集均使用相同的 MMCA 训练与评估流程。分别准备对应数据集的药物特征、蛋白特征、药物–蛋白相互作用表、数据划分文件和 checkpoint，并在每个数据集目录中运行以下命令：

```powershell
cd C:\Users\join\Desktop\1\MMCA
python scripts\validate_data.py
python MMCA.py                 # 训练；生成 bestModel\best_model.pt
python test.py                 # 测试
python crossTest.py            # 仅提供五折指标汇总函数/示例
```

将每个数据集的文件分别放入独立目录，并让脚本中的 `data/`、特征维度、负采样比例和 `RandomLinkSplit` 划分与论文设置保持一致。固定随机种子，保存每次划分的负样本和 checkpoint，并统一记录 AUC、AUPR、F1、precision、recall、sensitivity、specificity。

## 环境

推荐使用 Python 3.10、PyTorch 2.1.2 和 PyTorch Geometric 2.4.x。PyG 的 `torch_scatter`、`torch_sparse` 等扩展必须与 PyTorch 和 CUDA/CPU 版本匹配。

```powershell
conda create -n mmca python=3.10 -y
conda activate mmca

# GPU 示例（CUDA 12.1）；CPU 机器请改用 PyTorch 官方 CPU index-url
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv `
  -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
python -m pip install torch-geometric==2.4.0
python -m pip install -r requirements.txt
```

至少需要 `pandas`、`numpy`、`scikit-learn`、`tqdm`、`tabulate`、`torch`、`torch-geometric`、`torch-scatter` 和 `torch-sparse`。完整训练对显存和时间要求较高；先运行数据校验和一个短 epoch smoke test，再进行 300 epoch 训练。

## 文件结构

```text
MMCA/
├── MMCA.py
├── test.py
├── crossTest.py
├── data/
│   ├── drug_feature.csv
│   ├── protein_feature.csv
│   └── drug_protein.csv
├── feature_extraction/
│   ├── drug/       # Mol2vec、药物嵌入、GCN 中间文件和脚本
│   └── protein/    # 蛋白 Transformer 脚本和现有中间文件
├── scripts/validate_data.py
├── requirements.txt
├── COVID-19.zip
└── COVID-19_drug_protein.csv
```

## 数据完整性记录

SHA-256（用于校验传输或解压完整性）：

```text
data/drug_feature.csv    80C530DF0FC67834B8A2B82832D3A0EDDF025440D08A3A5D5D0405949DF60ED4
data/protein_feature.csv 447639B88D1B7CDAADC3E0A5245546181ADA41284EB1B69AEDCA3B6D44D21BFF
data/drug_protein.csv    CB197A5E6C309BFD4A23283A6354B90F714A0979191F746A49ABC94B3E618D80
```

## 许可证与上游

代码和原始仓库信息以 GitHub 项目为准：[https://github.com/Join-xiaobai/MMCA](https://github.com/Join-xiaobai/MMCA)。请在发表或再分发数据时保留上游许可证、数据集许可证和原始论文引用。
