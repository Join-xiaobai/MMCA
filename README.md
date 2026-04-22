# MMCA：多模态协同注意力药物-靶点相互作用预测模型
基于多模态协同注意力机制（Multi-Modal Co-Attention）实现高精度药物-靶点相互作用（DTI）预测，同时支持 COVID-19 相关药物挖掘场景。

---

## 仓库说明
本项目包含 MMCA 模型完整代码、数据集与运行脚本，基于 **MIT 协议**开源。
GitHub 地址：https://github.com/Join-xiaobai/MMCA

## 运行环境
- Python 3.9+
- PyTorch 1.13+
- 支持 CPU / GPU 加速
- 依赖：pandas, torch, torch_geometric 等

## 文件结构
```
├── MMCA.py            # 模型训练主程序
├── test.py            # 模型测试与指标计算
├── crossTest.py       # 5 折交叉验证
├── datas.zip          # 实验数据集压缩包（COVID-19数据集和drug_protein.csv药物-靶点交互数据集）
├── COVID-19.zip       # 新冠相关预测数据、代码和结果
├── data/
│   ├── drug_feature.csv      # 药物特征文件
│   ├── protein_feature.csv   # 蛋白质特征文件
│   └── drug_protein.csv      # 药物-靶点交互数据
├── outputs/            # 训练日志、结果输出
├── bestModel/          # 最优模型保存路径
└── README.md
```

## 快速使用
1. 解压数据
```
unzip datas.zip
```
2. 训练模型
```
python MMCA.py
```
3. 测试模型（修改模型路径后运行）
```
python test.py
```
4. 5 折交叉验证
```
python crossTest.py
```

## 功能说明
- 药物特征：基于分子图结构 + Mol2vec 嵌入
- 蛋白质特征：基于 Anc2vec 序列嵌入 + 手工理化特征
- 多模态协同注意力融合药物与靶点信息
- 输出：AUC、AUPR、F1-score、精确率、召回率等指标
- 支持复现论文全部实验结果

## 超参数调整
可根据硬件配置修改以下参数：
- batch_size
- learning_rate
- num_neighbors
- 训练轮数、隐藏层维度等

---
