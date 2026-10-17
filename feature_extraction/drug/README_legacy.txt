DTI_drug_feature.csv：是最终的DTI的特征（加上药物名字和列名）

p_score.csv：是模型保存的特征（没有药物名和列名）

备注：重新使用mol2vec塑造了所有药物的特征 feature 然后与 embeddings 的特征输入到训练好的模型中
生成新的特征。