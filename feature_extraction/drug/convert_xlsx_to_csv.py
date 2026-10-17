# 将 XLSX 文件转换为 CSV 文件并使用 UTF-8 编码

import pandas as pd

# 输入 XLSX 文件路径
input_file = './data/drug_mol2vec_features.xlsx'

# 输出 CSV 文件路径
output_file = './data/drug_mol2vec_features.csv'

# 读取 XLSX 文件
df = pd.read_excel(input_file)

# 将 DataFrame 保存为 CSV 文件,使用 UTF-8 编码
df.to_csv(output_file, index=False, encoding='utf-8')

print(f'XLSX file converted to CSV file successfully: {output_file}')