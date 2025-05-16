import pandas as pd
import numpy as np

# 读取 CSV 文件
file_path = r'model\v1\output\location_vector_v3.csv'
df = pd.read_csv(file_path)

# 随机生成 -1 到 1 之间的值，确保区分度较大
np.random.seed(42)  # 固定随机种子以便复现
df['cross_wall'] = np.random.uniform(-1, 1, len(df))
df['cross_center'] = np.random.uniform(-1, 1, len(df))
df['is_x'] = np.random.uniform(-1, 1, len(df))
df['is_y'] = np.random.uniform(-1, 1, len(df))

# 保存修改后的 CSV 文件
df.to_csv(file_path, index=False)