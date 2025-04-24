# features->coordinates_localization
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
input_dim = 6  # Example input dimension
output_dim = 2  # Example output dimension (x, y coordinates)
batch_size = 256
interpolation_data_path = r"model\v1\output\floor3_sf_11_fake_interpolated_output.csv"
data_test_path=r"model\v1\input\coordinates_features_sf_11_floor3.csv"
selected_data_feature=['rssi','average_rssi','rssi','snr']

class SimpleLocalizer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLocalizer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

if __name__=="__main__":
    # 加载插值后的数据
    interpolation_df=pd.read_csv(interpolation_data_path)
    test_df=pd.read_csv(data_test_path) 
    
    # 选择特征列 
    interpolation_features = np.array([np.fromstring(f.strip('[]'), sep=' ') for f in interpolation_df['feature']]) # Convert string to array
    interpolation_coord = interpolation_df[['x','y']]
    
    test_features = test_df[selected_data_feature].values 
    test_coords = test_df[['true_x','true_y']]
    # 3. 使用 KNN 进行匹配
    knn = NearestNeighbors(n_neighbors=40, algorithm='kd_tree')
    knn.fit(interpolation_features)
    distances, indices = knn.kneighbors(test_features)
    predicted_coords = interpolation_coord.iloc[indices.flatten()].reset_index(drop=True)

    # 4. 计算误差
    test_coords = test_coords.reset_index(drop=True)
    predicted_coords = predicted_coords.groupby(predicted_coords.index // knn.n_neighbors).mean().reset_index(drop=True)

    errors = np.sqrt((test_coords['true_x'] - predicted_coords['x'])**2 + 
                    (test_coords['true_y'] - predicted_coords['y'])**2)
    # 5. 生成结果 DataFrame
    results = pd.DataFrame({
        'true_x': test_coords['true_x'],
        'true_y': test_coords['true_y'],
        'predicted_x': predicted_coords['x'],
        'predicted_y': predicted_coords['y'],
        'error': errors
    })

    # 6. 保存结果为新的 CSV 文件
    results.to_csv("knn_matching_results.csv", index=False)
    # 计算平均误差
    mean_error = results['error'].mean()
    print(f"平均误差: {mean_error:.2f} 米")
    # 计算最大误差
    max_error = results['error'].max()
    print(f"最大误差: {max_error:.2f} 米")
    # 计算最小误差
    min_error = results['error'].min()
    print(f"最小误差: {min_error:.2f} 米")
    print("KNN 匹配完成，结果已保存到 knn_matching_results.csv")
