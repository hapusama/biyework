import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import griddata
input_data_path = "data/input_data.txt" # 这个是到时候用来定位的指纹数据
fake_data_path= [r"model\v1\output\floor3_sf_11_fake.pth"]   #生成指纹库的路径
location_vector_path=r"model\v1\output\location_vector_v3.csv"
savefig_path = r"model\v1\output\interpolated_feature.png"
X_interpolation_idx = [0, 1, 2,3,4,13,14,15,16,17]
Y_interpolation_idx = [6, 7,8,9,10,11]
tp=2
sf=11
def reload_fake_data(fake_data_path):
    for path in fake_data_path:
        fake_data = torch.load(path)
        match_and_generate_csv(fake_data, location_vector_path, path.replace('.pth', '_output.csv'))

def match_and_generate_csv(dataset, location_vector_path, output_csv_path):
    # Load location vector CSV  
    location_vector_df = pd.read_csv(location_vector_path)
    
    # Prepare a list to store the new rows
    new_rows = []
    
    # Iterate through the dataset
    for idx in range(len(dataset)):
        feature, label = dataset[idx]
        # 将label转换为整数
        label = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
        # Find the matching row in the location vector CSV
        matching_row = location_vector_df[location_vector_df['idx'] == label]
        
        if not matching_row.empty:
            true_x = matching_row.iloc[0]['true_x']
            true_y = matching_row.iloc[0]['true_y']
            
            # Append the new row with additional columns
            # features是个tensor数组，把它转换为numpy数组
            feature = feature.numpy() if isinstance(feature, torch.Tensor) else feature
            new_rows.append({
                'feature': feature,
                'label': label,
                'true_x': true_x,
                'true_y': true_y
            })
    
    # Create a new DataFrame with the new rows
    new_df = pd.DataFrame(new_rows)
    
    # Save the new DataFrame to a CSV file
    new_df.to_csv(output_csv_path, index=False)
    
def interpolation(df,nums):
    x=np.stack(df['true_x'].apply(
            lambda x: np.fromstring(x[1:-1], sep=' ') if isinstance(x, str) else x))
    y=np.stack(df['true_y'].apply(
        lambda x: np.fromstring(x[1:-1], sep=' ') if isinstance(x, str) else x))
    features = np.stack(df['feature'].apply(
        lambda x: np.fromstring(x[1:-1], sep=' ')[:-2] if isinstance(x, str) else x[:-2]
    ))
    # 去掉列里面含有正负一的行
    mask = ~np.any(np.isin(features, [-1, 1]), axis=1)
    features = features[mask]
    x = x[mask]
    y = y[mask]
    # 定义插值网格 这两个都是二维数组用于组合成网格
    grid_x, grid_y = np.mgrid[min(x):max(x):nums, min(y):max(y):nums]
    
    # 二维插值
    # 使用克里金插值方法
    interpolated_data = []
    for i in range(features.shape[1]):  # 对每个特征维度进行插值
        ok = OrdinaryKriging(
        x, y, features[:, i], 
        variogram_model='linear', 
        verbose=True, 
        enable_plotting=False, 
        )
        grid_feature, _ = ok.execute('grid', grid_x[:, 0], grid_y[0, :])
        for gx, gy, gf in zip(grid_x.flatten(), grid_y.flatten(), grid_feature.flatten()):
            interpolated_data.append([gx, gy, i, gf])
    
    # 创建新的DataFrame，将每个特征维度的值单独保存为一列
    result=[]
    
    df = pd.DataFrame(interpolated_data, columns=['x', 'y', 'feature_idx', 'value'])
    grouped=df.groupby(['x','y'])
    for (x,y) , group in grouped:
        group = group.sort_values(by='feature_idx')
        # 将 value 转换为四维数组
        feature_array = group['value'].to_numpy()
        result.append({'feature': feature_array, 'x': x, 'y': y})
    return pd.DataFrame(result,columns=['feature', 'x', 'y'])
    
    
if __name__=="__main__":
    reload_fake_data(fake_data_path)
    # 读取生成的csv文件
    location_vector=pd.read_csv(location_vector_path)
    # 遍历每个生成data的csv文件
    # for path in fake_data_path:
    #     output_csv_path = path.replace('.pth', '_output.csv')
    #     interpolated_features=[]
    #     df = pd.read_csv(output_csv_path)
    #     # 对每个点的每五十行数据的feature数组求平均值
    #     grouped = df.groupby('label')
    #     processed_rows = []
    #     steps=1
    #     for label, group in grouped:
    #         for i in range(0, len(group), steps):  # 每五十行数据分组
    #             subset = group.iloc[i:i + steps]
    #             avg_feature = np.mean(
    #                 np.stack(subset['feature'].apply(
    #                 lambda x: np.fromstring(x[1:-1], sep=' ') if isinstance(x, str) else x
    #                 )),
    #                 axis=0
    #             )
    #             processed_rows.append({
    #                 'label': label,
    #                 'true_x': subset['true_x'].iloc[0],
    #                 'true_y': subset['true_y'].iloc[0],
    #                 'feature': avg_feature
    #             })
        
    #     # 创建新的DataFrame
    #     df = pd.DataFrame(processed_rows)
        
    #     print(f"Data from {output_csv_path}:")
    #     print(df.head())
        
    #     # todo：现在的插值有个弊端，学不到特征内部的关系
    #     # 进行插值
    #     df=interpolation(df,nums=200j)
        
    #     # 过滤掉不符合条件的点
    #     distance_threshold = 5
    #     x_distance_threshold = 2
    #     y_distance_threshold = 2
    #     filtered_data=[]
    #     for _,row in df.iterrows():
    #         x,y = row['x'], row['y']
    #         x_condition = False
    #         y_condition = False
    
    #         # 遍历 Y_interpolation_idx,计算当前点与每个点的距离
    #         for i in Y_interpolation_idx:
    #             matching_row = location_vector[location_vector['idx'] == i]
    #             matching_point_coordinates=(matching_row.iloc[0]['true_x'], matching_row.iloc[0]['true_y'])
    #             nodes2=(x,y)
    #             distance=np.linalg.norm(np.array(matching_point_coordinates)-np.array(nodes2))
    #             if not matching_row.empty and abs(matching_row.iloc[0]['true_x']-x) < x_distance_threshold and distance < distance_threshold:
    #                 x_condition = True
    #                 break
    #         for i in X_interpolation_idx:
    #             matching_row = location_vector[location_vector['idx'] == i]
    #             matching_point_coordinates=(matching_row.iloc[0]['true_x'], matching_row.iloc[0]['true_y'])
    #             nodes2=(x,y)
    #             distance=np.linalg.norm(np.array(matching_point_coordinates)-np.array(nodes2))
    #             if not matching_row.empty and distance < distance_threshold and abs(matching_row.iloc[0]['true_y']-y) < y_distance_threshold:
    #                 y_condition = True
    #                 break
    #         if x_condition or y_condition:
    #             filtered_data.append(row)
    #     df = pd.DataFrame(filtered_data,columns=['feature', 'x', 'y'])   
    #     # 能不能可视化绘制一下目前df里面的点的坐标
    #     fig = plt.figure(figsize=(8, 6))
    #     ax = fig.add_subplot(111, projection='3d')
    #     # 这里只能从feature里面取出一个值来画图
    #     ax.scatter(df['x'], df['y'], df['feature'].apply(lambda x: x[1]), c='r', marker='o')
    #     ax.set_xlabel('X Coordinate') 
    #     # 保存图像
    #     plt.savefig(savefig_path.replace('.png', f'_{tp}_{sf}.png'))
    #     plt.show()
    #     df.to_csv(path.replace('.pth', '_interpolated_output.csv'), index=False)
