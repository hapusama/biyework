import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 定义路径损耗模型公式: RSSI = A - 10 * n * log10(d)
def path_loss_model(d, A, n):
    return A - 10 * n * np.log10(d)

fitting_data_path=r'data\processedData\FLOOR3\all_data.csv'
location_vector_path=r'model\v1\output\location_vector_v2.csv'
label_coordinate_path=r'PicdataProcessing\image.png'
area1_list=[0,1,2,3,4,5]
area2_list=[6,7,8,9,10,11,12,13,14]
area3_list=[15,16,17,18,19,20]

def fit_path_loss_per_sf(data_df, df, area_lists, area_names, colors, path_loss_model):
    sf_list = sorted(data_df['sf'].unique())
    for sf in sf_list:
        plt.figure(figsize=(8,6))
        print(f"\n=== SF={sf} ===")
        for area_idx, area_location_ids in enumerate(area_lists):
            all_distances = []
            all_rssis = []
            for i in area_location_ids:
                sample_df = data_df[(data_df['location_id'] == i) & (data_df['sf'] == sf)]
                if sample_df.empty:
                    continue
                distance = df[df['location_id'] == i]['distance_true'].values[0]
                rssi = sample_df['realtime_average_rssi'].values
                all_distances.extend([distance] * len(rssi))
                all_rssis.extend(rssi)
            if len(all_distances) == 0 or len(all_rssis) == 0:
                continue
            all_distances = np.array(all_distances)
            all_rssis = np.array(all_rssis)
            # 拟合
            popt, _ = curve_fit(path_loss_model, all_distances, all_rssis, p0=[-40, 2])
            # 绘制拟合曲线
            d_fit = np.linspace(all_distances.min(), all_distances.max(), 100)
            rssi_fit = path_loss_model(d_fit, *popt)
            plt.scatter(all_distances, all_rssis, color=colors[area_idx], alpha=0.3, label=f'{area_names[area_idx]} Data')
            plt.plot(d_fit, rssi_fit, color=colors[area_idx], label=f'{area_names[area_idx]} Fit: A={popt[0]:.2f}, n={popt[1]:.2f}')
            print(f'{area_names[area_idx]} 拟合参数: A={popt[0]:.2f}, n={popt[1]:.2f}')
        plt.xlabel('Distance (m)')
        plt.ylabel('RSSI (dBm)')
        plt.legend()
        plt.title(f'RSSI Path Loss Curve Fitting for Three Areas (SF={sf})')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    df = pd.read_csv(location_vector_path)
    idx_to_location_id = dict(zip(df['idx'], df['location_id']))
    area1_location_id_list = [idx_to_location_id[i] for i in area1_list]
    area2_location_id_list = [idx_to_location_id[i] for i in area2_list]
    area3_location_id_list = [idx_to_location_id[i] for i in area3_list]
    data_df = pd.read_csv(fitting_data_path)

    area_lists = [area1_location_id_list, area2_location_id_list, area3_location_id_list]
    area_names = ['Area 1', 'Area 2', 'Area 3']
    colors = ['r', 'g', 'b']

    fit_path_loss_per_sf(data_df, df, area_lists, area_names, colors, path_loss_model)
