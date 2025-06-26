import numpy as np
import os
import pandas as pd
from src.parameter_paser import parse_args_location

# 返回一个向量 (x_direction, y_direction, distance)，表示节点和网关之间的方向
def calculate_location_vector(node_coord, gateway_coord):
    # 方向从节点指向网关，用于描述节点与网关之间的相对方位
    direction_vector = np.array(gateway_coord) - np.array(node_coord)
    norm = np.linalg.norm(direction_vector)
    if norm == 0:
        direction_vector = np.zeros_like(direction_vector)
    else:
        direction_vector = direction_vector / norm
    location_vector = [direction_vector[0], direction_vector[1], norm]
    return location_vector

if __name__ == '__main__':
    args = parse_args_location()
    print(f"\nUsing configuration file: {args.configs}\n")

    output_dir = r"model\v1\output"
    os.makedirs(output_dir, exist_ok=True)

    location_vector_path = os.path.join(output_dir, args.location_vector_name)
    # 网关坐标
    gateway_coord_m = (args.gateway_X, args.gateway_Y)
    with open(location_vector_path, "w") as in_file:
        # 写入 CSV 文件头
        in_file.write("location_id,x,y,distance,idx\n")
        # 遍历 label_coordinate.csv 文件，计算节点和网关之间的方向向量
        df = pd.read_csv(r"model\v1\input\label_coordinate_v2.csv")
        idx = 0
        for index, row in df.iterrows():
            node_id = row['id']
            node_coord = (row['x'], row['y'])
            if node_coord != gateway_coord_m:  # 排除网关坐标
                location_vector = calculate_location_vector(node_coord, gateway_coord_m)
                in_file.write(f"{node_id},{location_vector[0]},{location_vector[1]},{location_vector[2]},{idx}\n")
                idx += 1
    # 确保文件已经写入成功后再读取
    df = pd.read_csv(location_vector_path)
    # 将 distance 列归一化
    df['distance'] = (df['distance'] - df['distance'].mean()) / df['distance'].std()
    df.to_csv(location_vector_path, index=False)