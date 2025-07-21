import os
import matplotlib.pyplot as plt

import pandas as pd
from classifier_finger import *
from src.parameter_paser import parse_args_finetune
import numpy as np
from collections import Counter
adjacent_list = [0, 1, 2, 3, 4, 5, 6, 7,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20]
# 计算两点之间的距离
def distance(a, b):
    ax, ay = coords[a]['true_x'], coords[a]['true_y']
    bx, by = coords[b]['true_x'], coords[b]['true_y']
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

# 随机生成一条路线，点可重复，但每个点最多出现两次，总点数不超过max_length，且相邻点距离不超过max_dist米
def generate_route(points, max_dist=17, max_length=20):
    # 每个点最多出现两次，必须覆盖所有点至少一次
    remaining = set(points)
    route = [np.random.choice(points)]
    remaining.discard(route[-1])
    counter = Counter(route)
    while (remaining or any(counter[p] < 2 for p in points)) and len(route) < max_length:
        # 只在未超出出现次数限制的点中选，且距离不超过max_dist
        candidates = [p for p in points if counter[p] < 2 and distance(route[-1], p) <= max_dist]
        if not candidates:
            break
        next_point = np.random.choice(candidates)
        route.append(next_point)
        counter[next_point] += 1
        remaining.discard(next_point)
    return route
if __name__=="__main__":
    args = parse_args_finetune()
    save_model_name_fake = args.save_model_name_fake
    input_dir = r"model\v1\input"
    output_dir = r"model\v1\output"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_dim = args.input_dim + 2  # 加2是因为有两个额外的坐标维度
    num_classes = args.num_locs
    finetune_data_name = args.route_data_name
    finetune_data_path = os.path.join(input_dir , finetune_data_name)
    loaded_data = torch.load(finetune_data_path)
    rssi = loaded_data['rssi']
    sf_tp = loaded_data['snr']
    label = loaded_data['label']
    # 将RSSI和sf_tp拼接
    input_data = np.concatenate((rssi, sf_tp), axis=1)
    
    model_check_point_pth = os.path.join(output_dir, save_model_name_fake)
    print(f"model_check_point_pth: {model_check_point_pth}")
    model = LocationClassifier(input_dim, num_classes)
    
    # load the model
    model.load_state_dict(torch.load(model_check_point_pth, map_location=device))
    model.to(device)
    model.eval()
    
    location_vector = os.path.join(output_dir, args.location_vector_name)
    location_vector = pd.read_csv(location_vector)
    dataset_path = os.path.join(output_dir, args.data_name_fake)
    
    loaded_data = torch.load(dataset_path)
    
    # 获取所有点的坐标
    coords = location_vector.set_index('idx')[['true_x', 'true_y']].to_dict('index')
    
    # 计算每个相邻点之间的平均距离
    avg_distances = []
    for i in range(len(adjacent_list) - 1):
        dist = distance(adjacent_list[i], adjacent_list[i + 1])
        avg_distances.append(dist)
    print("Average distances between adjacent points:", np.mean(avg_distances))
    input_data_list = []
    # 生成一条合法路线
    route_list =[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
    for route in route_list:
        route = generate_route(adjacent_list)
        print("Generated route:", route)
        route_data = []
        for point in route:
            # 找到所有label等于point的索引
            indices = np.where(label == point)[0]
            if len(indices) == 0:
                continue  # 如果没有对应的input data则跳过
            # 随机选择一个索引
            chosen_idx = np.random.choice(indices)
            # 取出对应的input data
            route_data.append(input_data[chosen_idx])
        # 将route data中的每一条数据输入model
        predicted_route = []
        with torch.no_grad():
            for data in route_data:
                last_point = predicted_route[-1] if predicted_route else None
                data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
                output = model(data_tensor)
                probs = torch.softmax(output, dim=1)
                topk_probs, topk_labels = torch.topk(probs, k=probs.shape[1], dim=1)

                topk_labels = topk_labels.view(-1).cpu().numpy()
                topk_probs = topk_probs.view(-1).cpu().numpy()
                chosen_label = topk_labels[0]

                if last_point is not None:
                    for idx in range(len(topk_labels)):
                        candidate_label = topk_labels[idx]
                        if distance(last_point, candidate_label) <= 17:
                            chosen_label = candidate_label
                            break
                predicted_route.append(chosen_label)
        print("Predicted route:", predicted_route)

       