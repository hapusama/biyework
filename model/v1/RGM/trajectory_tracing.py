import os
import pandas as pd
from classifier_finger import *
from src.parameter_paser import parse_args_finetune
import numpy as np
adjacent_list = [0, 1, 2, 3, 4, 5, 6, 7,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20]
# 计算两点之间的距离
def distance(a, b):
    ax, ay = coords[a]['true_x'], coords[a]['true_y']
    bx, by = coords[b]['true_x'], coords[b]['true_y']
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

# 随机生成一条路线，点可重复，但总点数不超过30，且相邻点距离不超过4.5米
def generate_route(points, max_dist=4.5, max_length=30):
    route = [np.random.choice(points)]
    while len(route) < max_length:
        candidates = [p for p in points if distance(route[-1], p) <= max_dist]
        if not candidates:
            break
        next_point = np.random.choice(candidates)
        route.append(next_point)
    return route
if __name__=="__main__":
    args = parse_args_finetune()
    save_model_name_fake = args.save_model_name_fake
    input_dir = r"model\v1\input"
    output_dir = r"model\v1\output"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_dim = args.input_dim
    num_classes = args.num_classes
    model_check_point_pth = os.path.join(output_dir, save_model_name_fake)
    print(f"model_check_point_pth: {model_check_point_pth}")
    model = LocationClassifier(input_dim, num_classes)
    
    # load the model
    model.load_state_dict(torch.load(model_check_point_pth, map_location=device))
    model.to(device)
    model.eval()
    
    location_vector = os.path.join(output_dir, args.location_vector_name)
    location_vector = pd.read_csv(location_vector)
    dataset_path = os.path.join(input_dir, args.data_name_fake)
    
    loaded_data = torch.load(dataset_path)
    
    # 获取所有点的坐标
    coords = location_vector.set_index('idx')[['true_x', 'true_y']].to_dict('index')
    # 生成一条合法路线
    for i in range(10):
        route = generate_route(adjacent_list)
        print("Generated route:", route)
        route_data = []
        for point in route:
            

    