import numpy as np
import math
from typing import Tuple, List

class FFZ_modeling:
    def __init__(self, 
                 floor_height: float, 
                 room_length: float, 
                 room_width: float, 
                 corridor_width: float,
                 wall_thickness: float = 0.2,
                 frequency: float = 868e6):
        """
        初始化FFZ建模类
        
        参数:
        floor_height: 楼层高度(米)
        room_length: 办公室长度(米)
        room_width: 办公室宽度(米)
        corridor_width: 走廊宽度(米)
        wall_thickness: 墙体厚度(米)，默认0.2米
        frequency: LoRa信号频率(Hz)，默认868MHz
        """
        self.floor_height = floor_height
        self.room_length = room_length
        self.room_width = room_width
        self.corridor_width = corridor_width
        self.wall_thickness = wall_thickness
        self.lambda_ = 3e8 / frequency  # 波长计算(光速/频率)
        
        # 计算总宽度 (左侧房间 + 走廊 + 右侧房间)
        self.total_width = 2 * room_width + corridor_width

    def calculate_unit_vector(self, 
                             node_pos: List[float], 
                             gateway_pos: List[float]) -> Tuple[np.ndarray, float]:
        """
        计算节点到网关的3D单位向量和距离
        
        参数:
        node_pos: 节点位置[x, y, z]
        gateway_pos: 网关位置[x, y, z]
        
        返回:
        unit_vector: 单位向量
        distance: 两点间距离
        """
        vector = np.array(gateway_pos) - np.array(node_pos)
        distance = np.linalg.norm(vector)
        if distance == 0:
            return np.array([0, 0, 0]), 0
        unit_vector = vector / distance
        return unit_vector, distance

    def _get_media_type(self, point: List[float]) -> str:
        """
        判断给定点的介质类型(内部方法)
        
        参数:
        point: 点的3D坐标[x, y, z]
        
        返回:
        media_type: 'air', 'wall' 或 'office'
        """
        x, y, z = point
        
        # 1. 检查是否在垂直墙体区域 (房间与走廊之间)
        left_wall_min = self.room_width - self.wall_thickness/2
        left_wall_max = self.room_width + self.wall_thickness/2
        right_wall_min = self.room_width + self.corridor_width - self.wall_thickness/2
        right_wall_max = self.room_width + self.corridor_width + self.wall_thickness/2
        
        if (left_wall_min <= x <= left_wall_max) or (right_wall_min <= x <= right_wall_max):
            return 'wall'
        
        # 2. 检查是否在水平墙体区域 (房间之间)
        # 计算y方向的周期(房间长度+墙体厚度)
        period = self.room_length + self.wall_thickness
        # 计算在周期内的相对位置
        y_mod = y % period
        
        # 如果y_mod在墙体区域
        if y_mod > self.room_length:
            return 'wall'
        
        # 3. 检查是否在办公室区域
        # 左侧办公室区域
        left_office = (0 <= x <= self.room_width)
        # 右侧办公室区域
        right_office = (self.room_width + self.corridor_width <= x <= self.total_width)
        
        if left_office or right_office:
            return 'office'
        
        # 4. 否则在走廊区域(空气)
        return 'air'

    def calculate_ffz_media_ratios(self, 
                                  node_pos: List[float], 
                                  gateway_pos: List[float],
                                  num_segments: int = 20,
                                  samples_per_segment: int = 50) -> Tuple[float, float, float]:
        """
        计算FFZ内传播介质比例
        
        参数:
        node_pos: 节点位置[x, y, z]
        gateway_pos: 网关位置[x, y, z]
        num_segments: FFZ分割段数
        samples_per_segment: 每段采样点数
        
        返回:
        air_ratio: 空气介质比例
        wall_ratio: 墙体介质比例
        office_ratio: 办公室介质比例
        """
        unit_vector, total_distance = self.calculate_unit_vector(node_pos, gateway_pos)
        
        # 如果节点和网关在同一位置，返回默认值
        if total_distance == 0:
            return 0.0, 0.0, 0.0
        
        # 初始化介质计数器
        air_count = 0
        wall_count = 0
        office_count = 0
        total_samples = num_segments * samples_per_segment
        
        # 创建垂直于传播方向的坐标系
        w = unit_vector
        # 选择一个与w不共线的向量
        t = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        # 计算u (与w垂直的单位向量)
        u = t - np.dot(t, w) * w
        u /= np.linalg.norm(u)
        # 计算v (与w和u都垂直的单位向量)
        v = np.cross(w, u)
        v /= np.linalg.norm(v)
        
        # 沿传播路径分段采样
        for i in range(num_segments):
            # 计算当前段中心的位置 (沿传播方向)
            segment_center = i / num_segments + 0.5 / num_segments
            point_on_line = np.array(node_pos) + segment_center * (np.array(gateway_pos) - np.array(node_pos))
            
            # 计算当前点到节点和网关的距离
            d1 = segment_center * total_distance
            d2 = total_distance - d1
            
            # 计算当前点的第一菲涅尔区半径
            radius = math.sqrt((self.lambda_ * d1 * d2) / (d1 + d2))
            
            # 在当前横截面采样
            for _ in range(samples_per_segment):
                # 在圆盘内随机生成点 (极坐标)
                r = radius * math.sqrt(np.random.random())  # 均匀分布在圆内
                theta = 2 * math.pi * np.random.random()
                
                # 计算偏移向量 (在u-v平面)
                offset = r * math.cos(theta) * u + r * math.sin(theta) * v
                
                # 计算采样点全局坐标
                sample_point = point_on_line + offset
                
                # 判断介质类型并计数
                media_type = self._get_media_type(sample_point)
                if media_type == 'air':
                    air_count += 1
                elif media_type == 'wall':
                    wall_count += 1
                elif media_type == 'office':
                    office_count += 1
        
        # 计算比例
        air_ratio = air_count / total_samples
        wall_ratio = wall_count / total_samples
        office_ratio = office_count / total_samples
        
        return air_ratio, wall_ratio, office_ratio

    def get_loc_vector(self, 
                      node_pos: List[float], 
                      gateway_pos: List[float]) -> List[float]:
        """
        获取位置向量
        
        参数:
        node_pos: 节点位置[x, y, z]
        gateway_pos: 网关位置[x, y, z]
        
        返回:
        loc_vector: [X, Y, Z, distance, air_ratio, wall_ratio, office_ratio]
        """
        unit_vector, distance = self.calculate_unit_vector(node_pos, gateway_pos)
        air_ratio, wall_ratio, office_ratio = self.calculate_ffz_media_ratios(node_pos, gateway_pos)
        
        return [
            unit_vector[0],  # X分量
            unit_vector[1],  # Y分量
            unit_vector[2],  # Z分量
            distance,        # 距离
            air_ratio,       # 空气比例
            wall_ratio,      # 墙体比例
            office_ratio     # 办公室比例
        ]

# 使用示例
if __name__ == "__main__":
    # 参数设置 (单位：米)
    floor_height = 3.0
    room_length = 10.0
    room_width = 8.0
    corridor_width = 2.0
    
    # 创建模型实例
    model = FFZ_modeling(floor_height, room_length, room_width, corridor_width)
    
    # 定义位置 (示例值)
    node_position = [5.0, 15.0, 1.5]    # 节点位置
    gateway_position = [15.0, 30.0, 2.0]  # 网关位置
    
    # 获取位置向量
    loc_vector = model.get_loc_vector(node_position, gateway_position)
    
    # 打印结果
    print("位置向量: [X, Y, Z, distance, air_ratio, wall_ratio, office_ratio]")
    print(loc_vector)