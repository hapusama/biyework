# todo:
# 2. 跨楼层如何做，每一层的rssi阈值判断？
# 3. 输入是什么？ 每一层网关的rssi + snr + average_rssi
## [{gateway_id: 1, rssi: -50, snr: 20, average_rssi: -55}, {gateway_id: 2, rssi: -60, snr: 25, average_rssi: -65}...]
# 4. 输出是什么？ 
## 真实坐标x,y + 楼层

#需要实现什么？
## 4. knn or mlp来进行坐标的预测
### 首先通过阈值判断来确定floor_id 然后丢给对应的模型进行预测
## 5. 输出来得到floor_id和坐标x,y

#实验验证流程：
## 1. floor3 sf11负责pretrain扩散模型
## 2. 少量floor2 sf11负责finetune扩散模型
## 3. finetune后模型用于floor2的指纹生成
## 4. 分别对floor2和floor3的指纹库进行插值 (fingerprint,label) -> (fingerprint,coordinate)
## 5. 对floor2和floor3的指纹库进行knn / mlp 训练 分别保存对应的模型
## 6. 定位部分：阈值判断为哪一层之后，将rssi snr average_rssi传入对应的模型进行预测
## 7. 输出坐标x,y和floor_id

#可能的优化：
## 1. generate过程直接生成坐标x,y和floor_id 对应的指纹
## 2. unet优化