import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from src.dataset import RealorFakeDataset
# 加载数据文件
genrated_data_pth = r"model\v1\output\floor3_sf_11_fake.pth"
data = torch.load(genrated_data_pth) 

# 转换为Numpy数组并处理数据类型
features = data.features.cpu().numpy()
features = np.asarray(features, dtype=np.float32)  # 确保为浮点型
labels = data.labels.cpu().numpy().astype(int)   # 强制转为整型标签

# 动态调整t-SNE参数
n_samples = len(features)
perplexity = min(30, max(5, n_samples // 5))  # 自适应调整
tsne = TSNE(n_components=2, random_state=42, 
            perplexity=perplexity, n_iter=1000, 
            learning_rate=200, init='random')

# 运行t-SNE降维
tsne_results = tsne.fit_transform(features)

# 可视化
plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=tsne_results[:, 0], y=tsne_results[:, 1],
    hue=labels,
    palette=sns.color_palette("hsv", len(np.unique(labels))),
    legend="full",
    s=50,  # 点大小
    alpha=0.8  # 透明度
)
plt.title("Generated Data Feature t-SNE Projection")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("tsne_visualization.png", dpi=300)  # 保存高分辨率图像
plt.show()