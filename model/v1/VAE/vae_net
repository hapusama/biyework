import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

class VAE_net(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE_net, self).__init__()