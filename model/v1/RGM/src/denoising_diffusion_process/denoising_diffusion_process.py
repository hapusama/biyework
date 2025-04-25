import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm.auto import tqdm

from .forward import *
from .samplers import *
from .backbones.simple_unet import *


class DenoisingDiffusionConditionalProcess(nn.Module):
    
    def __init__(self,
                 input_dim=4,
                 loc_dim=3,
                 channels=2,
                 dim_mults=(1, 2, 4, 8),
                 loss_fn=F.mse_loss,
                 schedule='linear',
                 num_timesteps=1000,
                 sampler=None,signal_feature_dim=8,sf_parameter=r"model\v1\output\sf_parameters.csv"):
        super().__init__()
        # self.trans_dim = signal_feature_dim #并非输入维度
        self.loc_dim = loc_dim
        self.channels = channels    #并非输入信道数
        self.loss_fn = loss_fn
        self.schedule = schedule
        self.num_timesteps = num_timesteps
        # Forward Process
        self.sf_parameters=sf_parameter
        self.forward_process = GaussianForwardProcess(num_timesteps=self.num_timesteps, 
                                                      schedule=self.schedule)
        
        # Reverse Process. defaults to a DDPM sampler if None is provided
        self.sampler = DDPM_Sampler(num_timesteps=self.num_timesteps, 
                                    schedule=self.schedule) if sampler is None else sampler
        
        # noise prediction networks
        self.model = UnetComplexBlock(input_dim, 
                                      loc_dim=loc_dim,
                                      channels=channels, 
                                      dim_mults=dim_mults,signal_feature_dim=signal_feature_dim)

    @torch.no_grad()
    def forward(self,
                data_shape,
                condition,
                sf,tp,true_distance,
                sampler=None,
                verbose=False):
        """
            forward() function triggers a complete inference cycle

            A custom sampler can be provided as an argument!
        """

        # read dimensions
        # b, c, h, w = x.shape
        b, h, w = data_shape
        device = next(self.model.parameters()).device
        condition = condition.to(device)
        
        # select sampler
        if sampler is None:
            sampler = self.sampler
            
        else:
            sampler.to(device)

        # time steps list
        num_timesteps = sampler.num_timesteps 
        it = reversed(range(0, num_timesteps))
        
        # x_t = torch.randn([b, c, h, w], device=device)
        x_t = torch.randn([b, h, w], device=device)

        for i in tqdm(it, desc='diffusion sampling', total=num_timesteps) if verbose else it:

            t = torch.full((b,), i, device=device, dtype=torch.long)
            z_t = self.model(x_t, t, condition,sf,tp,true_distance)   # prediction of noise

            # call forward function of DDPM_Sampler Class: 
            # Given approximation of noise z_t in x_t predict x_(t-1)
            # prediction of next state
            x_t = sampler(x_t, t, z_t)
        print(x_t.shape)
        # 物理模型
        df=pd.read_csv(self.sf_parameters)
        # 读取数据
        rows = df.set_index('sf').reindex(sf.cpu().numpy())  # 根据 sf 数组批量获取对应的参数
        
        if rows.isnull().values.any():
            print("没有找到对应的SF参数，请检查输入的SF值。")
            raise ValueError("没有找到对应的SF参数，请检查输入的SF值。")
        
        pl_0 = torch.tensor(rows['pl_0'].values, device=true_distance.device, dtype=true_distance.dtype)
        gamma_Sf = torch.tensor(rows['gamma_sf'].values, device=true_distance.device, dtype=true_distance.dtype)
        log_distance = torch.log10(true_distance)
        
        rssi_path_loss=self.path_loss_model(log_distance, gamma_Sf, pl_0,tp)
        rssi_path_loss=rssi_path_loss/150
        
        rate=0
        print("rssi_path_loss unique values:", torch.unique(rssi_path_loss))
        # 将 rssi_path_loss 加权到 out 的指定索引位置
        x_t[:, 0, [0, 1,2]] = (1 - rate) * x_t[:, 0, [0, 1,2]] + rate * rssi_path_loss.to(x_t.dtype).unsqueeze(-1)
   
        return x_t

    def path_loss_model(self, log_distance, gamma_Sf, pl_0, tp):
        # Ensure all inputs are tensors and have the same shape
        assert log_distance.shape == gamma_Sf.shape == pl_0.shape == tp.shape, \
            "All input tensors must have the same shape"
        return tp - pl_0 - 10 * gamma_Sf * log_distance
    
    # x is signal condition is location vector
    def p_loss(self, x, condition,sf,tp,true_distance):
        """
            Assumes output and input are in [-1,+1] range
            condition concludes sf and tp
        """        
        # batch size，channel(phase amplitude) input_dim
        # 当h与w不存在时，x.shape = (b, input_dim)
        if len(x.shape) == 2:
            b, h, w = x.shape[0], 1, x.shape[1]
            x = x.view(b, h, w)
        b,h,w = x.shape
        device = x.device
        #随机生成numsteps以内 256个不同整数
        t = torch.randint(0, self.forward_process.num_timesteps, (b,), device=device).long()

        # call forward function of GaussianForwardProcess Class:  
        # Get noisy sample at t given x_0
        output_noisy, noise = self.forward_process(x, t, return_noise=True)

        noise_hat = self.model(output_noisy, t, condition,sf,tp,true_distance)

        # apply loss    todo: 把mse换成交叉熵函数
        return self.loss_fn(noise, noise_hat)
    

    