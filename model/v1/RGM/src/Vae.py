import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
class Residual(nn.Module):
    def __init__(self,fn):
        super(Residual, self).__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)
    


class ConditionalVAE(pl.LightningModule):
    def __init__(
        self,
        latent_dim=32,
        input_dim=6,
        condition_dim=8,
        encoder_layers=[128, 64],
        decoder_layers=[64, 128],
        activation=nn.ReLU,
        learning_rate=0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.condition_dim = condition_dim

        # 构建编码器
        encoder_dims = [input_dim + condition_dim] + encoder_layers
        encoder = []
        for in_dim, out_dim in zip(encoder_dims[:-1], encoder_dims[1:]):
            encoder.append(nn.Linear(in_dim, out_dim))
            encoder.append(activation())
        self.encoder = nn.Sequential(*encoder)
        self.fc_mu = nn.Linear(encoder_layers[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_layers[-1], latent_dim)
        self.lr=learning_rate
        # 新增：初始化历史记录
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
        # 构建解码器
        decoder_dims = [latent_dim + condition_dim] + decoder_layers + [input_dim]
        decoder = []
        for in_dim, out_dim in zip(decoder_dims[:-2], decoder_dims[1:-1]):
            decoder.append(nn.Linear(in_dim, out_dim))
            decoder.append(activation())
        decoder.append(nn.Linear(decoder_dims[-2], decoder_dims[-1]))
        decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)
        
        # 条件编码
        self.condition_emb=nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, condition_dim*2),
            nn.GELU(),
            nn.Linear(condition_dim*2, condition_dim)
        )

    def encode(self, x, c):
        c = self.condition_emb(c)  # 对条件进行编码
        x_cond = torch.cat([x, c], dim=1) # (batch_size,input_dim*2 12)
        out = x_cond
        # out= self.encoder(out)
        for layer in self.encoder:
            if isinstance(layer, nn.Linear) and out.shape[-1] == layer.out_features:
                # 残差连接
                out = out + layer(out)
            else:
                out = layer(out)

        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar

    def decode(self, z, c):
        z_cond = torch.cat([z, c], dim=1)
        out = z_cond
        for layer in self.decoder:
            if isinstance(layer, nn.Linear) and out.shape[-1] == layer.out_features:
                # 残差连接
                out = out + layer(out)
            else:
                out = layer(out)
        return out

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def on_train_epoch_end(self):
        # 获取当前epoch的聚合损失值
        train_loss = self.trainer.callback_metrics['train_loss_epoch'].item()
        val_loss = self.trainer.callback_metrics['val_loss_epoch'].item()
        
        # 保存到历史记录
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)

        # 实时打印历史
        print(f"\nEpoch {self.current_epoch:02d} Summary:")
        print(f"Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}")
        print("History:")
        for i, (t_loss, v_loss) in enumerate(zip(self.history['train_loss'], self.history['val_loss'])):
            print(f"Epoch {i:02d}: {t_loss:.3f} / {v_loss:.3f}")

    def training_step(self, batch, batch_idx):
        x, c ,loc_id= batch
        recon_x, mu, logvar = self(x, c)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss
        loss=loss/batch[0].size(0)  # 平均损失
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, c, loc_id = batch
        recon_x, mu, logvar = self(x, c)    # 前向传播
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss
        loss=loss/batch[0].size(0)  # 平均损失
        self.log('val_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        #学习率下降策略
        # 使用AdamW优化器
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # 以ReduceLROnPlateau为例
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True),
            'monitor': 'val_loss_epoch',  # 监控的指标名要和log一致
            'interval': 'epoch',
            'frequency': 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}