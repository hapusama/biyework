import os
import torch
from pytorch_lightning import loggers
import pytorch_lightning as pl
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.parameter_paser import parse_args_pretrain
from src.dataset import generate_three_dataset_v2, ComplexDatasetLocs
from src.denoising_diffusion_process.samplers.DDPM import DDPM_Sampler
from src. pixel_diffusion import PixelDiffusionConditional_v2
from sklearn.preprocessing import MinMaxScaler
from src import EMA
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
torch.manual_seed(42)
np.random.seed(42)

if __name__ == '__main__':
    args = parse_args_pretrain()
    print(f"\nUsing configuration file: {args.config}\n")

    input_dir = r"model\v1\input"
    output_dir = r"model\v1\output"

    frac_for_valid = args.frac_for_valid
    frac_for_test = args.frac_for_test
    # FLOOR3.pth
    data_path_area_1 = data_path = os.path.join(input_dir, args.data_name)
    loaded = torch.load(data_path_area_1)
    rssi = loaded['rssi']   # shape [24576,7]
    snr = loaded['snr']
    label = loaded['label']
    
    location_vector_path = os.path.join(output_dir, args.location_vector_name)
    # 生成一个数据集, 32000个数据，每个数据有rssi, snr, label, location_vector
    complex_dataset = ComplexDatasetLocs(rssi, 
                                         snr, 
                                         label, 
                                         location_vector_path
                                         )
    
    train_data_set, valid_data_set, test_data_set = generate_three_dataset_v2(complex_dataset, 
                                                            frac_for_valid,
                                                            frac_for_test)

    input_dim = args.input_dim
    num_epochs = args.num_epochs_rgm
    batch_si = args.batch_size_rgm
    learning_rate = args.learning_rate_rgm
    loc_dim = args.loc_dim
    num_timesteps = args.num_timesteps
    schedule = args.schedule
    model_loss = F.mse_loss
    data_channels = args.data_channels
    # 通道的倍数 1 2 4 8
    dimension_scale = args.channel_dimension_scale
    # hidden_channels = args.hidden_input_channels
    # hidden_input_dim = args.hidden_input_len
    model_path_train_rgm = os.path.join(output_dir, args.rgm_pretrain_path)
    signal_feature_dim=args.signal_feature_dim
    rgm_logs = os.path.join(output_dir, f"rgm_log")
    os.makedirs(rgm_logs, exist_ok=True)

    tb_logger = loggers.TensorBoardLogger(save_dir=rgm_logs, 
                                        name='', 
                                        version="rgm_pretrain")
    # 降噪的过程（反向），加噪那一堆参数需要进一步了解，ddpm应该是降噪用的 下面那个是加噪的前向传播
    sampler_ddpm = DDPM_Sampler(num_timesteps=num_timesteps, schedule=schedule)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="model\\v1\\output\\lossmin",
        filename="val_loss_pretrain",  # seems does not used
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True)

    model = PixelDiffusionConditional_v2(train_data_set, 
                                        input_dim=input_dim, 
                                        loc_dim=loc_dim,
                                        channels=data_channels, 
                                        dim_mults=dimension_scale, 
                                        valid_dataset=valid_data_set, 
                                        batch_size=batch_si, 
                                        lr=learning_rate, 
                                        loss_fn=model_loss, 
                                        schedule=schedule, 
                                        num_timesteps=num_timesteps, 
                                        sampler=sampler_ddpm,signal_feature_dim=signal_feature_dim)

    train_loader = DataLoader(train_data_set, batch_size=batch_si, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(valid_data_set, batch_size=batch_si, shuffle=False, num_workers=4, persistent_workers=True)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    # 新增早停回调（监控 val_loss）
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",    # 监控验证损失
        patience=25,           # 连续10个epoch未改善则停止
        mode="min",            # 监控指标越小越好
        verbose=True           # 打印停止信息
    )

    trainer = pl.Trainer(max_epochs=num_epochs, 
                        callbacks=[EMA(0.9999),lr_monitor, checkpoint_callback,early_stop_callback], 
                        accelerator='gpu', 
                        devices=[0], 
                         enable_progress_bar=True,
                        check_val_every_n_epoch=1,
                        logger=tb_logger)
    trainer.fit(model, train_loader, val_loader)

    trainer.save_checkpoint(model_path_train_rgm)

        
