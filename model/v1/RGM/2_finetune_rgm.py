import os
import torch
from pytorch_lightning import loggers
import pytorch_lightning as pl
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.parameter_paser import parse_args_finetune
from src.dataset import ComplexDatasetLocs, generate_three_dataset_v3
from src.denoising_diffusion_process.samplers.DDPM import DDPM_Sampler
from src. pixel_diffusion import PixelDiffusionConditional_v2
from src import EMA


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.realpath(__file__))

    args = parse_args_finetune()
    print(f"\nUsing configuration file: {args.config}\n")

    input_dir = r"model\v1\input"
    output_dir = r"model\v1\output"
    os.makedirs(output_dir, exist_ok=True)

    frac_for_valid = args.frac_for_valid
    frac_for_test = args.frac_for_test

    # FLOOR3.pth
    data_path_area_1 = data_path = os.path.join(input_dir, args.data_name)
    print(f"data_path_area_1: {data_path_area_1}")
    loaded = torch.load(data_path_area_1)
    
    rssi = loaded['rssi']
    snr = loaded['snr']
    label = loaded['label']

    location_vector_path = os.path.join(output_dir, args.location_vector_name)
    # 生成一个数据集, 32000个数据，每个数据有amplitude, phase, label, location_vector
    complex_dataset = ComplexDatasetLocs(rssi, 
                                         snr, 
                                         label, 
                                         location_vector_path
                                         )
    
    ratios_loctions = args.ratios_locs
    train_data_set, valid_data_set, test_data_set \
        = generate_three_dataset_v3(complex_dataset, ratios_loctions, 
                                    frac_for_valid, frac_for_test)

    input_dim = args.input_dim
    num_epochs = args.num_epochs_rgm_ft
    batch_si = args.batch_size_rgm
    learning_rate = args.learning_rate_rgm
    loc_dim = args.loc_dim
    num_timesteps = args.num_timesteps
    schedule = args.schedule
    model_loss = F.mse_loss
    data_channels = args.data_channels
    dimension_scale = args.channel_dimension_scale

    model_path_fintune_rgm = os.path.join(output_dir, args.rgm_fine_tune_path)
    # model_path_fintune_rgm_run = os.path.join(output_dir, f"finetuned_rgm_running.ckpt")
    
    loaded_pretrained_rgm = os.path.join(output_dir, args.rgm_pretrain_path)
    print("loaded_pretrained_rgm: ", loaded_pretrained_rgm)
    # loaded_pretrained_rgm=r"model\v1\output\lossmin\pretrain-sf-11.ckpt"
    # loaded_pretrained_rgm=r"model\v1\output\lossmin\val_loss_pretrain.ckpt"
    rgm_logs = os.path.join(output_dir, f"rgm_log")
    os.makedirs(rgm_logs, exist_ok=True)

    tb_logger = loggers.TensorBoardLogger(save_dir=rgm_logs, 
                                        name='',     
                                        version="rgm_finetune")

    sampler_ddpm = DDPM_Sampler(num_timesteps=num_timesteps, schedule=schedule)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     dirpath=output_dir, 
    #     filename=model_path_fintune_rgm_run,
    #     monitor="val_loss",
    #     mode="min",
    #     save_top_k=1,
    #     verbose=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="model\\v1\\output\\lossmin",
        filename="val_loss_finetune",  # seems does not used
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True)

    print("\nThe loaded diffusion model: {}\n".format(loaded_pretrained_rgm))

    model = PixelDiffusionConditional_v2.load_from_checkpoint(checkpoint_path=loaded_pretrained_rgm, 
                                                                train_dataset=train_data_set, 
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
                                                                sampler=sampler_ddpm)
    
    trainer = pl.Trainer(max_epochs=num_epochs, 
                        callbacks=[EMA(0.9999)], 
                        log_every_n_steps=10,
                        accelerator='gpu', 
                        devices=[0], 
                        check_val_every_n_epoch=1,
                        logger=tb_logger)
    train_loader = DataLoader(train_data_set, batch_size=batch_si, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(valid_data_set, batch_size=batch_si, shuffle=False, num_workers=4, persistent_workers=True)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    # 新增早停回调（监控 val_loss）
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",    # 监控验证损失
        patience=50,           # 连续10个epoch未改善则停止
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

    trainer.save_checkpoint(model_path_fintune_rgm)



        
