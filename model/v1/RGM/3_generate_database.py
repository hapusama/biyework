import os
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from src.parameter_paser import parse_args_finetune
from src.dataset import ComplexDatasetLocs, ComplexDataset_real_imagary_v2
from src.denoising_diffusion_process.samplers.DDPM import DDPM_Sampler
from src. pixel_diffusion import PixelDiffusionConditional_v2
from src.utils import get_features_by_label_v4


if __name__ == '__main__':

    base_dir = os.path.dirname(os.path.realpath(__file__))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_args_finetune()
    print(f"\nUsing configuration file: {args.config}\n")

    input_dir = r"model\v1\input"
    output_dir = r"model\v1\output"

    num_locs = args.num_locs

    # using data from area b
    data_path_area_2 = data_path = os.path.join(input_dir, args.data_name_ft)
    loaded = torch.load(data_path_area_2)
    rssi = loaded['rssi']
    snr = loaded['snr']
    label = loaded['label']
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rssi = torch.tensor(scaler.fit_transform(rssi), dtype=torch.float32)
    snr = torch.tensor(scaler.fit_transform(snr), dtype=torch.float32)
    location_vector_path = os.path.join(output_dir, args.location_vector_name)

    complex_dataset = ComplexDatasetLocs(rssi, 
                                         snr, 
                                         label, 
                                         location_vector_path
                                         )
    
    input_dim = args.input_dim
    batch_si = args.batch_size_rgm
    learning_rate = args.learning_rate_rgm
    loc_dim = args.loc_dim
    num_timesteps = args.num_timesteps
    schedule = args.schedule
    model_loss = F.mse_loss
    data_channels = args.data_channels
    dimension_scale = args.channel_dimension_scale

    loaded_fine_tuned_rgm = os.path.join(output_dir, args.rgm_fine_tune_path)

    sampler_ddpm = DDPM_Sampler(num_timesteps=num_timesteps, schedule=schedule)

    print("\nThe loaded diffusion model: {}\n".format(loaded_fine_tuned_rgm))
    # 这里先不用 loaded_fine_tuned_rgm
    diffusion_model = PixelDiffusionConditional_v2.load_from_checkpoint(checkpoint_path=r"model\v1\output\lossmin\val_loss_pretrain.ckpt", 
                                                                train_dataset=complex_dataset, 
                                                                input_dim=input_dim, 
                                                                loc_dim=loc_dim, 
                                                                channels=data_channels, 
                                                                dim_mults=dimension_scale, 
                                                                valid_dataset=complex_dataset, 
                                                                batch_size=batch_si, 
                                                                lr=learning_rate, 
                                                                loss_fn=model_loss, 
                                                                schedule=schedule, 
                                                                num_timesteps=num_timesteps, 
                                                                sampler=sampler_ddpm)
    
    diffusion_model.to(device)
    #2, 16 2代表的是两根天线 16个complex值（input dim）
    input_vec, _, _ = complex_dataset[0]
    data_dimension, length = input_vec.shape

    # for saving generated and real collected data
    x_generated_list = []
    x_real_list = []
    loc_vec_list = []
    loc_int_list = []
    # generate data for each location x全都代表csi
    for loc_int in range(num_locs):
        print("\nGenerating CSI data for Location ID: {}\n".format(loc_int))
        real_data, loc_tensor, loc_int_tensor = get_features_by_label_v4(complex_dataset, loc_int)
        batch_input = loc_tensor.to(device)
        # 500
        number_samples_generated = real_data.shape[0]
        data_shape = [number_samples_generated, data_dimension, length]

        diffusion_model.eval()
        with torch.no_grad():   #batch_input: 位置向量
            generated_data = diffusion_model(data_shape, batch_input, sampler=sampler_ddpm, verbose=True)

        x_generated_list.append(generated_data.cpu())
        x_real_list.append(real_data.cpu())
        loc_vec_list.append(loc_tensor.cpu())
        loc_int_list.append(loc_int_tensor.cpu())
        
    tensor_generated_x = torch.cat(x_generated_list, dim=0)
    tensor_real_x = torch.cat(x_real_list, dim=0)
    tensor_loc_vec = torch.cat(loc_vec_list, dim=0)
    tensor_loc_int = torch.cat(loc_int_list, dim=0)

    complex_dataset_generated = ComplexDataset_real_imagary_v2(tensor_generated_x, 
                                                                tensor_real_x, 
                                                                tensor_loc_vec, 
                                                                tensor_loc_int)
    
    data_path_area_fake = data_path = os.path.join(output_dir, args.data_name_fake)

    torch.save(complex_dataset_generated, data_path_area_fake)



