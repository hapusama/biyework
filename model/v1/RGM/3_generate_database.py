import os
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from src.parameter_paser import parse_args_finetune,parse_args_pretrain
from src.dataset import ComplexDatasetLocs, ComplexDataset_real_imagary_v2,RealorFakeDataset
from src.denoising_diffusion_process.samplers.DDPM import DDPM_Sampler
from src. pixel_diffusion import PixelDiffusionConditional_v2
from src.utils import get_features_by_label_v4
from src.utils import get_condition_by_label

if __name__ == '__main__':

    base_dir = os.path.dirname(os.path.realpath(__file__))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_args_finetune()
    # args= parse_args_pretrain()
    print(f"\nUsing configuration file: {args.config}\n")

    input_dir = r"model\v1\input"
    output_dir = r"model\v1\output"

    num_locs = args.num_locs

    # using data from area b
    # todo 不同的单个sf下去跑跑看，试着把生成的数据和真实数据生成摘出来分开
    data_path_area_2 = data_path = os.path.join(input_dir, args.data_name_ft)
    loaded = torch.load(data_path_area_2)
    rssi = loaded['rssi']
    snr = loaded['snr']
    label = loaded['label']
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
    signal_feature_dim=args.signal_feature_dim
    loaded_fine_tuned_rgm = os.path.join(output_dir, args.rgm_fine_tune_path)
    # loaded_fine_tuned_rgm="model\\v1\\output\\2_finetuned_rgm.ckpt"
    loaded_fine_tuned_rgm=r"model\v1\output\lossmin\pretrain_only_sf_9.ckpt"
    sampler_ddpm = DDPM_Sampler(num_timesteps=num_timesteps, schedule=schedule)

    print("\nThe loaded diffusion model: {}\n".format(loaded_fine_tuned_rgm))
    # 这里先不用 loaded_fine_tuned_rgm
    diffusion_model = PixelDiffusionConditional_v2.load_from_checkpoint(checkpoint_path=loaded_fine_tuned_rgm, 
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
                                                                sampler=sampler_ddpm,signal_feature_dim=signal_feature_dim)
    
    diffusion_model.to(device)
    input_vec, _, _,_,_,_ = complex_dataset[0]
    #data_dimension, length = input_vec.shape
    data_dimension=data_channels
    length=input_dim
    # for saving generated and real collected data
    x_generated_list = []
    x_real_list = []
    x_real_loc_list=[]
    loc_vec_list = []
    loc_int_list = []
    # Get unique location IDs from the dataset
    unique_loc_ids = torch.unique(complex_dataset.labels).tolist()

    number_samples_generated=800 
    sf=9
    tp=2
    # 将sf扩维到[number_samples_generated,]的张量
    sf_list = torch.tensor(sf).unsqueeze(0).repeat(number_samples_generated)
    tp_list=torch.tensor(tp).unsqueeze(0).repeat(number_samples_generated)
    sf_list = (sf_list/12).unsqueeze(1)  # [number_samples_generated, 1]
    tp_list = (tp_list/12).unsqueeze(1)  # [number_samples_generated, 1]
    # 指定一个sf
    for loc_int in range(num_locs):
        print("\nGenerating CSI data for Location ID: {}\n".format(loc_int))
        # real_data[0]: [4,]
        if loc_int in unique_loc_ids:
            real_data, _, _,_,_,_ = get_features_by_label_v4(complex_dataset, loc_int)
        
        x,y,distance,true_distance = get_condition_by_label(loc_int,location_vector_path)
        
        loc_int_tensor = torch.tensor(loc_int).unsqueeze(0).repeat(number_samples_generated)
        x_tensor = torch.tensor(x).unsqueeze(0).repeat(number_samples_generated)
        y_tensor = torch.tensor(y).unsqueeze(0).repeat(number_samples_generated)
        distance_tensor = torch.tensor(distance).unsqueeze(0).repeat(number_samples_generated)
        
        true_distance=torch.tensor(true_distance.repeat(number_samples_generated))
        
        # cat
        x_tensor = x_tensor.unsqueeze(1)  # [number_samples_generated, 1]
        y_tensor = y_tensor.unsqueeze(1)  # [number_samples_generated, 1]
        distance_tensor = distance_tensor.unsqueeze(1)  # [number_samples_generated, 1]

        condition = torch.cat((x_tensor, y_tensor, distance_tensor,sf_list,tp_list), dim=1)   # [number_samples_generated, 5]
        #将condition转换为浮点型
        condition = condition.float()
        condition=condition.to(device)
        loc_int_tensor=loc_int_tensor.to(device)
        real_data = real_data.to(device)
        true_distance=true_distance.to(device)
        data_shape = [number_samples_generated, 1, length]  # todo：把channel写入yml   
        
        if not isinstance(sf, torch.Tensor):
            sf = torch.tensor(sf, dtype=torch.int, device=device)
        else:
            sf = sf.clone().detach().to(dtype=torch.int, device=device)

        if not isinstance(tp, torch.Tensor):
            tp = torch.tensor(tp, dtype=torch.int, device=device)
        else:
            tp = tp.clone().detach().to(dtype=torch.int, device=device)

        input_sf=torch.stack([sf]*number_samples_generated, dim=0)  #[@,]
        input_tp=torch.stack([tp]*number_samples_generated, dim=0)  #[@,]
        diffusion_model.eval()
        with torch.no_grad():
            generated_data = diffusion_model(data_shape, condition,input_sf,input_tp,true_distance,sampler=sampler_ddpm, verbose=True) 
        selected_feature=condition[:,-2:]  #[@,2]  
        loc_tensor_last2=condition[:,-2:]
        # todo: 保存数据sf tp是否保存归一化之前的
        generated_data=generated_data.view(-1,length)
        generated_data=torch.cat((generated_data, selected_feature), dim=1) #dim=2表示在最后一维cat
        print("generated_data shape: ", generated_data.shape)
        print("generated_data : ", generated_data)
        print("real_data.shape without sf tp : ", real_data.shape)
        print("real_data without sf tp:", real_data)
        loc_vec_list.append(condition.cpu())    #[@,5]
        loc_int_list.append(loc_int_tensor.cpu())   #[@,]
        x_generated_list.append(generated_data.cpu())
    
    tensor_generated_x = torch.cat(x_generated_list, dim=0)
    tensor_loc_vec = torch.cat(loc_vec_list, dim=0)
    tensor_loc_int = torch.cat(loc_int_list, dim=0)
    
    
    # 分别保存两个数据集
    fake_Dataset = RealorFakeDataset(tensor_generated_x,tensor_loc_int)
    data_path_area_real = os.path.join(output_dir, args.data_name_real)
    data_path_area_fake = os.path.join(output_dir, args.data_name_fake)
    
    torch.save(fake_Dataset, data_path_area_fake)



