import os
from classifier_finger import *
from src.parameter_paser import parse_args_finetune

if __name__=="__main__":
    args = parse_args_finetune()
    save_model_name_fake = args.save_model_name_fake
    input_dir = r"model\v1\input"
    output_dir = r"model\v1\output"
    
    input_dim = args.input_dim
    num_classes = args.num_classes
    model_check_point_pth = os.path.join(output_dir, save_model_name_fake)
    print(f"model_check_point_pth: {model_check_point_pth}")
    model = 
    
