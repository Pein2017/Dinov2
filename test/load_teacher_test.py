import torch
import logging
from collections import OrderedDict

def load_model_and_log_weights(ckpt_path, log_file='./test/load_teacher_test.log'):
    # Set up logging
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
    
    # Load the checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # Assuming the model is stored under the 'teacher' key; adjust if different
    model_state = checkpoint.get('teacher', checkpoint)
    
    def log_info(message):
        logging.info(message)
        print(message)
    
    # Log and print each parameter's name and attributes
    for name, param in model_state.items():
        if isinstance(param, torch.Tensor):
            info = f'Name: {name}\n\tShape: {param.shape}\n\tDtype: {param.dtype}'
        elif isinstance(param, OrderedDict):
            info = f'Name: {name}\n\tType: OrderedDict\n\tKeys: {list(param.keys())}'
        else:
            info = f'Name: {name}\n\tType: {type(param)}'
        log_info(info)
        log_info('-' * 80)  # Separator line

if __name__ == '__main__':
    ckpt_path = '/data/training_code/Pein/dinov2/bbu_logs/bbu_vits14-bs_256/eval/training_7849/teacher_checkpoint.pth'
    load_model_and_log_weights(ckpt_path)
    