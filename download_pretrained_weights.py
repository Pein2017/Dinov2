import torch
import os
import torchvision.transforms as transforms
from PIL import Image
import shutil

# mkdir -p ./pretrained
os.makedirs('./pretrained', exist_ok=True)

torch.hub.set_dir('./pretrained')

dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitb14.eval()

# Set device to CPU and move model to device
device = torch.device('cpu')
dinov2_vitb14.to(device)

test_img_path = '/data/training_code/Pein/dinov2/mixed_data_root/bbu_grounding_wire_cleaned/test/0000212/0000212-1.jpeg'

# Add inference code
image = Image.open(test_img_path)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Prepare input tensor on CPU
input_tensor = transform(image).unsqueeze(0).to(device)
output = dinov2_vitb14(input_tensor)
print(output)

