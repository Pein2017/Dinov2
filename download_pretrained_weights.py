import logging
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# mkdir -p ./pretrained
os.makedirs("./pretrained", exist_ok=True)

torch.hub.set_dir("./pretrained")

dinov2_vitb14 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14",
    trust_repo=True,
    source="github",
)
dinov2_vitb14.eval()

# Verify CUDA availability and set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    dinov2_vitb14.to(device)
else:
    raise EnvironmentError("CUDA is not available. Please check your GPU setup.")

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="./tensorboard_logs")

test_img_path = "/data/training_code/Pein/dinov2/mixed_data_root/bbu_grounding_wire_cleaned/test/0000212/0000212-1.jpeg"

# Add inference code
image = Image.open(test_img_path)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# Prepare input tensor on GPU
input_tensor = transform(image).unsqueeze(0).to(device)
output = dinov2_vitb14(input_tensor)


# Configure logging
logging.basicConfig(
    filename="./test/pretrained_model_arch.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)

# Log the model's state_dict keys and their corresponding tensor shapes
state_dict_info = "\n".join(
    [f"{key}: {value.shape}" for key, value in dinov2_vitb14.state_dict().items()]
)

logging.info(f"Model state_dict:\n{state_dict_info}")

logging.info(output.shape)

# Replace print statement with logging
logging.info(dinov2_vitb14)
