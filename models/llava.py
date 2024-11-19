datasets_path = 'eepy/datasets'
hf_cache_path = 'eepy/hf-cache'


from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets.rocov2 import ROCOv2Dataset


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model and processor
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=hf_cache_path)
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", cache_dir=hf_cache_path)

# Define paths
train_image_dir = 'sample_datasets/ROCOv2/train'
train_caption_file = 'sample_datasets/ROCOv2/sample_train_captions.csv'
valid_image_dir = 'sample_datasets/ROCOv2/valid'
valid_caption_file = 'sample_datasets/ROCOv2/sample_valid_captions.csv'
test_image_dir = 'sample_datasets/ROCOv2/test'
test_caption_file = 'sample_datasets/ROCOv2/sample_test_captions.csv'


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset instances
train_dataset = ROCOv2Dataset(train_image_dir, train_caption_file, processor, image_transform)
valid_dataset = ROCOv2Dataset(valid_image_dir, valid_caption_file, processor, image_transform)
test_dataset = ROCOv2Dataset(test_image_dir, test_caption_file, processor, image_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)


output_dir = 'eepy/llava-lora/output'
logging_dir = 'eepy/llava-lora/logging'
ckpt_dir = 'eepy/llava-lora/ckpt'