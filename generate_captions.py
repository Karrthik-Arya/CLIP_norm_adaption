from transformers import OFATokenizer, OFAModel
import os
import pandas as pd
import json
from PIL import Image
import torch
from torchvision import transforms

device = "cuda:1" if torch.cuda.is_available() else "cpu"
ckpt_dir = "../OFA-large-caption"
ofa_tokenizer = OFATokenizer.from_pretrained(ckpt_dir)
ofa_model = OFAModel.from_pretrained(ckpt_dir, use_cache=False)
ofa_model.to(device)
ofa_model.eval()

mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 256
patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(), 
    transforms.Normalize(mean=mean, std=std)
])

def generate_caption(image):
    """Generate caption for an image using the OFA model."""
    with torch.no_grad():
        inputs = ofa_tokenizer("what does the image describe?", return_tensors="pt").to(device)
        images = patch_resize_transform(image).unsqueeze(0).to(device)
        outputs = ofa_model.generate(**inputs, patch_images=images)
        caption = ofa_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return caption

def generate_train_captions(root, subset):
    IMAGE_PATH = {
        "train": { 
             "questions": "v2_OpenEnded_mscoco_train2014_questions.json",
             "answers":  "v2_mscoco_train2014_annotations.json",
             "img_folder": "train2014"
             },
        "val": {
            "questions": "v2_OpenEnded_mscoco_val2014_questions.json", 
            "answers": "v2_mscoco_val2014_annotations.json",
            "img_folder": "val2014"
        }
    }
    q_path = os.path.expanduser(os.path.join(root, IMAGE_PATH[subset]["questions"]))

    with open(q_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data["questions"])
    image_paths = df["image_id"].apply(
        lambda x: f"{IMAGE_PATH[subset]["img_folder"]}/COCO_{IMAGE_PATH[subset]["img_folder"]}_{x:012d}.jpg")
    
    captions = []
    for image_path in image_paths:
        img = Image.open(image_path).convert('RGB')
        caption = generate_caption(img)
        captions.append(caption)
    
    df["caption"] = captions

    df.to_csv(data["questions"])

generate_train_captions('data/vqa_v2','train')
generate_train_captions('data/vqa_v2','val')
