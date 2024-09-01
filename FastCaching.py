import torch
import open_clip
from PIL import Image

# 加载OpenCLIP模型，设置设备为GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b-s34b_b79k')
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# 定义数据文件路径
captions_file = './dataset/captions.txt'
output_file = './dataset/output.pt'

# 定义一个转换图像的函数
def transform_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).to(device)

# 读取captions.txt文件并处理
image_tensors = []
text_tensors = []
processed_images = {}

with open(captions_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()[1:]  # 跳过标题行
    for line in lines:
        # 使用第一个逗号分隔图像路径和文本
        image_path, text = line.strip().split(',', 1)
        image_path = "./dataset/images/" + image_path.strip()

        # 处理图像：如果之前已经处理过相同图像，则复用
        if image_path not in processed_images:
            image_tensor = transform_image(image_path).unsqueeze(0)
            image_tensor = model.encode_image(image_tensor)
            processed_images[image_path] = True
            image_tensors.append(image_tensor)

        # 转换文本为张量并转移到GPU
        text_tensor = tokenizer([text]).to(device)
        text_tensor = model.encode_text(text_tensor)
        text_tensors.append(text_tensor)

        print(len(image_tensors))

# 将图像张量和文本张量保存为.pt文件
torch.save({'images': image_tensor, 'texts': text_tensor}, output_file)

print(f"数据已保存到 {output_file}")