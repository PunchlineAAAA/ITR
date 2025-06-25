import torch
import open_clip
from PIL import Image

# 加载OpenCLIP模型，设置设备为GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b-s34b_b79k')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
model.load_state_dict(torch.load('./local_model/clip/open_clip_pytorch_model.bin'))
model = model.to(device)
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# 定义数据文件路径
captions_file = './dataset/captions.txt'
output_file = './dataset/output.pt'


# 读取图像函数
# image_path: 图片路径
def transform_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).to(device)


# 读取captions.txt文件并处理
image_tensors = []
text_tensors = []

# 使用 torch.no_grad() 以节省内存
with torch.no_grad():
    # 按 captions.txt 格式读取
    with open(captions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            # 使用第一个逗号分隔图像路径和文本
            image_path, text = line.strip().split(',', 1)
            image_path = "./dataset/images/" + image_path.strip()

            # 处理图像
            image_tensor = transform_image(image_path).unsqueeze(0)
            image_tensor = model.encode_image(image_tensor)

            # 转换文本为张量
            text_tensor = tokenizer([text]).to(device)
            text_tensor = model.encode_text(text_tensor)

            # 保存每一对图像和文本张量
            image_tensors.append(image_tensor.cpu())
            text_tensors.append(text_tensor.cpu())

            # 释放未使用的内存
            del image_tensor, text_tensor
            torch.cuda.empty_cache()

            print(f"处理了 {len(text_tensors)} 条数据")

# 将图像张量和文本张量保存为.pt文件
torch.save({'images': image_tensors, 'texts': text_tensors}, output_file)

print(f"数据已保存到 {output_file}")
