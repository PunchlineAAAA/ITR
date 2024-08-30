
import torch
from Loss import Loss
from Dataset import CustomDataset

# 设备检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型并载入训练好的参数
model = Loss()
model_load_path = './saved_models/model_epoch_10.pth'
model.load_state_dict(torch.load(model_load_path))
model = model.image_text_cross_attention.to(device)
model.eval()

# 数据加载
dataset = CustomDataset("./dataset/shorter_captions.txt")
all_images = [image for image, _ in dataset]
all_captions = [caption for _, caption in dataset]
data_len = len(all_images)


def retrieve_top_k_texts(image_path, top_k=3, batch_size=8):
    image_paths = [image_path] * data_len
    similarities = torch.empty(data_len)

    # 计算相似度
    with torch.no_grad():
        for i in range(0, data_len, batch_size):
            batch_captions = all_captions[i:i+batch_size]
            batch_image_paths = image_paths[i:i + batch_size]
            batch_similarities = model(batch_captions, batch_image_paths)
            similarities[i:i + batch_size] = batch_similarities


    # 获取相似度最高的 top_k 个文本
    top_k_indices = torch.topk(similarities, top_k).indices
    top_k_captions = [all_captions[i] for i in top_k_indices]

    return top_k_captions


def retrieve_top_k_images(text_input, top_k=3):
    # 计算相似度
    with torch.no_grad():
        similarities = [model(text_input, image) for image in all_images]

    # 获取相似度最高的 top_k 个文本
    top_k_indices = torch.topk(torch.tensor(similarities), top_k).indices
    top_k_images = [all_images[i] for i in top_k_indices]

    return top_k_images


if __name__ == "__main__":
    # 示例：检索与输入图片最相似的文本
    image_input = "./dataset/images/1009434119_febe49276a.jpg"
    top_texts = retrieve_top_k_texts(image_input, top_k=3)
    print("Top 3 similar texts:", top_texts)