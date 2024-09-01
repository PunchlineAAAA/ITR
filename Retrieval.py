import torch
import open_clip
import torch.nn.functional as F

# 检查GPU是否可用，并将设备设置为GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的 openclip 模型
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.to(device)
model.eval()

# 读取 output.pt 文件
data = torch.load("./dataset/output.pt")

# 从 data 中提取出 image_tensor 和 text_tensor
image_tensor = data["images"].to(device)
text_tensor = data["texts"].to(device)

print(image_tensor.shape)


# 相似度计算函数
def get_top_k_similar_images(text_input, image_tensor, top_k=5):
    # 将文本输入转换为张量
    with torch.no_grad():
        text_features = model.encode_text(open_clip.tokenize([text_input]).to(device))

    # 对图像特征和文本特征进行 L2 正则化
    text_features = F.normalize(text_features, dim=-1)
    image_tensor = F.normalize(image_tensor, dim=-1)

    # 计算相似度
    similarities = torch.matmul(image_tensor, text_features.T).squeeze(1)

    # 获取相似度最高的 top_k 个图像索引
    top_k_indices = similarities.topk(top_k).indices

    return top_k_indices, similarities[top_k_indices]


if __name__ == "__main__":
    user_input = "cat on a sunny day"

    # 调用函数获取与输入文本最相似的 top_k 张图像
    top_k_indices, top_k_similarities = get_top_k_similar_images(user_input, image_tensor, top_k=5)

    print("Top K Similar Images Indices:", top_k_indices)
    print("Top K Similarities:", top_k_similarities)