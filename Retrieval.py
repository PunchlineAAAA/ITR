import torch
import open_clip
import torch.nn.functional as F
from PIL import Image
from Loss import Loss

# 检查GPU是否可用，并将设备设置为GPU，否则使用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的 openclip 模型
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.to(device)
model.eval()

# 读取 output.pt 文件
data = torch.load("./dataset/output.pt")

# 从 data 中提取出 image_tensor 和 text_tensor
image_list = data["images"]  # 假设这是一个 list，其中每个元素是 shape 为 [1, 512] 的张量
text_list = data["texts"]    # 假设这是一个 list，其中每个元素是 shape 为 [1, 512] 的张量
image_list_filtered = [image_list[i] for i in range(0, len(image_list), 5)]

# 将 image_list 和 text_list 转换为张量
image_tensor = torch.stack(image_list_filtered).squeeze(1)  # 将形状从 [N, 1, 512] 转换为 [N, 512]
text_tensor = torch.stack(text_list).squeeze(1)    # 将形状从 [N, 1, 512] 转换为 [N, 512]

# 确保张量在 GPU 上
image_tensor = image_tensor.to(device)
text_tensor = text_tensor.to(device)

# 定义数据文件路径
captions_file = './dataset/captions.txt'
image_paths = []
text_descs = []

# 使用 torch.no_grad() 以节省内存
with torch.no_grad():
    with open(captions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # 跳过标题行
        for line in lines:
            # 使用第一个逗号分隔图像路径和文本
            image_path, text = line.strip().split(',', 1)
            image_path = "./dataset/images/" + image_path.strip()
            image_paths.append(image_path)
            text_descs.append(text)

# 加载模型
loss = Loss()
loss.load_state_dict(torch.load("saved_models/model_epoch_10.pth"))
model_a = loss.image_text_cross_attention.to(device)
model_a.eval()


# 与图片相似度计算函数（快）
def get_top_k_similar_images_fast(text_input, image_tensor, top_k=5):
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


# 与文字相似度计算函数（快）
def get_top_k_similar_text_fast(image_input, text_tensor, top_k=5):
    # 打开图片
    image = Image.open(image_input).convert("RGB")

    # 嵌入
    image = preprocess(image).unsqueeze(0).to(device)
    image_tensor = model.encode_image(image)

    # 对图像特征和文本特征进行 L2 正则化
    text_features = F.normalize(text_tensor, dim=-1)
    image_tensor = F.normalize(image_tensor, dim=-1)

    # 计算相似度
    similarities = torch.matmul(text_features, image_tensor.T).squeeze(1)

    # 获取相似度最高的 top_k 个图像索引
    top_k_indices = similarities.topk(top_k).indices

    return top_k_indices, similarities[top_k_indices]


# 与图像相似度计算函数（快）
def get_top_k_similar_image_slow(top_k_indices, image_paths, user_input, previous_confidence, top_k=3):
    top_k_image_paths = [image_paths[i * 5] for i in top_k_indices]
    user_input = [user_input] * len(top_k_indices)

    # 计算
    res = model_a(user_input, top_k_image_paths) + previous_confidence

    # 找到相似度的最大的 top_k 个元素的下标
    _, top_k_res_indices = torch.topk(res, top_k)

    # 使用这些下标查询 top_k_indices 以获得最终的结果
    final_top_k_indices = [top_k_indices[i] for i in top_k_res_indices]

    # 假设这里只使用flickr8k
    final_top_k_results = [image_paths[i * 5] for i in final_top_k_indices]

    return final_top_k_results


# 与文字相似度计算函数（慢）
def get_top_k_similar_text_slow(top_k_indices, text_descs, user_input, previous_confidence, top_k=3):
    top_k_text_descs = [text_descs[i] for i in top_k_indices]
    user_input = [user_input] * len(top_k_indices)

    # 计算
    res = model_a(top_k_text_descs, user_input) + previous_confidence

    # 找到相似度最大的 top_k 个元素的下标
    _, top_k_res_indices = torch.topk(res, top_k)

    # 使用这些下标查询 top_k_indices 以获得最终的结果
    final_top_k_indices = [top_k_indices[i] for i in top_k_res_indices]

    # 假设这里只使用flickr8k
    final_top_k_results = [text_descs[i] for i in final_top_k_indices]

    return final_top_k_results


if __name__ == "__main__":
    # user_input = "./dataset/images/1000268201_693b08cb0e.jpg"
    #
    # # 调用函数获取与输入文本最相似的 top_k 张图像
    # top_k_indices, top_k_similarities = get_top_k_similar_text_fast(user_input, text_tensor, top_k=5)
    #
    # print("Top K Similar Images Indices:", top_k_indices)
    # print("Top K Similarities:", top_k_similarities)
    #
    # final_top_k_indices = get_top_k_similar_text_slow(top_k_indices, text_descs, user_input, top_k_similarities)
    #
    # print(final_top_k_indices)

    user_input = "A girl going into a wooden building"

    top_k_indices, top_k_similarities = get_top_k_similar_images_fast(user_input, image_tensor)

    final_top_k_results = get_top_k_similar_image_slow(top_k_indices, image_paths, user_input, top_k_similarities)

    print(final_top_k_results)