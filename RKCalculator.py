import Retrieval as r
from Dataset import TestDataset
import torch

# 定义数据文件路径
captions_file = './dataset/captions.txt'
image_paths = []
text_descs = []

# 使用 torch.no_grad() 以节省内存
with torch.no_grad():
    # 根据 captions.txt 处理文件
    with open(captions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]
        for line in lines:
            image_path, text = line.strip().split(',', 1)
            image_path = "./dataset/images/" + image_path.strip()
            image_paths.append(image_path)
            text_descs.append(text)

# 手动分割测试集
image_paths = image_paths[3950:40000]
text_descs = text_descs[39500:40000]


# 计算图搜文 R@K
def calculate_i2t(size):
    total_hit = 0
    # 遍历 image_paths，步长为 5
    for i in range(0, len(image_paths), 5):
        batch_image_paths = image_paths[i]  # 每次取出5个图像路径
        batch_text_descs = text_descs[i:i + 5]  # 每次取出5个文本描述

        # 搜文逻辑
        top_k_indices, top_k_similarities = r.get_top_k_similar_text_fast(batch_image_paths, r.text_tensor, top_k=size+2)

        final_top_k_results = r.get_top_k_similar_text_slow(top_k_indices, r.text_descs, batch_image_paths, top_k_similarities, top_k=size)

        intersection = set(batch_text_descs).intersection(set(final_top_k_results))

        # 判断交集
        if intersection:
            total_hit += 1

    return total_hit / (len(image_paths) / 5)


# 计算文搜图 R@K
def calculate_t2i(size):
    total_hit = 0

    for i in range(0, len(image_paths)):
        # 取出图片以及文本
        image_path = image_paths[i]
        text_desc = text_descs[i]

        # 搜图逻辑
        top_k_indices, top_k_similarities = r.get_top_k_similar_images_fast(text_desc, r.image_tensor, top_k=size+2)

        final_top_k_results = r.get_top_k_similar_image_slow(top_k_indices, r.image_paths, text_desc, top_k_similarities, top_k=size)

        # 判断交集
        if image_path in final_top_k_results:
            total_hit += 1

    return total_hit / len(image_paths)


if __name__ == "__main__":
    print("R@K5:", calculate_t2i(5))
    print("R@K10:", calculate_t2i(10))