import Retrieval as r
from Dataset import TestDataset
import torch

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

image_paths = image_paths[39900:40000]
text_descs = text_descs[39900:40000]


def calculate_i2t(size):
    total_hit = 0
    # 遍历 image_paths，步长为 5
    for i in range(0, len(image_paths), 5):
        batch_image_paths = image_paths[i]  # 每次取出5个图像路径
        batch_text_descs = text_descs[i:i + 5]  # 每次取出5个文本描述

        top_k_indices, top_k_similarities = r.get_top_k_similar_text_fast(batch_image_paths, r.text_tensor, top_k=size+2)

        final_top_k_results = r.get_top_k_similar_text_slow(top_k_indices, r.text_descs, batch_image_paths, top_k_similarities, top_k=size)

        intersection = set(batch_text_descs).intersection(set(final_top_k_results))

        # 输出交集元素的个数
        if intersection:
            total_hit += 1

        print(i)

    return total_hit / (len(image_paths) / 5)

def calculate_t2i(size):
    total_hit = 0

    for i in range(0, len(image_paths)):
        image_path = image_paths[i]
        text_desc = text_descs[i]

        top_k_indices, top_k_similarities = r.get_top_k_similar_images_fast(text_desc, r.image_tensor, top_k=size)

        final_top_k_results = r.get_top_k_similar_image_slow(top_k_indices, r.image_paths, text_desc, top_k_similarities, top_k=size)

        # 输出交集元素的个数
        if image_path in final_top_k_results:
            total_hit += 1

        print(i)

    return total_hit / len(image_paths)

if __name__ == "__main__":
    print("R@K5:", calculate_t2i(5))
    print("R@K10:", calculate_t2i(10))