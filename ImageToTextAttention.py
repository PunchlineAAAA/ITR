import torch
import torch.nn.functional as F

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数
alpha = 20


def preassign_attention(image_regions, text_words):
    # 获取 batch_size, num_boxes, emb_dim 和 seq_len
    batch_size, num_boxes, emb_dim = image_regions.shape
    _, seq_len, _ = text_words.shape

    # 初始化
    all_attention_scores = []

    for b in range(batch_size):
        # 获取当前批次的图像区域和文本词语
        image_regions_b = image_regions[b]
        text_words_b = text_words[b]

        similarities = []

        # 计算余弦相似度
        for region in image_regions_b:
            similarities_row = []

            for word in text_words_b:
                cosine_sim = alpha * F.cosine_similarity(region.to(device), word.to(device), dim=0)
                similarities_row.append(cosine_sim)

            similarities.append(similarities_row)

        # 转化为 Tensor 张量
        similarities = torch.tensor(similarities)

        # 每行 softmax 归一化 (每个词语对应的区域相似度)
        attention_scores_b = F.softmax(similarities, dim=1)

        all_attention_scores.append(attention_scores_b)

    # 堆叠所有批次的 attention_scores
    attention_scores = torch.stack(all_attention_scores)

    return attention_scores


def calculate_score(attention_scores):
    # 获取维度信息
    batch_size, image_size, text_size = attention_scores.shape

    # 初始化得分
    F_scores = torch.zeros_like(attention_scores)

    # 计算文本对区域的得分
    for b in range(batch_size):
        for j in range(image_size):
            for i in range(text_size):
                score = 0.0

                for t in range(text_size):
                    f_value = attention_scores[b, j, i] - attention_scores[b, j, t]
                    g_value = torch.sqrt(attention_scores[b, j, t])
                    score += f_value * g_value

                F_scores[b, j, i] = score

    # 赋值 irrelevant 为 1e-8
    F_scores = torch.clamp(F_scores, min=1e-8)

    # 归一化
    row_sums = F_scores.sum(dim=2, keepdim=True)
    F_scores = F_scores / row_sums

    return F_scores


def text_share_semantic(text_words, F_scores):
    # 获取维度信息
    batch_size, image_size, text_size = F_scores.shape
    _, _, emb_dim = text_words.shape
    text_share_semantics = torch.zeros(batch_size, image_size, emb_dim)

    text_words = text_words.to(device)

    # 计算共享语义
    for b in range(batch_size):
        for j in range(image_size):
            total = torch.zeros(emb_dim, device=device)

            for i in range(text_size):
                total += F_scores[b, j, i] * text_words[b, i]

            text_share_semantics[b, j] = total

        return text_share_semantics


def relevance(text_share_semantics, image_regions):
    batch_size, box_num, emb_dim = image_regions.shape
    sim = torch.zeros(batch_size, device=device)

    # 遍历
    for b in range(batch_size):
        batch_sim = 0.0

        for idx in range(box_num):
            word = text_share_semantics[b, idx]
            image = image_regions[b, idx]
            batch_sim += F.cosine_similarity(word.to(device), image.to(device), dim=0)

        sim[b] = batch_sim / box_num

    return sim