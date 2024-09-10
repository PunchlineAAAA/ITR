import torch
import torch.nn.functional as F

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 放大系数
alpha = 20


# 预分配注意力
# image_regions: 通过 Faster-RNN 分出的图形区域特征张量
# text_words: 通过 BERT 分词得到的特征张量
# attention_scores: 注意力评分
def preassign_attention(image_regions, text_words):
    # 获取 batch_size, num_boxes, emb_dim 和 seq_len
    batch_size, num_boxes, emb_dim = image_regions.shape
    _, seq_len, _ = text_words.shape

    # 初始化
    all_attention_scores = []

    # 处理批次
    for b in range(batch_size):
        # 获取当前批次的图像区域和文本词语
        image_regions_b = image_regions[b]
        text_words_b = text_words[b]

        similarities = []

        # 计算余弦相似度
        for word in text_words_b:
            similarities_row = []

            for region in image_regions_b:
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


# 相关度评分
# attention_scores: 注意力评分
# F_scores: 相关性评分（对应论文中的F）
def calculate_score(attention_scores):
    # 获取维度信息
    batch_size, text_size, image_size = attention_scores.shape

    # 初始化得分
    F_scores = torch.zeros_like(attention_scores)

    # 计算文本对区域的得分
    for b in range(batch_size):
        for i in range(text_size):
            for j in range(image_size):
                score = 0.0

                for t in range(image_size):
                    f_value = attention_scores[b, i, j] - attention_scores[b, i, t]
                    g_value = torch.sqrt(attention_scores[b, i, t])
                    score += f_value * g_value

                F_scores[b, i, j] = score

    # 赋值 irrelevant 为 1e-8
    F_scores = torch.clamp(F_scores, min=1e-8)

    # 归一化
    row_sums = F_scores.sum(dim=2, keepdim=True)
    F_scores = F_scores / row_sums

    return F_scores


# 提取图像与文本共享特征
# F_scores: 相关度评分
# text_words: 文本分词特征
# image_share_semantics: 图像与文本共享的特征
def image_share_semantic(image_regions, F_scores):
    # 获取维度信息
    batch_size, text_size, image_size = F_scores.shape
    _, _, emb_dim = image_regions.shape
    image_share_semantics = torch.zeros(batch_size, text_size, emb_dim)

    image_regions = image_regions.to(device)

    # 计算共享语义
    for b in range(batch_size):
        for i in range(text_size):
            total = torch.zeros(emb_dim, device=device)

            for j in range(image_size):
                total += F_scores[b, i, j] * image_regions[b, j]

            image_share_semantics[b, i] = total

    return image_share_semantics


# 计算相关度
# image_share_semantics: 图像与文本共享的特征
# text_words: 文本特征
# sim: 相似度
def relevance(text_words, image_share_semantics):
    batch_size, seq_len, emb_dim = text_words.shape
    sim = torch.zeros(batch_size, device=device)

    # 遍历求平均余弦相似度
    for b in range(batch_size):
        batch_sim = 0.0

        for idx in range(seq_len):
            word = text_words[b, idx]
            image = image_share_semantics[b, idx]
            batch_sim += F.cosine_similarity(word.to(device), image.to(device), dim=0)

        sim[b] = batch_sim / seq_len

    return sim
