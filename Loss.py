import torch
import torch.nn as nn
from ImageTextCrossAttention import ImageTextCrossAttention

# 设备检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义参数
alpha = 0.2

class Loss(nn.Module):
    def __init__(self, margin=0.2):
        super(Loss, self).__init__()
        self.margin = margin
        self.image_text_cross_attention = ImageTextCrossAttention()

    def forward(self, captions, images):
        batch_size = len(captions)
        similarity_matrix = torch.zeros((batch_size, batch_size), device=device)

        # 计算相似度矩阵
        for i in range(batch_size):
            # 固定 caption 计算
            fixed_caption = [captions[i]] * batch_size
            similarity_vector = self.image_text_cross_attention(fixed_caption, images)
            similarity_matrix[i, :] = similarity_vector

        # 计算损失
        loss = self.compute_loss(similarity_matrix)

        return loss.requires_grad_()

    def compute_loss(self, similarity_matrix):
        batch_size = similarity_matrix.size(0)

        # 对角线元素
        diag_elements = similarity_matrix.diag()

        # 找到每行、每列的最大值，排除对角线元素
        row_max, _ = similarity_matrix.masked_fill(torch.eye(batch_size, device=device).bool(),
                                                   float('-inf')).max(dim=1)
        col_max, _ = similarity_matrix.masked_fill(torch.eye(batch_size, device=device).bool(),
                                                   float('-inf')).max(dim=0)

        # 计算 S1 和 S2
        S1 = torch.clamp(self.margin + row_max - diag_elements, min=0)
        S2 = torch.clamp(self.margin + col_max - diag_elements, min=0)

        # 累加所有对角线元素的计算结果
        loss = torch.sum(S1 + S2)

        return loss