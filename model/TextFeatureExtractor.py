import torch
from transformers import BertModel, BertTokenizer
import torch.nn as nn


class TextFeatureExtractor(nn.Module):
    # def __init__(self, model_name="./model/bert"):
    def __init__(self, model_name="bert-base-uncased"):
        super(TextFeatureExtractor, self).__init__()

        # 检查是否有GPU可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 定义分词器和模型
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)

    # text: 文本
    def forward(self, text):
        # 分词
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)

        # 获取词嵌入
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 提取嵌入
        last_hidden_state = outputs.last_hidden_state

        return last_hidden_state


if __name__ == "__main__":
    tte = TextFeatureExtractor()
    text = ["Hello", "Hi", "I'm fine"]
    print(tte(text).shape)
