import torch
from Loss import Loss
from Dataset import CustomDataset
import torch.nn.functional as F

# 设备检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型并载入训练好的参数
model = Loss()
model_load_path = './saved_models/model_epoch_1.pth'
model.load_state_dict(torch.load(model_load_path))

model = model.image_text_cross_attention
image_feature_extractor = model.image_feature_extractor
text_feature_extractor = model.text_feature_extractor

image_feature_extractor.eval()
text_feature_extractor.eval()
model.eval()

# 数据加载
dataset = CustomDataset("./dataset/captions.txt")
all_images = [image for image, _ in dataset]
all_captions = [caption for _, caption in dataset]
data_len = len(all_images)


# 缓存构造
def cache_building(batch_size=128):
    img_features = []
    cap_features = []

    for i in range(0, data_len, batch_size):
        images = all_images[i:i + batch_size]
        captions = all_captions[i:i + batch_size]

        # 提取图像特征
        img_batch_features = image_feature_extractor(images)
        img_features.append(img_batch_features)

        cap_batch_features = text_feature_extractor(captions)
        cap_features.append(cap_batch_features)

    # 堆叠
    img_fea = torch.cat(img_features, dim=0)
    cap_fea = torch.cat(cap_features)

    torch.save({'img_features': img_fea, "cap_features": cap_fea}, "./cached_file.pt")

    return cap_fea, img_fea


if __name__ == "__main__":
    cache_building()