import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# 加载预训练的 Faster R-CNN 模型
faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to('cuda' if torch.cuda.is_available() else 'cpu')

for param in faster_rcnn.parameters():
    param.requires_grad = False

faster_rcnn.eval()


class ImageFeatureExtractor(nn.Module):
    def __init__(self, d=768, k=36):
        super(ImageFeatureExtractor, self).__init__()

        # 检查是否有可用的 GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # 加载预训练的 ResNet101 模型，并修改最后的全连接层
        self.resnet101 = torchvision.models.resnet101(pretrained=True)
        self.resnet101.fc = nn.Linear(self.resnet101.fc.in_features, d)
        self.resnet101 = self.resnet101.to(self.device)
        self.resnet101.eval()

        # 参数设置
        self.k = k  # 前 k 个显著区域
        self.d = d  # 特征维度

    def forward(self, image_paths):
        batch_images = []

        for path in image_paths:
            # 导入图像
            image = Image.open(path).convert("RGB")
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            batch_images.append(image)

        images = torch.cat(batch_images, dim=0)

        # 检测物体
        with torch.no_grad():
            detections = faster_rcnn(images)

            for detection in detections:
                # 获取边界框和置信度分数
                boxes = detection['boxes'].to(self.device)
                scores = detection['scores'].to(self.device)

                # 选择前 k 个显著区域
                k = min(self.k, len(boxes))
                top_k_indices = scores.topk(k).indices
                top_k_boxes = boxes[top_k_indices]

        # 卷积特征
        conv_features = []

        for box in top_k_boxes:
            # 提取坐标
            x_min, y_min, x_max, y_max = map(int, box)

            # 裁剪图片
            image_parts = images[:, :, y_min:y_max, x_min:x_max]

            # 提取卷积特征
            with torch.no_grad():
                pooled_feature = self.resnet101(image_parts)

            conv_features.append(pooled_feature.cpu())

        return torch.stack(conv_features).transpose(0, 1)


if __name__ == "__main__":
    model = ImageFeatureExtractor()
    image_paths = ["./dataset/images/667626_18933d713e.jpg", "./dataset/images/3637013_c675de7705.jpg"]
    features = model(image_paths)
    print(features.shape)