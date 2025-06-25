import os.path
import numpy as np
from model.Dataset import CustomDataset, ValidationDataset
import torch
from torch.utils.data import DataLoader
from model.Loss import Loss
import torch.optim as optim


# 验证模型训练结果函数
# model: 模型
# dataloader: 数据加载器（外面定义）
# device: GPU 或者 CPU
# @return: 损失
def evaluate_model(model, dataloader, device):
    model.eval()  # 切换到评估模式
    running_loss = 0.0
    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, (images, captions) in enumerate(dataloader):
            # images, captions = images.to(device), captions.to(device)
            loss = model(captions, images)
            running_loss += loss.item()

    return running_loss / len(dataloader)


# 读取存档点
# checkpoint_path: 存档点文件路径
# model: 模型
# optimizer: 优化器

# epoch: 上次执行到的 epoch
# loss: 上次的 loss
def load_checkpoint(checkpoint_path, model, optimizer):
    # 判断是否存在
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")

        # 读取存档
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss
    else:
        # 异常处理
        print("No checkpoint found, starting from scratch")
        return 0, float('inf')


# 存档
# epoch: 当前 epoch
# model: 模型
# optimizer: 优化器
# loss: 损失
# checkpoint_path: 存档文件所在路径
def save_checkpoint(epoch, model, optimizer, loss, checkpoint_path):
    # 存档逻辑
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")


if __name__ == "__main__":
    # 设备检测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 参数设置
    batch_size = 32
    learning_rate = 0.0002
    num_epochs = 10
    model_save_path = "./saved_models/"
    checkpoint_path = os.path.join(model_save_path, "latest_checkpoint.pth")

    # 数据加载
    dataset = CustomDataset("./dataset/captions.txt")
    validation_dataset = ValidationDataset("./dataset/captions.txt")

    # 实例化模型
    model = Loss().to(device)

    optimizer = optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=learning_rate)

    # 加载检查点（如果存在）
    start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)

    # 训练开始
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # 随机打乱训练
        indices = np.random.choice(39000, 1000, replace=False)
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=8)

        # 开始训练
        model.train()
        running_loss = 0.0

        # 批处理
        for batch_idx, (images, captions) in enumerate(dataloader):
            optimizer.zero_grad()

            # 前向传播和计算损失
            loss = model(captions, images)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(f"Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # 验证阶段（每个epoch进行）
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        validation_loss = evaluate_model(model, validation_dataloader, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {validation_loss:.4f}')

        # 保存检查点
        save_checkpoint(epoch + 1, model, optimizer, epoch_loss, checkpoint_path)

        # 保存模型参数
        torch.save(model.state_dict(), f'{model_save_path}model_epoch_{epoch + 1}.pth')
        print(f'Model saved after epoch {epoch + 1}')