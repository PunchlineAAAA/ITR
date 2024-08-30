import os.path

from Dataset import CustomDataset
import torch
from torch.utils.data import DataLoader
from Loss import Loss
import torch.optim as optim


def load_checkpoint(checkpoint_path, model, optimizer):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss
    else:
        print("No checkpoint found, starting from scratch")
        return 0, float('inf')


def save_checkpoint(epoch, model, optimizer, loss, checkpoint_path):
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
    batch_size = 8
    learning_rate = 0.0002
    num_epochs = 10
    model_save_path = "./saved_models/"
    checkpoint_path = os.path.join(model_save_path, "latest_checkpoint.pth")

    # 数据加载
    dataset = CustomDataset("./dataset/shorter_captions.txt")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 实例化模型
    model = Loss().to(device)

    optimizer = optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=learning_rate)

    # 加载检查点（如果存在）
    start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)

    # 训练开始
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0

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

        # 保存检查点
        save_checkpoint(epoch + 1, model, optimizer, epoch_loss, checkpoint_path)

        # 保存模型参数
        torch.save(model.state_dict(), f'{model_save_path}model_epoch_{epoch + 1}.pth')
        print(f'Model saved after epoch {epoch + 1}')