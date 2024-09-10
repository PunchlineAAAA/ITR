from torch.utils.data import Dataset


# 训练集类
class CustomDataset(Dataset):
    def __init__(self, captions_file):
        image_paths = []
        text_descs = []

        # 根据 captions.txt 格式读取
        with open(captions_file, 'r') as file:
            next(file)
            lines = file.readlines()

            for line in lines:
                split_lines = line.split(",", 1)
                image_paths.append(split_lines[0])
                text_descs.append(split_lines[1])

        # 抽取训练集
        self.image_paths = image_paths[:39000]
        self.text_descs = text_descs[:39000]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = "./dataset/images/" + self.image_paths[idx]
        cap = self.text_descs[idx]

        return img, cap


# 验证集类
class ValidationDataset(Dataset):
    def __init__(self, captions_file):
        image_paths = []
        text_descs = []

        # 根据 captions.txt 格式读取
        with open(captions_file, 'r') as file:
            next(file)
            lines = file.readlines()

            for line in lines:
                split_lines = line.split(",", 1)
                image_paths.append(split_lines[0])
                text_descs.append(split_lines[1])

        # 抽取验证集
        self.image_paths = image_paths[39001:39500]
        self.text_descs = text_descs[39001:39500]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = "./dataset/images/" + self.image_paths[idx]
        cap = self.text_descs[idx]

        return img, cap


# 测试集类
class TestDataset(Dataset):
    def __init__(self, captions_file):
        image_paths = []
        text_descs = []

        # 根据 captions.txt 格式读取
        with open(captions_file, 'r') as file:
            next(file)
            lines = file.readlines()

            for line in lines:
                split_lines = line.split(",", 1)
                image_paths.append(split_lines[0])
                text_descs.append(split_lines[1])

        # 抽取测试集
        self.image_paths = image_paths[39501:]
        self.text_descs = text_descs[39501:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = "./dataset/images/" + self.image_paths[idx]
        cap = self.text_descs[idx]

        return img, cap