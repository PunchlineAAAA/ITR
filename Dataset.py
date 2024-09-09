from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, captions_file):
        image_paths = []
        text_descs = []

        with open(captions_file, 'r') as file:
            next(file)
            lines = file.readlines()

            for line in lines:
                split_lines = line.split(",", 1)
                image_paths.append(split_lines[0])
                text_descs.append(split_lines[1])

        self.image_paths = image_paths
        self.text_descs = text_descs

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = "./dataset/images/" + self.image_paths[idx]
        cap = self.text_descs[idx]

        return img, cap

class ValidationDataset(Dataset):
    def __init__(self, captions_file):
        image_paths = []
        text_descs = []

        with open(captions_file, 'r') as file:
            next(file)
            lines = file.readlines()

            for line in lines:
                split_lines = line.split(",", 1)
                image_paths.append(split_lines[0])
                text_descs.append(split_lines[1])

        self.image_paths = image_paths[39001:39500]
        self.text_descs = text_descs[39001:39500]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = "./dataset/images/" + self.image_paths[idx]
        cap = self.text_descs[idx]

        return img, cap

class TestDataset(Dataset):
    def __init__(self, captions_file):
        image_paths = []
        text_descs = []

        with open(captions_file, 'r') as file:
            next(file)
            lines = file.readlines()

            for line in lines:
                split_lines = line.split(",", 1)
                image_paths.append(split_lines[0])
                text_descs.append(split_lines[1])

        self.image_paths = image_paths[39501:]
        self.text_descs = text_descs[39501:]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = "./dataset/images/" + self.image_paths[idx]
        cap = self.text_descs[idx]

        return img, cap