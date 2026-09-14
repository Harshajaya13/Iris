import numpy as np
import torch
import os

class DataLoader:
    def __init__(self, block_size, batch_size, device, data_dir="."):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        train_path = os.path.join(data_dir, "train.bin")
        val_path = os.path.join(data_dir, "val.bin")

        self.train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
        self.val_data = np.memmap(val_path, dtype=np.uint16, mode="r")

    def get_batch(self, split="train"):
        d = self.train_data if split == "train" else self.val_data
        idx = torch.randint(0, len(d) - self.block_size, (self.batch_size,))

        x_list = [torch.from_numpy((d[i : i + self.block_size]).astype(np.int64)) for i in idx]
        y_list = [torch.from_numpy((d[i + 1 : i + 1 + self.block_size]).astype(np.int64)) for i in idx]

        x = torch.stack(x_list)
        y = torch.stack(y_list)

        if "cuda" in str(self.device):
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)

        return x, y