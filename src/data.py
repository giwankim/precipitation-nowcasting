import numpy as np
import torch


class NowcastingDataset(torch.utils.data.Dataset):
    def __init__(self, paths, test=False):
        self.paths = paths
        self.test = test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        data = np.load(path)
        x = data[:, :, :4]
        # x = x / 255.0
        x = x.astype(np.float32)
        x = torch.tensor(x, dtype=torch.float)
        x = x.permute(2, 0, 1)
        if self.test:
            return x
        else:
            y = data[:, :, 4]
            # y = y / 255.0
            y = y.astype(np.float32)
            y = torch.tensor(y, dtype=torch.float)
            y = y.unsqueeze(-1)
            y = y.permute(2, 0, 1)

            return x, y
