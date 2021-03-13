import pytorch_lightning as pl
from rossler_map import RosslerMap
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class RosslerDataset(Dataset):
    def __init__(self, nb_samples, init_pos, delta_t, history, only_y, fake_prop):
        super().__init__()
        self.nb_samples = nb_samples
        self.init_pos = init_pos
        self.delta_t = delta_t
        self.history = history
        self.only_y = only_y
        self.fake_n = int(fake_prop * nb_samples)
        self.mask = torch.ones((self.history + 2, 1))
        self.mask[-1, 0] = 0  # Set target to 0

        self.rm = RosslerMap(delta_t=delta_t)

        self.w, self.t = self.rm.full_traj(nb_samples, init_pos)
        self.w = torch.tensor(self.w).float()
        fake_w = 20 * torch.randn((self.fake_n, 3 - 2 * int(only_y))) 
        self.w = torch.cat((self.w, fake_w), dim=0)

    def __len__(self):
        # Omit the last one because we do not know the next position
        return self.nb_samples - 1 - self.history + self.fake_n

    def __getitem__(self, idx):
        # if self.only_y:
        #     ret = self.w[idx : idx + self.history + 2, 1:2]
        # else:
        #     ret = self.w[idx : idx + self.history + 2]
        # if idx >= self.nb_samples - 1 - self.history:
        #     ret *= self.mask
        ret = self.w[idx : idx + self.history + 2]
        return ret


class RosslerDataModule(pl.LightningDataModule):
    def __init__(self, init_pos, nb_samples, batch_size, train_prop, delta_t, history, only_y, fake_prop):
        super().__init__()
        self.batch_size = batch_size
        self.train_prop = train_prop
        self.nb_samples = nb_samples
        self.init_pos = init_pos
        self.delta_t = delta_t
        self.history = history
        self.only_y = only_y
        self.fake_prop = fake_prop

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        self.full_dataset = RosslerDataset(self.nb_samples,
                                           self.init_pos,
                                           self.delta_t,
                                           self.history,
                                           self.only_y,
                                           self.fake_prop)

        if stage == 'fit' or stage is None:
            train_size = int(len(self.full_dataset) * self.train_prop)
            val_size = len(self.full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(self.full_dataset, [train_size, val_size])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test_dataset = self.full_dataset

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=4,
                                           pin_memory=False)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=4,
                                           pin_memory=False)