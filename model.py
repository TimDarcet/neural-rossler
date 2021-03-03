import torch
import torch.nn as nn
import pytorch_lightning as pl


class FC_block(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=torch.relu):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.nonlin = nonlinearity
    
    def forward(self, x):
        return self.nonlin(self.fc(x))


class FC_predictor(pl.LightningModule):
    def __init__(self, n_hidden, in_size, hidden_size, out_size, lr):
        super().__init__()
        self.save_hyperparameters()
        self.n_hidden = n_hidden
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.lr = lr
        if n_hidden == 0:
            self.fcs = [FC_block(in_size, 3)]
        else:
           self.fcs = nn.ModuleList([FC_block(in_size, hidden_size)]\
                                  + [FC_block(hidden_size, hidden_size) for _ in range(n_hidden - 1)]\
                                  + [FC_block(hidden_size, out_size)])
        self.crit = nn.MSELoss(reduction="mean")

    def forward(self, x):
        h = x
        for fc in self.fcs:
            h = fc(h)
        return x[:, -self.out_size:] + h
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim

    def training_step(self, batch, batch_idx):
        w, w_new = batch[:, :-1, :], batch[:, -1, :]
        w = w.flatten(start_dim=1, end_dim=2)
        w_pred = self(w)
        loss = self.crit(w_new, w_pred)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        w, w_new = batch[:, :-1, :], batch[:, -1, :]
        w = w.flatten(start_dim=1, end_dim=2)
        w_pred = self(w)
        loss = self.crit(w_new, w_pred)
        self.log('val_loss', loss, prog_bar=True)
        return loss
