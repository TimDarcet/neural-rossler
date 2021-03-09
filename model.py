import torch
import torch.nn as nn
import pytorch_lightning as pl


class FC_block(nn.Module):
    def __init__(self, in_features, out_features, nonlinearity=torch.relu, dropout=0.2):
        super().__init__()
        self.residual = (in_features == out_features) and False
        self.fc = nn.Linear(in_features, out_features)
        self.nonlin = nonlinearity
        # self.bn = nn.BatchNorm1d(out_features)
        # self.do = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc(x)
        h = self.nonlin(h)
        # h = self.bn(h)
        # h = self.do(h)
        if self.residual:
            return x + h
        else:
            return h


class FC_predictor(pl.LightningModule):
    def __init__(self,
                 n_hidden,
                 in_size,
                 hidden_size,
                 out_size,
                 lr,
                 jacobian_loss,
                 a,
                 c,
                 jacobian_alpha,
                 norm_alpha,
                 optim):
        super().__init__()
        self.save_hyperparameters()
        self.n_hidden = n_hidden
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.lr = lr
        self.jacobian_loss = jacobian_loss
        self.a = a
        self.c = c
        self.jacobian_alpha = jacobian_alpha
        self.norm_alpha = norm_alpha
        self.optim = optim

        if n_hidden == 0:
            self.fcs = [FC_block(in_size, out_size)]
        else:
           self.fcs = nn.ModuleList([FC_block(in_size, hidden_size)]\
                                  + [FC_block(hidden_size, hidden_size) for _ in range(n_hidden - 1)]\
                                  + [nn.Linear(hidden_size, out_size)])
        self.crit = nn.MSELoss(reduction="mean")

    def forward(self, x):
        h = x
        for fc in self.fcs:
            h = fc(h)
        # h = self.fcs(x)
        # print(h.sum(dim=1))
        return x[:, -self.out_size:] + h
    
    def configure_optimizers(self):
        if self.optim == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.09)
        elif self.optim == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            assert False
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.2, verbose=True)
        return optim

    def training_step(self, batch, batch_idx):
        w_hist, w_last, w_new = batch[:, :-2, :], batch[:, -2, :], batch[:, -1, :]
        if self.jacobian_loss:
            w_last.requires_grad = True
        w_flat = torch.cat((w_hist.flatten(start_dim=1, end_dim=2), w_last), dim=1)

        w_pred = self(w_flat)
        # print(batch[0], w_flat[0], w_pred[0])
        pred_loss = self.crit(w_new, w_pred)
        norm_loss = (w_pred ** 2).sum()
        total_loss = pred_loss 
        total_loss += self.norm_alpha * norm_loss

        if self.jacobian_loss and self.out_size == 3:
            outputs = tuple(w_pred.sum(dim=0))
            # print(outputs, w_last.shape)
            j = torch.zeros((batch.shape[0], self.out_size, self.out_size))
            for i in range(self.out_size):
                j[:, i, :] = torch.autograd.grad(outputs[i], w_last, create_graph=True, only_inputs=True, allow_unused=True)[0]
            j_flat = j.flatten(start_dim=1, end_dim=2)
            j_flat_true = self.batch_jacobian(w_last)
            j_loss = self.crit(j_flat_true, j_flat)
            total_loss += self.jacobian_alpha * j_loss
            self.log('train_j_loss', j_loss, prog_bar=True)
            

        self.log('train_norm_loss', norm_loss, prog_bar=True)
        self.log('train_pred_loss', pred_loss, prog_bar=True)
        self.log('train_loss', total_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        w, w_new = batch[:, :-1, :], batch[:, -1, :]
        w = w.flatten(start_dim=1, end_dim=2)
        w_pred = self(w)
        loss = self.crit(w_new, w_pred)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def batch_jacobian(self, batch):
        batch_size = batch.shape[0]
        c0 = torch.tensor([0, -1, -1, 1, self.a, 0])[None, :].expand(batch_size, -1)
        c1 = batch[:, 2:3]
        c2 = torch.zeros((batch_size, 1))
        c3 = batch[:, 0:1] - self.c
        j = torch.cat((c0, c1, c2, c3), dim=1)
        return j
