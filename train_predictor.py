import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import FC_predictor
from data import RosslerDataModule


def main(args):
    # Settings
    pl.seed_everything(42)
    
    # Handle the data
    dm = RosslerDataModule(init_pos=args.init_pos,
                           nb_samples=args.nb_samples,
                           batch_size=args.batch_size,
                           train_prop=args.train_prop,
                           delta_t=args.delta_t,
                           history=args.history)

    # Define model
    model = FC_predictor(n_hidden=args.n_hidden,
                         in_size=3 * (1 + args.history),
                         hidden_size=args.hidden_size,
                         lr=args.lr)

    # Exp logger
    logger = TensorBoardLogger('logs/tensorboard_logs')

    # Define training
    checkpointer = ModelCheckpoint(monitor='val_loss',
                                   save_top_k=3,
                                   mode='min',
                                   save_last=True,
                                   filename='{epoch}-{val_loss:.2f}-{train_loss:.2f}')
    trainer = pl.Trainer(gpus=1,
                         auto_select_gpus=True,
                         max_epochs=args.epochs,
                         callbacks=[checkpointer],
                         logger=logger)

    # Train
    trainer.fit(model, dm)


if __name__ ==  '__main__':
    parser = ArgumentParser()
    parser.add_argument('--init-pos', nargs="+", type=float,  default=[-5.75, -1.6, 0.02])
    parser.add_argument('--nb-samples', type=int, default=1000000)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--train-prop', type=float, default=0.8)
    parser.add_argument('--delta-t', type=float, default=1e-3)
    parser.add_argument('--history', type=int, default=0)

    parser.add_argument('--n-hidden', type=int, default=10)
    parser.add_argument('--hidden-size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()
    main(args)
