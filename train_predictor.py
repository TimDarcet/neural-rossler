import os
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from model import FC_predictor
from data import RosslerDataModule


def main(arg_groups):
    # Settings
    pl.seed_everything(42)
    
    # Handle the data
    # dm = RosslerDataModule(init_pos=args.init_pos,
    #                        nb_samples=args.nb_samples,
    #                        batch_size=args.batch_size,
    #                        train_prop=args.train_prop,
    #                        delta_t=args.delta_t,
    #                        history=args.history,
    #                        only_y=args.only_y)
    dm = RosslerDataModule(**arg_groups['Datamodule parameters'])


    # Define model
    # model = FC_predictor(n_hidden=args.n_hidden,
    #                      in_size=(3 - 2 * int(args.only_y)) * (1 + args.history),
    #                      hidden_size=args.hidden_size,
    #                      out_size=3 - 2 * int(args.only_y),
    #                      lr=args.lr,
    #                      jacobian_loss=args.jacobian_loss,
    #                      a=0.2,
    #                      c=5.7,
    #                      jacobian_alpha=args.jacobian_alpha,
    #                      norm_alpha=args.norm_alpha)
    model = FC_predictor(in_size=(3 - 2 * int(args.only_y)) * (1 + args.history),
                         out_size=3 - 2 * int(args.only_y),
                         a=0.2,
                         c=5.7,
                         **arg_groups['Model parameters'])

    # Exp logger
    logger = TensorBoardLogger('logs/tensorboard_logs')

    # Define training
    checkpointer = ModelCheckpoint(monitor='val_loss',
                                   save_top_k=3,
                                   mode='min',
                                   save_last=True,
                                   filename='{epoch}-{val_loss:.2f}-{train_loss:.2f}')
    # trainer = pl.Trainer(gpus=1,
    #                      max_epochs=args.epochs,
    #                      callbacks=[checkpointer],
    #                      val_check_interval=0.5,
    #                      logger=logger,
    #                      resume_from_checkpoint=args.checkpoint)
    trainer = pl.Trainer(**arg_groups['Trainer parameters'],
                         gpus=1,
                         callbacks=[checkpointer],
                         val_check_interval=0.5,
                         logger=logger)


    # Train
    trainer.fit(model, dm)


if __name__ ==  '__main__':
    parser = ArgumentParser()

    # Datamodule parameters
    datamodule_params = parser.add_argument_group('Datamodule parameters')
    datamodule_params.add_argument('--init-pos', nargs="+", type=float,  default=[-5.75, -1.6, 0.02])
    datamodule_params.add_argument('--nb-samples', type=int, default=5000000)
    datamodule_params.add_argument('--batch-size', type=int, default=64)
    datamodule_params.add_argument('--train-prop', type=float, default=0.8)
    datamodule_params.add_argument('--delta-t', type=float, default=1e-3)
    datamodule_params.add_argument('--history', type=int, default=0)
    datamodule_params.add_argument('--only-y', type=bool, default=False)

    # Model parameters
    model_params = parser.add_argument_group('Model parameters')
    model_params.add_argument('--n-hidden', type=int, default=2)
    model_params.add_argument('--hidden-size', type=int, default=10)
    model_params.add_argument('--lr', type=float, default=1e-0)
    model_params.add_argument('--jacobian-loss', type=bool, default=False)
    model_params.add_argument('--jacobian-alpha', type=float, default=0.001)
    model_params.add_argument('--norm-alpha', type=float, default=0)
    model_params.add_argument('--optim', type=str, default="sgd")

    # Trainer parameters
    trainer_params = parser.add_argument_group('Trainer parameters')
    trainer_params.add_argument('--max-epochs', type=int, default=20)
    trainer_params.add_argument('--checkpoint', type=str, default=None, dest="resume_from_checkpoint")
    #"logs/tensorboard_logs/default/version_95/checkpoints/epoch=9-val_loss=0.00-train_loss=0.00.ckpt"

    args = parser.parse_args()

    # Parse different arg groups
    arg_groups={}
    for group in parser._action_groups:
        arg_groups[group.title] = {a.dest:getattr(args,a.dest,None) for a in group._group_actions}
        print(group.title)
    
    main(arg_groups)
