import argparse
from scipy.interpolate import interp1d
import numpy as np
from model import FC_predictor
import torch
import matplotlib.pyplot as plt
from time_series import Rossler_model


parser = argparse.ArgumentParser()
parser.add_argument("--init", nargs="+", type=float,  default=[-5.75, -1.6, 0.02])
parser.add_argument("--sigma", type=float,  default=0.0001)
parser.add_argument("--nsteps", nargs="+", type=int,  default=torch.arange(100))
parser.add_argument("--nexps", type=int,  default=1024)
args = parser.parse_args()


# delta_t = 10000 / max(args.nsteps)
# history = 0
# only_y = False
# init = args.init
# init = torch.tensor(init)

# ROSSLER = Rossler_model(delta_t)
# ROSSLER.nb_steps += 1

# output_base = ROSSLER.full_traj_batch(init[None, :])[args.nsteps]

# init_noised = init[None, :] + args.sigma * torch.randn(args.nexps, 3)
# outputs = ROSSLER.full_traj_batch(init_noised)[args.nsteps]

# # outputs = []
# # for _ in range(args.nexps):
# #     init_here = [init + args.sigma * torch.randn(3)]
# #     outputs.append(ROSSLER.full_traj(initial_condition=init_here, history=history, only_y=only_y)[args.nsteps])
# # outputs = np.stack(outputs)
# print('predicted trajectories')

# # print(outputs.shape, output_base.shape)
# mse = ((outputs - output_base) ** 2).sum(dim=2).mean(dim=1)  # Mean variance
# mstd = torch.sqrt(((outputs - output_base) ** 2).sum(dim=2)).mean(dim=1)  # Mean std
# print(mse.shape, args.nsteps.shape)
# print(torch.log(mse / mse[0]) / (args.nsteps - args.nsteps[0]))
# print(torch.log(mstd / mstd[0]) / (args.nsteps - args.nsteps[0]))
# plt.plot(args.nsteps, mse)
# plt.show()
# plt.plot(args.nsteps, mstd)
# plt.show()

# # fig = plt.figure()
# # ax = fig.gca(projection='3d')
# # ax.scatter(outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2])
# # # ax.scatter(outputs[:, 1, 0], outputs[:, 1, 1], outputs[:, 1, 2], c='red')
# # plt.show()

# ROSSLER.save_traj(outputs)
