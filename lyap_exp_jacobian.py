import argparse
from scipy.interpolate import interp1d
import numpy as np
from model import FC_predictor
import torch
import matplotlib.pyplot as plt
from time_series import Rossler_model

print('ye')
parser = argparse.ArgumentParser()
parser.add_argument("--init", nargs="+", type=float,  default=[-5.75, -1.6, 0.02])
parser.add_argument("--sigma", type=float,  default=0.0001)
parser.add_argument("--nsteps", type=int,  default=1000)
parser.add_argument("--nexps", type=int,  default=1024)
args = parser.parse_args()
print('ye')

# Fake delta_t to have the wanted number of iterations in the rossler model
# True delta_t is 1e-2 in model
fake_delta_t = 10000 / args.nsteps
delta_t = 1e-2
print('ye')

init = torch.tensor(args.init)
print('ye')

rm = Rossler_model(fake_delta_t)
print("Built model")
traj = rm.full_traj_batch(init[None, :])
print("Calculated trajectory")
jacobians = rm.batch_model_jacobian(traj).numpy()
print("Calculated jacobians")

d = 3
w = np.eye(d)
rs = []
chk = 0

for i in range(traj.shape[0]):
    jacob = jacobians[i]
    #WARNING this is true for the jacobian of the continuous system!
    w_next = np.dot(expm(jacob * delta_t), w) 
    #if delta_t is small you can use:
    #w_next = np.dot(np.eye(n)+jacob * delta_t,w)

    w_next, r_next = qr(w_next)

    # qr computation from numpy allows negative values in the diagonal
    # Next three lines to have only positive values in the diagonal
    d = np.diag(np.sign(r_next.diagonal()))
    w_next = np.dot(w_next, d)
    r_next = np.dot(d, r_next.diagonal())

    rs.append(r_next)
    w = w_next
    if i//(max_it/100)>chk:
        print(i//(max_it/100))
        chk +=1

print(np.mean(np.log(rs), axis=0) / delta_t)
