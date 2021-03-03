import argparse
from scipy.interpolate import interp1d
import numpy as np
from model import FC_predictor
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--init", nargs="+", type=float,  default=[-5.75, -1.6, 0.02])
value = parser.parse_args()

class Rossler_model:
    def __init__(self, delta_t):
        self.delta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = int(10 // self.delta_t)

        self.initial_condition = np.array(value.init)
        self.predictor = FC_predictor.load_from_checkpoint("logs/tensorboard_logs/default/version_27/checkpoints/last.ckpt")
        self.predictor.eval()

    def full_traj(self, initial_condition=None, history=0, only_y=False):
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary. 

        if initial_condition is None:
            initial_condition = self.initial_condition
        if only_y:
            trajectory = [torch.tensor(initial_condition).float()[1:2]] * (history + 1)
        else:
            trajectory = [torch.tensor(initial_condition).float()] * (history + 1)
        for _ in range(self.nb_steps):
            last_positions = torch.cat(trajectory[-1-history:])[None, :]
            trajectory.append(self.predictor(last_positions)[0])

        # y = trajectory
        t = np.linspace(0, 10000, int(10 // self.delta_t))
        t_new = np.linspace(0, 10000, int(10000 // 1e-2))
        y = interp1d(t_new, t, trajectory)
        return y

    def save_traj(self,y):
        #save the trajectory in traj.npy file
        # y has to be a numpy array: y.shape = (1000000,)
          
        np.save('traj.npy', y)
        
    
if __name__ == '__main__':
    delta_t = 1e-3  # MEF: THIS IS HARDCODED
    history = 3
    only_y = True
    ROSSLER = Rossler_model(delta_t)

    y = ROSSLER.full_traj(history=history, only_y=only_y)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(y[:,0], y[:,1], y[:,2])

    ROSSLER.save_traj(y)

