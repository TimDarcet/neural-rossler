import argparse
from script.interpolate import interp1d
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--init", nargs="+", type=float,  default=[-5.75, -1.6, 0.02])
value = parser.parse_args()

class Rossler_model:
    def __int__(self, delta_t):
        self.deta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = 10000 // self.delta_t

        self.rosler_nn = LOAD_YOUR_MODEL
        self.initial_condition = np.array(value.init)
        self.predictor = None  # TODO

    def full_traj(self, initial_condition=None):
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary. 

        if initial_condition is None:
            initial_condition = self.initial_condition
        trajectory = [initial_condition]
        for _ in range(self.nb_steps):
            trajectory.append(self.predictor(trajectory[-1]))

        t = np.linspace(0, 10000, 10000 // self.delta_t)
        t_new = np.linspace(0, 10000, 10000 // 1e-2)
        y = interp1d(t_new, t, trajectory)
        return y

    def save_traj(self,y):
        #save the trajectory in traj.npy file
        # y has to be a numpy array: y.shape = (1000000,)
          
        np.save('traj.npy', y)
        
    
if __name__ == '__main__':

    ROSSLER = Rossler_model(delta_t)

    y = ROSSLER.full_traj()

    ROSSLER.save_traj(y)

