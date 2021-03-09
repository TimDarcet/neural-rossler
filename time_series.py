import argparse
from scipy.interpolate import interp1d
import numpy as np
from model import FC_predictor
import torch
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--init", nargs="+", type=float,  default=[-5.75, -1.6, 0.02])
value = parser.parse_args()

class Rossler_model:
    def __init__(self, delta_t):
        self.delta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = int(100 // self.delta_t)

        self.initial_condition = np.array(value.init)
        self.predictor = FC_predictor.load_from_checkpoint("logs/tensorboard_logs/default/version_95/checkpoints/epoch=9-val_loss=0.00-train_loss=0.00.ckpt")
        self.predictor.eval()

    def full_traj(self, initial_condition=None, history=0, only_y=False):
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary. 

        if initial_condition is None:
            initial_condition = self.initial_condition
        # if only_y:
        #     trajectory = [torch.tensor(initial_condition).float()[1:2]] * (history + 1)
        # else:
        #     trajectory = [torch.tensor(initial_condition).float()] * (history + 1)
        trajectory = initial_condition
        for _ in range(self.nb_steps):
            # last_positions = torch.cat(trajectory[-1-history:])[None, :]
            last_positions = trajectory[-1][None, :]
            trajectory.append(self.predictor(last_positions)[0])

        y = trajectory  # TODO
        # t = np.linspace(0, 10000, int(10000 // self.delta_t))
        # t_new = np.linspace(0, 10000, int(10000 // 1e-2))
        # y = interp1d(t_new, t, trajectory)
        return y

    def save_traj(self,y):
        # save the trajectory in traj.npy file
        # y has to be a numpy array: y.shape = (1000000,)
          
        np.save('traj.npy', y)
        
    
if __name__ == '__main__':
    delta_t = 1e-3  # MEF: THIS IS HARDCODED
    history = 0
    only_y = False
    # init = [[-5.75,       -1.6,         0.02      ],
    #         [-5.74824068, -1.60674422,  0.019968  ],
    #         [-5.74647383, -1.61348797,  0.01993645],
    #         [-5.74469945, -1.62023126,  0.01990533],
    #         [-5.74291755, -1.62697407,  0.01987464],
    #         [-5.74112812, -1.6337164,   0.01984438],
    #         [-5.73933117, -1.64045823,  0.01981455],
    #         [-5.73752669, -1.64719956,  0.01978513],
    #         [-5.73571469, -1.65394037,  0.01975612],
    #         [-5.73389517, -1.66068067,  0.01972751]]
    # init = [torch.tensor(x[1])[None] for x in init]
    init = [-5.75,       -1.6,         0.02      ]
    init = [torch.tensor(init)]

    ROSSLER = Rossler_model(delta_t)

    y = ROSSLER.full_traj(initial_condition=init, history=history, only_y=only_y)
    print('predicted trajectory')
    y = torch.stack(y).detach().numpy()
    # y = [x.item() for x in y]
    # print(y)
    # plt.plot(y)
    # sns.kdeplot(x=y, bw_adjust=0.1)
    # plt.show()
    print(*y.mean(axis=1))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(y[:,0], y[:,1], y[:,2])
    plt.show()

    ROSSLER.save_traj(y)

