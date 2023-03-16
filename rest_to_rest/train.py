import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import numpy as np
from util import *
from environment import GridEnvironment
from agent import DynamicalSystem
from qlearning import *
from PIL import Image

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 24,
    "text.latex.preamble" : r"\usepackage{amsmath}\usepackage{siunitx}\sisetup{per-mode=symbol}"
})


def polyx_fourth(t):
    return 0.0013 * t**4 - 0.0629 * t**3 + 0.8751 * t**2 - 2.8376 * t + 2.2531

def polyy_fourth(t):
    return 0.0024 * t**4 - 0.0641 * t**3 + 0.3837 * t**2 + 0.7598 * t - 1.2001

def polyx_third(t):
    return -0.0211 * t**3 + 0.4353 * t**2 - 1.1507 * t + 0.5172

def polyy_third(t):
    return 0.0116 * t**3 - 0.4133 * t**2 + 3.8168 * t - 4.3458


if __name__ == '__main__':
    map = np.ones((10, 15))
    map[7, 0] = 0
    map[3, 2] = 0
    map[9, 4] = 0
    map[4, 6] = 0
    map[8, 8] = 0
    map[0:3, 4:9] = 0
    map[7:10, 12:15] = 0
    map[2:4, 11] = 0
    map[2, 9:11] = 0

    initial_state = (0, 0, 0, 0)
    final_state = (11, 0, 0, 0)
    
    vstates = generate_states_vector(map, [-2, -1, 0, 1, 2])
    vactions = generate_actions_vector([-1, 0, 1])

    env = GridEnvironment(map)
    agent = DynamicalSystem(vstates, vactions, -2, 2)
    algorithm = {
        'Q': QLearning(initial_state, final_state, agent, env, 100000, 0.9, 0.9, 0.1, False),
        'SARSA': SARSA(initial_state, final_state, agent, env, 100000, 0.5, 0.9, 0.1, False)
    }

    try:
        while 1:
            cmd = input('cmd> ').split()
            if cmd[0] == 'map':
                size = np.shape(map)
                fig, ax = plt.subplots()
                fig.canvas.manager.set_window_title('Occupancy Grid Map')
                fig.set_size_inches((10, 10))
    
                ax.set_xlabel(r'$x$ [m]', fontsize=22)
                ax.set_ylabel(r'$y$ [m]', fontsize=22)
    
                ax.set_xticks(np.arange(0, size[1], 1))
                ax.set_yticks(np.arange(0, size[0], 1))
    
                ax.set_xticks(np.arange(-0.5, size[1], 1), minor=True)
                ax.set_yticks(np.arange(-0.5, size[0], 1), minor=True)

                ax.tick_params(which='minor', bottom=False, left=False)

                ax.grid(which='minor')
                ax.imshow(map, cmap='gray', origin='lower')
                ax.plot(initial_state[0], initial_state[1], 'go', markersize=20)
                ax.plot(final_state[0], final_state[1], 'r*', markersize=20)
                fig.show()
            elif cmd[0] == 'compare':
                convgQ, _ = algorithm['Q'].learn()
                algorithm['Q'].save('models/Q.npy')
                convgSARSA, _ = algorithm['SARSA'].learn()
                algorithm['SARSA'].save('models/SARSA.npy')
 
                fig, ax = plt.subplots()
                ax.set_xlabel(r'Episode', fontsize=22)
                ax.set_ylabel(r'$$\max_u Q_k(Z_1, u)$$', fontsize=22)
                ax.set_yticks(np.arange(-8, 3, 2))
                ax.set_ylim([-8.5, 2.5])
                ax.plot(np.arange(np.shape(convgQ)[0]), convgQ, label='Q Learning')
                ax.plot(np.arange(np.shape(convgQ)[0]), convgSARSA, label=r'SARSA')
                ax.legend()
                fig.show()
            elif cmd[0] == 'learn':
                algorithm[cmd[1]].learn()
                algorithm[cmd[1]].save('models/' + cmd[1])
            elif cmd[0] == 'load':
                algorithm[cmd[1]].load(cmd[2])
            elif cmd[0] == 'plot':
                path, aseq = algorithm[cmd[1]].get_policy()
                print(path)
                print(aseq)
                fig = plt.figure('Summary', figsize=(15, 20), tight_layout=True)
                gs = gridspec.GridSpec(4, 2)

                size = np.shape(map)
                ax = fig.add_subplot(gs[0, 0])
                ax.set_xlabel(r'$x$ [\si{\metre}]')
                ax.set_ylabel(r'$y$ [\si{\metre}]')
                ax.set_xlabel(r'$x$ [\si{\metre}]')
                ax.set_ylabel(r'$y$ [\si{\metre}]')
                ax.set_xticks(np.arange(0, size[1], 1))
                ax.set_yticks(np.arange(0, size[0], 1))
                ax.set_xticks(np.arange(-0.5, size[1], 1), minor=True)
                ax.set_yticks(np.arange(-0.5, size[0], 1), minor=True)
                ax.tick_params(which='minor', bottom=False, left=False)
                ax.tick_params(axis='both', which='major', labelsize=18)
                ax.grid(which='minor')
                ax.imshow(map, cmap='gray', origin='lower')
                ax.plot(path[:, 0], path[:, 1], 'b--')
                ax.plot(path[0, 0], path[0, 1], 'go')
                ax.plot(path[-1, 0], path[-1, 1], 'r*')

                img = np.asarray(Image.open('img/rviz.png'))
                ax = fig.add_subplot(gs[0, 1])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(img)
                
                measurements = np.load('/home/bencic/measurements.npy')
                pos = measurements[0]
                vel = measurements[1]
                acc = measurements[2]
                size = pos.shape[0]

                ax = fig.add_subplot(gs[1, 0])
                ax.set_xlabel(r'$t$ [\si{\second}]')
                ax.set_ylabel(r'$x$ [\si{\metre}]')
                ax.set_xticks(np.arange(0, 15, 1))
                ax.set_yticks(np.arange(0, 15, 2))
                ax.set_xlim([-0.5, 14.5])
                ax.set_ylim([-0.5, 14.5])
                ax.plot(np.arange(np.shape(path)[0]), path[:, 0], label='ITG')
                ax.plot(np.linspace(0, 14, size), pos[:, 0], label='UAV Response')
                ax.legend(prop={'size': 18})
                
                ax = fig.add_subplot(gs[1, 1])
                ax.set_xlabel(r'$t$ [\si{\second}]')
                ax.set_ylabel(r'$y$ [\si{\metre}]')
                ax.set_xticks(np.arange(0, 15, 1))
                ax.set_yticks(np.arange(0, 15, 2))
                ax.set_xlim([-0.5, 14.5])
                ax.set_ylim([-0.5, 14.5])
                ax.plot(np.arange(np.shape(path)[0]), path[:, 1], label='ITG')
                ax.plot(np.linspace(0, 14, size), pos[:, 1], label='UAV Response')
                ax.legend(prop={'size': 18})
                
                ax = fig.add_subplot(gs[2, 0])
                ax.set_xlabel(r'$t$ [\si{\second}]')
                ax.set_ylabel(r'$v_x$ [\si{\metre\per\second}]')
                ax.set_xticks(np.arange(0, 15, 1))
                ax.set_yticks(np.arange(-2, 3, 1))
                ax.set_xlim([-0.5, 14.5])
                ax.set_ylim([-2.5, 2.5])
                ax.plot(np.arange(np.shape(path)[0]), path[:, 2], label='ITG')
                ax.plot(np.linspace(0, 14, size), vel[:, 0], label='UAV Response')
                ax.legend(prop={'size': 18})

                ax = fig.add_subplot(gs[2, 1])
                ax.set_xlabel(r'$t$ [\si{\second}]')
                ax.set_ylabel(r'$v_y$ [\si{\metre\per\second}]')
                ax.set_xticks(np.arange(0, 15, 1))
                ax.set_yticks(np.arange(-2, 3, 1))
                ax.set_xlim([-0.5, 14.5])
                ax.set_ylim([-2.5, 2.5])
                ax.plot(np.arange(np.shape(path)[0]), path[:, 3], label='ITG')
                ax.plot(np.linspace(0, 14, size), vel[:, 1], label='UAV Response')
                ax.legend(prop={'size': 18})

                ax = fig.add_subplot(gs[3, 0])
                ax.set_xlabel(r'$t$ [\si{\second}]')
                ax.set_ylabel(r'$u_{x}$ [\si{\metre\per\second\squared}]')
                ax.set_xticks(np.arange(0, 15, 1))
                ax.set_yticks(np.arange(-2, 3, 1))
                ax.set_xlim([-0.5, 14.5])
                ax.set_ylim([-1.5, 1.5])
                ax.plot(np.arange(np.shape(aseq)[0]), aseq[:, 0], label='ITG')
                ax.plot(np.linspace(0, 14, size), acc[:, 0], label='UAV Response')
                ax.legend(prop={'size': 18})

                ax = fig.add_subplot(gs[3, 1])
                ax.set_xlabel(r'$t$ [\si{\second}]')
                ax.set_ylabel(r'$u_{y}$ [\si{\metre\per\second\squared}]')
                ax.set_xticks(np.arange(0, 15, 1))
                ax.set_yticks(np.arange(-2, 3, 1))
                ax.set_xlim([-0.5, 14.5])
                ax.set_ylim([-1.5, 1.5])
                ax.plot(np.arange(np.shape(aseq)[0]), aseq[:, 1], label='ITG')
                ax.plot(np.linspace(0, 14, size), acc[:, 1], label='UAV Response')
                ax.legend(prop={'size': 18})

                fig.show()
            elif cmd[0] == "control":
                path, aseq = algorithm[cmd[1]].get_policy()

                measurements = np.load('/home/bencic/measurements_itg.npy')
                pos = measurements[0]
                vel = measurements[1]
                acc = measurements[2]
                size = pos.shape[0]

                third = np.load('/home/bencic/measurements_3th.npy')
                third_pos = third[0]
                third_vel = third[1]
                third_acc = third[2]
                third_size = third_pos.shape[0]

                fourth = np.load('/home/bencic/measurements_4th.npy')
                fourth_pos = fourth[0]
                fourth_vel = fourth[1]
                fourth_acc = fourth[2]
                fourth_size = fourth_pos.shape[0]

                print(pos.shape[0])
                print(third_acc.shape[0])
                print(fourth_acc.shape[0])

                fig = plt.figure('Summary', figsize=(15, 17), tight_layout=True)
                gs = gridspec.GridSpec(3, 2)

                ax = fig.add_subplot(gs[0, 0])
                ax.set_xlabel(r'$t$ [\si{\second}]')
                ax.set_ylabel(r'$x$ [\si{\metre}]')
                ax.set_xticks(np.arange(0, 15, 1))
                ax.set_yticks(np.arange(0, 15, 2))
                ax.set_xlim([-0.5, 14.5])
                ax.set_ylim([-0.5, 14.5])
                ax.plot(np.arange(np.shape(path)[0]), path[:, 0], label='ITG')
                ax.plot(np.linspace(0, 14, size), pos[:, 0], label='UAV Response')
                ax.plot(np.linspace(0, 14, third_size), third_pos[:, 0], label='Third Order Response')
                ax.plot(np.linspace(0, 14, fourth_size), fourth_pos[:, 0], label='Fourth Order Response')
                ax.legend(prop={'size': 18})
                
                ax = fig.add_subplot(gs[0, 1])
                ax.set_xlabel(r'$t$ [\si{\second}]')
                ax.set_ylabel(r'$y$ [\si{\metre}]')
                ax.set_xticks(np.arange(0, 15, 1))
                ax.set_yticks(np.arange(0, 15, 2))
                ax.set_xlim([-0.5, 14.5])
                ax.set_ylim([-0.5, 14.5])
                ax.plot(np.arange(np.shape(path)[0]), path[:, 1], label='ITG')
                ax.plot(np.linspace(0, 14, size), pos[:, 1], label='UAV Response')
                ax.plot(np.linspace(0, 14, third_size), third_pos[:, 1], label='Third Order Response')
                ax.plot(np.linspace(0, 14, fourth_size), fourth_pos[:, 1], label='Fourth Order Response')
                ax.legend(prop={'size': 18})
                
                ax = fig.add_subplot(gs[1, 0])
                ax.set_xlabel(r'$t$ [\si{\second}]')
                ax.set_ylabel(r'$v_x$ [\si{\metre\per\second}]')
                ax.set_xticks(np.arange(0, 15, 1))
                ax.set_yticks(np.arange(-2, 3, 1))
                ax.set_xlim([-0.5, 14.5])
                ax.set_ylim([-2.5, 2.5])
                ax.plot(np.arange(np.shape(path)[0]), path[:, 2], label='ITG')
                ax.plot(np.linspace(0, 14, size), vel[:, 0], label='UAV Response')
                ax.plot(np.linspace(0, 14, third_size), third_vel[:, 0], label='Third Order Response')
                ax.plot(np.linspace(0, 14, fourth_size), fourth_vel[:, 0], label='Fourth Order Response')
                ax.legend(prop={'size': 18})

                ax = fig.add_subplot(gs[1, 1])
                ax.set_xlabel(r'$t$ [\si{\second}]')
                ax.set_ylabel(r'$v_y$ [\si{\metre\per\second}]')
                ax.set_xticks(np.arange(0, 15, 1))
                ax.set_yticks(np.arange(-2, 3, 1))
                ax.set_xlim([-0.5, 14.5])
                ax.set_ylim([-2.5, 2.5])
                ax.plot(np.arange(np.shape(path)[0]), path[:, 3], label='ITG')
                ax.plot(np.linspace(0, 14, size), vel[:, 1], label='UAV Response')
                ax.plot(np.linspace(0, 14, third_size), third_vel[:, 1], label='Third Order Response')
                ax.plot(np.linspace(0, 14, fourth_size), fourth_vel[:, 1], label='Fourth Order Response')
                ax.legend(prop={'size': 18})

                ax = fig.add_subplot(gs[2, 0])
                ax.set_xlabel(r'$t$ [\si{\second}]')
                ax.set_ylabel(r'$u_{x}$ [\si{\metre\per\second\squared}]')
                ax.set_xticks(np.arange(0, 15, 1))
                ax.set_yticks(np.arange(-2, 3, 1))
                ax.set_xlim([-0.5, 14.5])
                ax.set_ylim([-1.5, 1.5])
                ax.plot(np.arange(np.shape(aseq)[0]), aseq[:, 0], label='ITG')
                ax.plot(np.linspace(0, 14, size), acc[:, 0], label='UAV Response')
                ax.plot(np.linspace(0, 14, third_size), third_acc[:, 0], label='Third Order Response')
                ax.plot(np.linspace(0, 14, fourth_size), fourth_acc[:, 0], label='Fourth Order Response')
                ax.legend(prop={'size': 18})

                ax = fig.add_subplot(gs[2, 1])
                ax.set_xlabel(r'$t$ [\si{\second}]')
                ax.set_ylabel(r'$u_{y}$ [\si{\metre\per\second\squared}]')
                ax.set_xticks(np.arange(0, 15, 1))
                ax.set_yticks(np.arange(-2, 3, 1))
                ax.set_xlim([-0.5, 14.5])
                ax.set_ylim([-1.5, 1.5])
                ax.plot(np.arange(np.shape(aseq)[0]), aseq[:, 1], label='ITG')
                ax.plot(np.linspace(0, 14, size), acc[:, 1], label='UAV Response')
                ax.plot(np.linspace(0, 14, third_size), third_acc[:, 1], label='Third Order Response')
                ax.plot(np.linspace(0, 14, fourth_size), fourth_acc[:, 1], label='Fourth Order Response')
                ax.legend(prop={'size': 18})

                fig.show()
            elif cmd[0] == "rmse":
                path, aseq = algorithm[cmd[1]].get_policy()

                third = np.load('path/to/measurements')
                third_pos = third[0]
                third_vel = third[1]
                third_acc = third[2]
                third_size = third_pos.shape[0]

                fourth = np.load('path/to/measurements')
                fourth_pos = fourth[0]
                fourth_vel = fourth[1]
                fourth_acc = fourth[2]
                fourth_size = fourth_pos.shape[0]

                gt_x = path[:, 0]
                gt_y = path[:, 1]


                print(gt_x)

                rmse_third_x = math.sqrt(np.mean((third_pos[np.linspace(0,  third_size - 1, 15).astype('int'), 0] - gt_x)**2))
                rmse_third_y = math.sqrt(np.mean((third_pos[np.linspace(0,  third_size - 1, 15).astype('int'), 1] - gt_y)**2))
                rmse_fourth_x = math.sqrt(np.mean((fourth_pos[np.linspace(0,  fourth_size - 1, 15).astype('int'), 0] - gt_x)**2))
                rmse_fourth_y = math.sqrt(np.mean((fourth_pos[np.linspace(0,  fourth_size - 1, 15).astype('int'), 1] - gt_y)**2))

                rmse_third_x = math.sqrt(np.mean((polyx_third(np.linspace(0,  14, 15)) - gt_x)**2))
                rmse_third_y = math.sqrt(np.mean((polyy_third(np.linspace(0,  14, 15)) - gt_y)**2))
                rmse_fourth_x = math.sqrt(np.mean((polyx_fourth(np.linspace(0,  14, 15)) - gt_x)**2))
                rmse_fourth_y = math.sqrt(np.mean((polyy_fourth(np.linspace(0,  14, 15)) - gt_y)**2))

                print("RMSE 3rd x: " + str(rmse_third_x))
                print("RMSE 3rd y: " + str(rmse_third_y))
                print("RMSE 4th x: " + str(rmse_fourth_x))
                print("RMSE 4th y: " + str(rmse_fourth_y))
            elif cmd[0] == "exit":
                raise KeyboardInterrupt
            else:
                print('unknown command')
    except KeyboardInterrupt:
        sys.exit()
