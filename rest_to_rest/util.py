import numpy as np
import matplotlib.pyplot as plt

def generate_states_vector(map_width, map_height, vels):
    states = np.empty((0, 4), dtype='int')
    for i in range(map_width):
        for h in range(map_height):
            for _, j in enumerate(vels):
                for _, k in enumerate(vels):
                    states = np.vstack((states, (i, h, j, k)))
    return states

def generate_actions_vector(actions):
    return np.array(np.meshgrid(actions, actions)).T.reshape(-1, 2)
