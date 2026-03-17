import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

def config(seed = 0): 

    # setting random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # setting plotting style
    plt.rc('text', usetex=True)
    font = {
        'family' : 'serif',
        # 'weight' : 'normal',
        # 'size'   : 16
    }
    plt.rc('font', **font)
    # mpl.rcParams['axes.linewidth'] = 2