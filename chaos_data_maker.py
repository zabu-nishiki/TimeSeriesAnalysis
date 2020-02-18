#!/usr/bin/python
# -*- Coding: utf-8 -*-
import time
import numpy as np
from utils import StrangeAttractor

if __name__=='__main__':
    '''
    ——————————————————————————————
    Lorenz:
    dt' = 0.1000
    dt  = 0.0001
    dt  = 0.0100(デフォルト値)
    Chen:
    dt' = 0.0500
    dt  = 0.0001
    dt  = 0.0100(デフォルト値)
    Rossler:
    dt' = 2.0000
    dt  = 0.0020
    dt  = 0.0100(デフォルト値)
    ——————————————————————————————
    '''
    date_ini = time.time()
    dt  = 0.0001
    size_filter = 500
    sample_size = 80000
    data = np.zeros((sample_size, 3))

    state_ini_lorenz = np.array([2.5, 2.5, 25])
    state_ini_chen = np.array([-0.1, 0.5, -0.6])
    state_ini_rossler = np.array([2.5, 2.5, 2.5])
    #current_state = state_ini_lorenz
    current_state = state_ini_chen
    #current_state = state_ini_rossler

    #function_name = StrangeAttractor.LorenzAttractor
    function_name = StrangeAttractor.ChenAttractor
    #function_name = StrangeAttractor.RosslerAttractor
    
    for idx in range(size_filter*10000):
        next_state = StrangeAttractor.RungeKutta(
            function_name , current_state, dt)
        current_state = next_state
    print('The initialize is done.')

    idx_1 = 0
    for idx_0 in range(size_filter*sample_size):
        next_state = StrangeAttractor.RungeKutta(
            function_name , current_state, dt)
        current_state = next_state
        if idx_0 % size_filter < 1e-6:
            data[idx_1, :] = current_state
            idx_1 += 1
            if idx_1 % 8000 < 1e-6:
                print('The iteration: ' + str(idx_1//8000) + ' / 10')

    print('The number of the samples: ' + str(len(data[:, 0])))

    date_end = time.time()
    print('The collapsed time: ' + str((date_end - date_ini)/60) + ' min.' )
    print('End the calculation.')
    #np.save('data/lorenz/xyz_lorenz',data)
    np.save('data/chen/xyz_chen',data)
    #np.save('data/rossler/xyz_rossler',data)