'''
activity analysis
'''

import numpy as np


def act_dist(a_arr, nb = 200):
    '''
    activity distribution
    '''
    hist, be = np.histogram(a_arr, bins = nb)
    
    XX,YY = np.meshgrid(a_arr, a_arr)
    dist = np.abs(XX-YY)
    
