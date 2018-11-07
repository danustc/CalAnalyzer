'''
Visualization tool of anatomical labeling statistics.
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns; sns.set()


def activity_view(coords, activity, cm = 'viridis', fsize = (6,5)):
    xx, yy, zz = coords[:,0], coords[:,1], coords[:,2]
    fig_3d = plt.figure(figsize = fsize)
    ax = fig_3d.add_subplot(111, projection = '3d')
    ax.scatter(xx,yy,zz, c = activity, cmap = cm)
    ax.set_xlabel('X', fontsize = 12)
    ax.set_ylabel('Y', fontsize = 12)
    ax.set_zlabel('Z', fontsize = 12)

    return fig_3d


def mask_view(coords, mask_labels, mask_colors, fsize = (6,5)):
    '''
    Have a mask view 
    '''





def label_scatter(mask_counts, color = 'coral'):
    '''
    mask_counts: the count of each mask
    '''
    ax = sns.boxplot(data = mask_counts, orient = 'h')
    #ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)

    return ax


#label_scatter()

