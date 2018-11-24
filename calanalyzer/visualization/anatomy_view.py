'''
Visualization tool of anatomical labeling statistics.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns; sns.set()


def activity_view(coords, activity, cm = 'viridis', fsize = (6,5)):
    xx, yy, zz = coords[:,0], coords[:,1], coords[:,2]
    fig_3d = plt.figure(figsize = fsize)
    ax = fig_3d.add_subplot(111, projection = '3d')
    ax.scatter(xx,yy,zz, c = activity, cmap = cm, s = 4)
    ax.set_xlabel('X', fontsize = 12)
    ax.set_ylabel('Y', fontsize = 12)
    ax.set_zlabel('Z', fontsize = 12)

    return fig_3d



def label_scatter(mask_counts, color = 'coral'):
    '''
    mask_counts: the count of each mask
    '''
    ax = sns.boxplot(data = mask_counts, orient = 'h')
    #ax.set_xticklabels(ax.get_xticklabels(), rotation = 45)

    return ax


# -----------Below is a dynamic 3D viewer. 

def stack_viewer(volume):
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] //2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)

    fig.canvas.draw()
#label_scatter()


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index-1)
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index+1)
    ax.images[0].set_array(volume[ax.index])


def animation(volume, write = False, fpath = 'test.mp4'):
    NC = volume.shape[0]
    fig, ax = plt.subplots()
    ax = fig.axes[0]
    ax.imshow(volume[0])
    ims = []

    if write:
        FMW = manimation.writers['ffmpeg']
        writer = FMW(fps = 2)
        with writer.saving(fig, fpath, 100):
            for ii in range(NC):
                ax.images[0].set_array(volume[ii])
                writer.grab_frame()

    else:
        for ii in range(NC):
            im = plt.imshow(volume[ii], animated = True, title = fpath+ str(ii))
            plt.tight_layout()
            fig.savefig(fpath + str(ii))
            ims.append([im])


        ani = manimation.ArtistAnimation(fig, ims, interval = 100, blit = True, repeat_delay = 1000)
        plt.show()
