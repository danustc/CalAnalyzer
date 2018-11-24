'''
spatial analysis of clusters
Created by Dan on 09/25/2018
'''
import numpy as np

from scipy.ndimage.filters import gaussian_filter


def coord_cluster(coords, cind_label):
    '''
    group coordinates into clusters based on labeling.
    '''
    NL = len(cind_label)
    cc = []
    for ci in cind_label:
        cc.append(coords[ci])

    return cc


def mutual_information_cloud(cloud_a, cloud_b):
    '''
    calculate mutual information between two clouds.
    Needs to design this algorithm carefully.
    '''



def dist_pointcloud(coords, x_range, y_range, dbin = 0.30, smooth_sig = 1.0 ):
    '''
    coords: 2-column array, ordered in x-y.
    nr: bin width
    '''
    # evaluate the distribution 
    xbin = int(x_range//dbin)
    ybin = int(y_range//dbin)
    sig = smooth_sig/dbin # the sigma in the unit of dbin
    sig_2d = [sig, sig]
    H, y_edges, x_edges = np.histogram2d(coords[:,1], coords[:,0], bins = (ybin, xbin), density = False) # in H, x-axis is the rows and y-axis is the columns
    SH = gaussian_filter(H, sig_2d, mode = 'constant') # smooth.
    norm_fact = SH.sum()*dbin*dbin # normalize with the integral
    SH = SH/norm_fact

    return SH



def scatter2vox(coords, activities, vox_size, r_range ):
    '''
    convert coordinates of scatter points into voxels and calculate mean intensities.
    coords: coordinates ordered in x,y,z.
    vox_size: dx, dy, dz
    '''
    xx, yy, zz = coords[:,0], coords[:,1], coords[:,2]
    dx, dy, dz = vox_size
    RX, RY, RZ = r_range

    NC = len(coords) # the number of neurons

    NX = int(RX//dx)
    NY = int(RY//dy)
    NZ = int(RZ//dz)

    lx = np.arange(NX+1)*dx
    ly = np.arange(NY+1)*dy
    lz = np.arange(NZ+1)*dz

    IX = np.searchsorted(lx[:-1], xx)
    IY = np.searchsorted(ly[:-1], yy)
    IZ = np.searchsorted(lz[:-1], zz)

    act_cube = np.zeros((NZ+1, NY+1, NX+1))
    count_cube = np.ones((NZ+1, NY+1, NX+1)) + 1.0e-06

    for cc in range(NC):
        ind_x, ind_y, ind_z = IX[cc], IY[cc], IZ[cc]
        act_cube[ind_z, ind_y, ind_x] +=activities[cc]
        count_cube[ind_z, ind_y, ind_x] +=1

    act_cube = act_cube/count_cube
    count_cube = count_cube.astype('int')
    return act_cube, count_cube

        # also

