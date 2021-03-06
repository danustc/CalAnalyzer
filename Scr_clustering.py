'''
Sample script: clustering of a large group of neurons
'''
import numpy as np
import os
from calanalyzer.analysis.single_analysis import grinder
import calanalyzer.analysis.spectral_clustering as sc
import matplotlib.pyplot as plt
from calanalyzer.analysis.hierachical_sc import hrc_sc
from calanalyzer.visualization.signal_plot import compact_dffplot, dff_rasterplot
from calanalyzer.visualization.cluster_navigation import multi_cluster_show
import tifffile as tf
from calanalyzer.analysis.spatial import coord_cluster
import glob
import scipy as sp
from sklearn.neighbors import BallTree

global_datapath_ubn = '/home/sillycat/Programming/Python/data_test/FB_resting_15min/'
portable_datapath = '/media/sillycat/DanData/'


def scratch():
    data_path = glob.glob(global_datapath_ubn + 'Jul2017/Jul19_2017_B3_ref_lb_cleaned.npz')
    dff_data = np.load(data_path[0])
    signal_test = dff_data['signal'][10:,:4000]


    ccc = sc.Corr_sc()
    ccc.load_data(signal_test)


    ccc.link_evaluate(sca = 1.04)
    ccc.affinity(mode = 'cut', NK = 100)
    pk, fig_eig = ccc.laplacian_evaluation()
    fig_eig.savefig('eigen')
    affi_mat = ccc.affi_mat
    hist, be = sc.corr_distribution(affi_mat)
    L, D = sc.laplacian(affi_mat, mode = 'un')
    w,v = sc.sc_eigen(L, DM = D)
    NR, NV = v.shape
    print(NR, NV)
    plt.close('all')
    plt.plot(v[:,1:5])
    plt.show()



def main():
    data_list = glob.glob(global_datapath_ubn + 'Jul2017/*ref_lb_cleaned.npz')
    vol_img = tf.imread(global_datapath_ubn+'MAX_Aug23_B4_ref.tif')
    for data_path in data_list:
        #data_path = global_datapath_ubn + 'Aug23_2018_B4_ref_cleaned.npz'
        fish_flag = '_'.join(os.path.basename(data_path).split('_')[:3])
        print("Fish:", fish_flag)
        dff_data = np.load(data_path)
        signal_test = dff_data['signal'][10:,:10000]
        ccc = sc.Corr_sc()
        ccc.load_data(signal_test)
        ccc.link_evaluate(sca = 1.05)
        ccc.affinity(mode = 'cut', NK = 0, Niter = 3)
        pk, fig_eig = ccc.laplacian_evaluation()

        fig_eig.savefig(fish_flag + '_eigen')
        plt.close(fig_eig)

        ccc.clustering(n_clu = 13)
        fig_ccc = compact_dffplot(ccc.cl_average, fsize = (6,6))
        fig_ccc.savefig('direct_' + fish_flag)
        plt.close(fig_ccc)

        direct_pop = [len(ig) for ig in ccc.ind_groups]
        for ii in range(12):
            ci = ccc.ind_groups[ii]
            cluster_signal = signal_test[:, ci]
            fig_raster = dff_rasterplot(cluster_signal)
            #fig_raster.savefig(fish_flag + 'straight_'+str(ii))
            plt.close(fig_raster)
        coord_test = dff_data['raw_coord']
        cc = coord_cluster(coord_test, ccc.ind_groups)
        fig_coord = multi_cluster_show(cc[:12], vol_img, cm = 'viridis', layout = (4,3), sup_title = fish_flag, fsize = (7.5, 8.2))
        fig_coord.savefig('cldirect_' + fish_flag)
        plt.close(fig_coord)

# ----------------------------Below is the divide-and-conquer treatment ------
        HS_class = hrc_sc(signal_test, n_group = 10)
        print(HS_class.__dict__.keys())
        HS_class.divide_sc(mode = 'random')
        HS_class.groupwise_population_labeling()
        cluster_cg = HS_class.cluster_corrcheck()
        merged_label, cl_average, cind = HS_class.merge_clusters(cluster_cg)
        NT, NC = cl_average.shape
        cc = coord_cluster(coord_test, cind)

        for ii in range(NC):
            ci = cind[ii]
            cluster_signal = signal_test[:, ci]
            fig_raster = dff_rasterplot(cluster_signal)
            fig_raster.savefig('dq_'+fish_flag+ '_' + str(ii))
            plt.close(fig_raster)

        dq_pop = [len(ig) for ig in cind]

        fig_merged = compact_dffplot(cl_average, fsize = (6,6))
        fig_merged.savefig('mg_'+ fish_flag)

        plt.close(fig_merged)

        fig_coord = multi_cluster_show(cc[:12], vol_img, cm = 'viridis', layout = (4,3), sup_title = fish_flag, fsize = (7.5, 8.2))
        fig_coord.savefig('cldq' + fish_flag)
        plt.close(fig_coord)


if __name__ == '__main__':
    #scratch()
    main()
