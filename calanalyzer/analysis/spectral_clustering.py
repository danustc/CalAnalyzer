'''
My practice of spectral clustering, written from scratch
'''
import numpy as np
import scipy.sparse.linalg as linalg
from scipy import stats
from scipy.signal import argrelextrema
from scipy.spatial.distance import correlation
from sklearn.cluster import KMeans, SpectralClustering
import matplotlib.pyplot as plt
from collections import deque

def knn_dumbest(dist_mat, NK):
    NC = dist_mat.shape[0]

    query = np.zeros([NC, NK])
    for cc in range(NC):
        ind = np.argsort(dist_mat[cc])
        query[cc] = ind[:NK]

    return query.astype('uint16')



def symmetric(mat, tol = 1.0e-08):
    return np.allclose(mat, mat.T, atol = tol)

# --------------------Below are key functions for spectral clustering steps -------------------


def corr_to_similarity(cm, mode = 'Gauss', gamma = 1.):
    '''
    Map correlation coefficient to distance.
    '''
    dist = np.tan(0.5*np.pi*(1. - cm)) # the distance matrix
    #dist = -order*np.log(cm) # mapping (0,1) to (inf, 0)
    if mode == 'Gauss':
        sim = np.exp(-gamma * dist**2)

    return sim



def laplacian(W, mode = 'un'):
    '''
    construct graph laplacian from the weight matrix
    there are three modes:
        un: unnormalized
        sym: symmetrically normalized
        rw: random walk-type, normalized
    '''
    D_sum = W.sum(axis = 1) # diagonal matrix
    D = np.diag(D_sum)
    L = D - W
    D_isq = np.diag(1./(D_sum**0.5))
    if mode =='sym':
        L_sym = np.dot(D_isq, L).dot(D_isq)
        return L_sym, D
    elif mode == 'rw':
        # warning: the eigen value of L_rw is not all non-negative. I need to double check the tutorial.
        L_rw = np.diag(1./D_sum).dot(L)
        return L_rw, D
    else:
        return L, D


def sc_eigen(L, n_cluster = 20, DM = None):
    '''
    compute the embedding space of L and cluster. L can be unnormalized or normalized.
    '''
    w, v = linalg.eigsh(L, k = n_cluster, M = DM, sigma = 0, which = 'LM') # compute the 1st n_cluster smallest eigenvalues and eigenvectors. Shift-invert mode.
    print("representing eigenvalues:", w[:n_cluster])

    return w, v




def weakest_connection(corr_mat):
    '''
    check if any individuals are disconnected with anybody else.
    In addition to the weakest link, also compute a general trend of the link strength among the group.
    '''
    nd_mat = corr_mat - np.diag(np.diag(corr_mat))# kill the diagonal elements
    nd_mat[nd_mat < 0] = 0.
    max_corr = nd_mat.max(axis = 0) # the strongest correlation of each cell to everybody else
    sorted_mc = sorted(max_corr) # sort the strongest correlation
    diff_smc = np.diff(sorted_mc) # differential link strength
    peak = argrelextrema(diff_smc, np.greater)[0] # find peaks in the sorted sorted mc.
    if len(peak) ==0:
        weak_link = sorted_mc[0]
    else:
        weak_link = sorted_mc[:peak[0]+1] # give a list of cut-off options

    return weak_link, peak



def corr_distribution(corr_mat, nb = 200, uprange = 0.5):
    # set the diagonal to zero first
    corr_mat_sd = corr_mat - np.diag(2*np.diag(corr_mat))
    hist, be = np.histogram(corr_mat_sd.ravel(),  bins = nb, density = True, range = (0, uprange))
    return hist, be


def leigen_nclusters(eigen_list, pad = 1.0e-06):
    '''
    determining how many clusters the eigenvalue array suggests by finding the kinks in the eigen value list.
    '''
    eig_diff = np.diff(eigen_list)

    relative_diff = eig_diff/(eigen_list[:-1]+pad)
    peak = np.argmax(relative_diff) + 1
    #peaks = argrelextrema(relative_diff, np.greater)[0]+1 # find kinks

    return peak


def label_assignment(raw_data, n_cl, y_labels, verbose = True):
    '''
    raw_data: NT x NC, NT: # of trials, NC: # of cells
    After spectral clustering, calculate the population and average of each group.
    Have to double check this function.
    '''
    # Create a distant matrix
    NT, NC = raw_data.shape
    total_ind = np.arange(NC)
    ind_groups = []
    g_population = np.zeros(n_cl)

    for ii in range(n_cl):
        ind_clu = total_ind[y_labels == ii]
        ind_groups.append(ind_clu)
        g_population[ii] = len(ind_clu)

    print("cluster population:", g_population)
    sort_pop = np.argsort(g_population).astype('int')
    ind_groups = [ind_groups[sort_pop[ii]] for ii in range(n_cl)] # ok, this reorders groups based on population numbers

    cl_average = np.zeros([NT, n_cl])
    for ii, idg in zip(range(n_cl), ind_groups):
        n_pop = len(idg)
        print("population of group", ii, ":", n_pop)
        # add a check point: if this cluster has one member only, then do not do the average
        if n_pop > 1:
            cl_average[:,ii] = raw_data[:,idg].mean(axis = 1)
        else:
            cl_average[:,ii] = raw_data[:,idg[0]]

    return ind_groups,  cl_average


# --------------------Class of spectral clustering -----

class Corr_sc(object):
    '''
    spectral clustering based on correlation coefficient.
    '''
    def __init__(self, raw_data = None):
        '''
        Initialize the class, compute the correlation matrix.
        '''
        if raw_data is not None:
            self.load_data(raw_data)
        else:
            print("No data loaded.")


    def link_evaluate(self, histo = False, sca = 1.150):
        '''
        Evaluate how densely/intensely this graph is linked
        '''
        wk_link, peak = weakest_connection(self.corr_mat)
        if histo:
            hi, be = corr_distribution(self.corr_mat)
        if np.isscalar(wk_link):
            self.thresh = sca*wk_link
        else:
            self.thresh = sca*wk_link[-1]
        print("The threshold:", self.thresh)

    def load_data(self, new_data, neg_rect = 1.0e-09):
        self.data = new_data
        self.NT, self.NC = new_data.shape
        self.corr_mat = np.corrcoef(new_data.T)
        if neg_rect is not None:
            self.corr_mat[self.corr_mat < 0] = neg_rect

        #self.link_evaluate() # Do I need this extra link_evaluate?


    def affinity(self, thresh = None, mode = 'adapt', NK = 0, Niter = 3):
        '''
        calculate affinity matrix.
        mode == 'linear': just use the correlation matrix as the affinity, cut off at the threshold
        mode == 'Gauss': map (-1, 1) to (inf, 0) first, then calculate affinity using gaussian function.
        '''
        if thresh is None:
            try:
                thresh = self.thresh
            except AttributeError:
                print("The default threshold is not set.")
                return
        if mode == 'cut':
            affi_mat = np.copy(self.corr_mat)
            affi_mat[affi_mat < thresh] = 1.0e-09

        elif mode == 'adapt':
            '''
            set the threshold for each neuron.
            '''
            act_mat = np.ones((self.NC, self.NC))*1.0e-09
            for cc in range(self.NC):
                '''
                fit the histogram to Gaussian, cut of at sigma
                '''
                corr_trace = self.corr_mat[cc]
                m,s = stats.norm.fit(corr_trace[corr_trace < 0.6])
                for ii in range(Niter):
                    m,s = stats.norm.fit(corr_trace[corr_trace < m+3*s])

                print("cutoff:", m+2*s)
                ind = corr_trace > (m+3*s)

                act_mat[cc, ind] = 1.
                act_mat[ind, cc] = 1.

            affi_mat = self.corr_mat*act_mat
        if NK>0: # perform K-nearest neighbor
            act_mat = np.zeros((self.NC, self.NC))
            dist_mat = 1.-self.corr_mat
            knn_query = knn_dumbest(dist_mat, NK)

            for kk in range(1,NK):
                act_mat[knn_query[:,0],knn_query[:,kk]] = 1.
                act_mat[knn_query[:,kk],knn_query[:,0]] = 1.

            affi_mat = affi_mat*act_mat

        if not(symmetric(affi_mat)):
            affi_mat = (affi_mat+affi_mat.T)*0.5

        self.affi_mat = affi_mat


    def laplacian_evaluation(self, plotout = True, ncl = 25):

        L, D = laplacian(self.affi_mat, mode = 'un') # use the random-walk normalized Laplacian instead of unnormalized version.   
        w, v = sc_eigen(L, n_cluster = ncl, DM = D) # calculate the first 20th eigen values and eigen states
        peak_position = leigen_nclusters(w) # where should I cut off?
        if plotout:
            fig_plot = plt.figure(figsize = (6,3))
            ax = fig_plot.add_subplot(111)
            ax.plot(np.arange(1, ncl+1), w, '-x')
            ax.scatter(peak_position, w[peak_position-1], s = 150, facecolors = 'none', edgecolors = 'orange', linewidth = 2)
            txt = 'Zero eigen-values: '+str(peak_position)
            ax.text(0, w.mean(),txt, fontsize = 13)
            #fig_plot.show()
        else:
            fig_plot = None

        return peak_position, fig_plot


    def clustering(self, n_clu = 5):
        SC = SpectralClustering(n_clusters = n_clu, affinity = 'precomputed', n_init = 20)
        y_labels = SC.fit_predict(self.affi_mat)
        self.ind_groups, self.cl_average = label_assignment(self.data, n_clu, y_labels)



    def clearup(self):
        '''
        clear up all the contents and keep the class shell only.
        There might be more elegant ways to do it, but I will start crude.
        '''
        self.data = None
        self.corr_mat = None
        self.affi_mat = None
        self.ind_groups = None
        self.cl_average = None
