# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 18:10:50 2022

@author: Stefano
"""
from network_functions import *
from network_theory import *

import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import cumfreq
from tqdm import tqdm
  

def load_logbin (m = 1, N = 10000, mode = "PA", scale = 1., 
                 density = True, truecount = False, pad = True):
    """
    The function imports the klist saved as f'{mode}//avklist_m{m}_N{N}.npy', 
    then logbins the data inside. 
    
    Parameters
    ----------
    First, the parameters with which data was simulated:
        mode : "PA" or "RA" or "PR" or "RP"
        m : int
        N : int
    
    scale : float
        Scaling factor. Default is 1., i.e. normal binning.
    
    density : Bool
        If True (default), normalise binning.
    pad : Bool
        If True (default), keep zeros. 

    Returns
    -------
    binned data: ndarray
        data per bin average over realisations.
    binned uncertainty: ndarray
        std of results averaged together.
    bin_centres : ndarray
    """
    klist = np.load(f"{mode}/avklist_m{m}_N{N}.npy")
    return multi_logbin (klist, scale, 
                         density = density, truecount = truecount, pad = pad)
   
        
def multi_logbin (data, scale = 1., start = "min", density = True, 
                  truecount = False, pad = False):
    """
    Logbinning of dataset with width of bins scaling as scale ** j. 
    Parameters
    ----------
    data : ndarray
        Data to be logbinned.
    scale : float
        Scaling factor. Default is 1., i.e. normal binning.
    density : Bool
        If True (default), normalise binning.
    pad : Bool
        If True (default), keep zeros. 

    Returns
    -------
    binned data: ndarray
        data per bin average over realisations.
    binned uncertainty: ndarray
        std of results averaged together.
    bin_centres : ndarray

    """
    Ds = data.ndim
    if Ds > 2:
        raise Exception(f"The data has too many dimensions ({Ds})")
    elif Ds == 2:
        L = 0
        vals = []
        lengths = []
        for data_i in data:
            cent_i, vals_i = logbin(data_i, scale, zeros = True, 
                                   density = density, truecount = truecount, 
                                   pad = pad)
            len_i = len(cent_i)
            vals.append(vals_i)
            lengths.append(len_i)
            if len_i > L:
                L = len_i * 1
                centres = cent_i * 1
        for i in range(len(data)):
            #add zeros to the end if the array is shorter
            vals[i] = np.pad(vals[i], (0, L-lengths[i]), constant_values = 0)
            #compute average
        hist = np.mean(vals, axis = 0)
        stds = np.std(vals, axis = 0)

    return hist, stds, centres


def zipf_f(klist):
    for i in range(len(klist)):
        klist[i] = np.sort(klist[i])[::-1]
    ranked_avg = np.mean(klist, axis = 0)
    ranked_std = np.std(klist, axis = 0)
    r = np.arange(1, len(ranked_avg)+1)
    return ranked_avg, ranked_std, r
    
def multi_cumfreq(klist, ks):
    kmin = min(ks)
    kmax_long = ks[-1] -1
    cumobserved = []
    for i in range(len(klist)):
        cumobserved.append(cumfreq(klist[i], len(ks), (kmin, kmax_long))[0])
    cdfk = np.mean(cumobserved, axis = 0)
    cdfk_std = np.std(cumobserved, axis = 0)
    cdfk_std = cdfk_std / max(cdfk)
    cdfk = cdfk / max(cdfk)
    return cdfk, cdfk_std






#%%############################################################################
# # Modified from:
# # Max Falkenberg McGillivray
# # mff113@ic.ac.uk
# # 2019 Complexity & Networks course
# #
# # logbin230119.py v2.0
# # 23/01/2019
# # Email me if you find any bugs!
# #
# # For details on data binning see Appendix E from
# # K. Christensen and N.R. Moloney, Complexity and Criticality,
# # Imperial College Press (2005).
# #############################################################################

def logbin(data, scale = 1., zeros = False, density = True, truecount = False,
           pad = False, edges = False, cutmin = True):
    """
    Max Falkenberg McGillivray. Complexity & Network course, 2019. 
    mff113@ic.ac.uk
    
    logbin(data, scale = 1., zeros = False)

    Log-bin frequency of unique integer values in data. Returns probabilities
    for each bin.

    Array, data, is a 1-d array containing full set of event sizes for a
    given process in no particular order. For instance, in the Oslo Model
    the array may contain the avalanche size recorded at each time step. For
    a complex network, the array may contain the degree of each node in the
    network. The logbin function finds the frequency of each unique value in
    the data array. The function then bins these frequencies in logarithmically
    increasing bin sizes controlled by the scale parameter.

    Minimum binsize is always 1. Bin edges are lowered to nearest integer. Bins
    are always unique, i.e. two different float bin edges corresponding to the
    same integer interval will not be included twice. Note, rounding to integer
    values results in noise at small event sizes.

    Parameters
    ----------

    data: array_like, 1 dimensional, non-negative integers
          Input array. (e.g. Raw avalanche size data in Oslo model.)

    scale: float, greater or equal to 1.
          Scale parameter controlling the growth of bin sizes.
          If scale = 1., function will return frequency of each unique integer
          value in data with no binning.

    zeros: boolean
          Set zeros = True if you want binning function to consider events of
          size 0.
          Note that output cannot be plotted on log-log scale if data contains
          zeros. If zeros = False, events of size 0 will be removed from data.

    Returns
    -------

    x: array_like, 1 dimensional
          Array of coordinates for bin centres calculated using geometric mean
          of bin edges. Bins with a count of 0 will not be returned.
    y: array_like, 1 dimensional
          Array of normalised frequency counts within each bin. Bins with a
          count of 0 will not be returned.
    z : edges
    """
    if scale < 1:
        raise ValueError('Function requires scale >= 1.')
    count = np.bincount(data)
    tot = np.sum(count)
    smax = np.max(data)
    if scale > 1:
        jmax = np.ceil(np.log(smax)/np.log(scale))
        if zeros:
            binedges = scale ** np.arange(jmax + 1)
            binedges[0] = 0
        else:
            binedges = scale ** np.arange(1,jmax + 1)
            # count = count[1:]
        binedges = np.unique(binedges.astype('uint64'))
        x = (binedges[:-1] * (binedges[1:]-1)) ** 0.5
        y = np.zeros_like(x)
        count = count.astype('float')
        #print(len(y), len(binedges))
        #if density:
        for i in range(len(y)):
            y[i] = np.sum(
                count[binedges[i]:binedges[i+1]])
            if y[i] != 0:
                if binedges[i] < np.amin(data) and cutmin:
                    binedges[i] = int(np.amin(data))
            if not truecount:
                y[i]/=(binedges[i+1] - binedges[i])
        # elif density == False:
        #     for i in range(len(y)):
        #         y[i] = np.sum(count[binedges[i]:binedges[i+1]])
            # print(binedges[i],binedges[i+1])
        # print(smax,jmax,binedges,x)
        # print(x,y)
        #print(len(y), len(binedges))
    else:
        x = np.nonzero(count)[0]
        y = count[count != 0].astype('float')
        if zeros != True and x[0] == 0:
            x = x[1:]
            y = y[1:]
            
    if density and not truecount:
        y /= tot
        
    if not pad:
        x = x[y!=0]
        y = y[y!=0]
    else:
        while y[0] == 0:
            x = x[1:]
            y = y[1:]
            binedges = binedges[1:]
            
    if edges:
        return x,y,binedges
    return x, y