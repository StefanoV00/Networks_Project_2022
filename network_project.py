# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:59:46 2022

@author: Stefano
"""
from network_functions import *
from network_analysis_functions import *
from network_theory import *
from network_plotf import *
from network_statistics import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from tqdm import tqdm
from timeit import default_timer as timer 
import networkx as nx

from scipy.stats import cumfreq, chisquare, kstest, ks_2samp
from scipy.stats import anderson_ksamp , cramervonmises, norm
from scipy.optimize import curve_fit, NonlinearConstraint
from iminuit import Minuit

from os.path import exists
from uncertainties import ufloat

ps = {"text.usetex": True,
      "font.size" : 16,
      "font.family" : "Times New Roman",
      "axes.labelsize": 15,
      "legend.fontsize": 13,
      "xtick.labelsize": 13,
      "ytick.labelsize": 13,
      "figure.figsize": [7.5, 6],
      "mathtext.default": "default"
       }
pl.rcParams.update(ps)
del ps

#%% LOAD AND OBSERVE
m = 1
N = int(1e6)
mode = "PAalt"
a = np.load(f"{mode}/avklist_m{m}_N{N}.npy")

#%%LOAD, BIN, PLOT
m = [1, 10, 100]#[::-1]
#m = 10
#N = [int(Ti) for Ti in [1e4, 5e4, 1e5, 5e5, 1e6]]
#m = 10
N = int(1e6)
mode = "PA"
scale = 1.1
mscale, theory, wrongtheory, collapse = (0, 1, 0, 0.)
logbinned, cumulative, zipf = (1, 0, 0)

a = k_plot(m = m, N = N, mode = mode, scale = scale, 
       mscale = mscale, theory = theory, wrongtheory = wrongtheory,
       collapse = collapse, 
       logbinned = logbinned, cumulative = cumulative, zipf = zipf)

del m, N, mode, scale
del mscale, theory, wrongtheory
del logbinned, cumulative, zipf

#%% STATISTICS
m = 100
N = int(1e6)
mode = "RPmulti"
scale1 = 1.05
scale2 = 1.05
wrongtheory = 0
n1, n2 = np.array([1,1]) * 6000
#n1, n2 = 0.75, 0.40
a = statistics(m, N, mode, scale1, scale2, logcut = n1, cumcut = n2, 
           wrongtheory = wrongtheory, avg = False, chi2 = False, KSall = 1)

del m, N, mode
del scale1, scale2, wrongtheory, n1, n2

#%% LARGEST k
m = 100
Tlist = [1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5, 1e6]
mode = "PA"
theory = [1, 4]
(a, b), (ast, bstd) = plot_k1_N(m, Tlist, mode, theory, dont = False, fit = False)
del theory

#%% FIT LARGE K
mlist = [1, 10, 50, 100]
Tlist = [1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5]#, 1e6]
mode = "PA"
k1_fit(mlist, Tlist, mode)
del mlist, Tlist, mode

#%% LARGEST k
m = 2
Tlist = [1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5, 1e6]
mode = "RPmulti"
theory = [1, 2, 3, 4]
(a, b), (ast, bstd) = plot_k1_N(m, Tlist, mode, theory, dont = False, fit = True)
del theory
#%%
pl.ylim(600, 1700)
order = [3, 0, 1, 2]
handles, labels = pl.gca().get_legend_handles_labels()
pl.legend([handles[i] for i in order], [labels[i] for i in order], ncol = 2)
# mode = "PAalt"
# theory = [1]
# (a, b), (ast, bstd) = plot_k1_N(m, Tlist, mode, theory, dont = False, fit = False)
# del m, Tlist, mode, theory
# #%%
# L=pl.legend()
# labela = "Model (a): "+L.get_texts()[0]._text
# labelb = "Model (b): "+L.get_texts()[2]._text
# dataa  = "Model (a): "+L.get_texts()[3]._text
# dataa  = "Model (a): "+L.get_texts()[3]._text

# L.get_texts()[0].set_text("Model (a): ")
# labelb = L.get_texts()[1]._text
# labelc = L.get_texts()[2]._text
#%%
mlist = [1, 10, 25, 50, 100]
N = int(1e6)
mode = "PA"
theory = [1,2,3]
(a, b), (ast, bstd) = plot_k1_m(mlist, N, mode, theory)
del mlist, N, mode, theory



#%%COLLAPSE
m = [1, 10, 25, 50, 100]#[::-1]
m = 25
N = [int(Ti) for Ti in [1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5, 1e6]]
#m = 10
#N = int(1e6)
mode = "PA"
scale = 1.1
mscale, theory, wrongtheory, collapse = (1, 1, 0, 0.497)
logbinned, cumulative, zipf = (1, 0, 0)

a = k_plot(m = m, N = N, mode = mode, scale = scale, 
       mscale = mscale, theory = theory, wrongtheory = wrongtheory,
       collapse = collapse, 
       logbinned = logbinned, cumulative = cumulative, zipf = zipf)

del m, N, mode, scale
del mscale, theory, wrongtheory
del logbinned, cumulative, zipf

#%%
L=pl.legend(loc = "upper left", ncol = 2, fancybox = True)
#%%
# #%% QUICK CHECKS
# m = 100
# N = int(1e5)
# mode = "PA"
# klists = np.load(f"{mode}/avklist_m{m}_T{N}.npy")
# kmax = np.amax(klists)

# m = 25
# N = int(1e6)
# mode = "PA"
# scale1 = 1.1
# scale2 = 1.05
# wrongtheory = 1

# N = N + 2*m + 1
# klists = np.load(f"{mode}/avklist_m{m}_T{N}.npy")
# klists_flat = klists.flatten()
# kbinned, kstd, kcentres = load_logbin(m , N, mode, scale, truecount = True,
#                                       pad = True)
# [pk, pkstd, pk_centres], [cdfk, cdfk_std, ks] = k_plot(m , N, mode, scale1, 
#                                                   wrongtheory = wrongtheory, 
#                                                   logbinned = True, 
#                                                   cumulative = True)
# pk_centres_flat, pk_flat = logbin(klists_flat, scale = scale2, truecount = True)
# if mode == "PA":
#     theory1 = PA_pk1(kcentres, m)
#     theory2 = PA_pk2(kcentres, m)
#     cumtheory1 = PA_cdfk1(ks, m)
#     cumtheory2 = PA_cdfk2(ks, m)
    
#     # FOR KS TEST
#     D1 = max(cdfk - cumtheory1)
#     D2 = max(cdfk - cumtheory2)
# if mode == "RA":
#     theory = RA_pk(kcentres, m)
#     cumtheory = RA_cdfk(ks, m)
    
#     #FOR KS TEST
#     D = max(cdfk - cumtheory)



