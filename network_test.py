# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 20:41:55 2022

@author: Stefano
"""
from network_theory import *
from network_functions import *
import networkx as nx

import numpy as np
import random as rand
import matplotlib.pyplot as pl
from tqdm import tqdm
from timeit import default_timer as timer 
import networkx as nx

from scipy.stats import chisquare
from scipy.optimize import curve_fit, NonlinearConstraint
from iminuit import Minuit

from numba import njit
from uncertainties import ufloat

ps = {"text.usetex": True,
      "font.size" : 17,
      "font.family" : "Times New Roman",
      "axes.labelsize": 20,
      "legend.fontsize": 15,
      "xtick.labelsize": 15,
      "ytick.labelsize": 15,
      "figure.figsize": [7.5, 6],
      "mathtext.default": "default"
       }
pl.rcParams.update(ps)
del ps

#%% GET A TASTE
m = 5
mode = "PAalt"
G = BAnetwork(m, mode = mode)
G.grow(1, m=m, mode =mode)
G.draw()

#%% TEST SPEED
m = 1
T = int(1e6)
G = BAnetwork(m, mode = "auto")
start = timer()
G.grow(T = T, m = m, mode = "PA", r = 0, seed = 0)
end = timer()
print(end-start); del end, start
if not G.check():
    print(f"Realisation check failed")
G.draw()
    
#%% TEST PA-ASSIGN
# Initialise and randomise the Graph
m = 50
T = 200 - 2*m -1
G = BAnetwork(m)
G.grow(T)
original = G.get_klist() *1
# Test
T = 100000
for i in range(T):
    G.PAassign_static(m)
obtained = G.get_klist()

x = np.arange(0, len(original))
kmin_orig = min(original);  kav_orig = np.mean(original)
kmin = min(obtained);       kav = np.mean(obtained)
ktot_orig = sum(original); ktot = sum(obtained)


# Plot results of test
pl.figure(tight_layout = True)
x = np.arange(0, len(original))
ktot_orig = sum(original)
pl.step(x, original / ktot_orig, label = "Original", c="b")
ktot = sum(obtained)
pl.step(x, obtained / ktot, label = "Final", c="darkred")
pl.legend()
pl.ylim([ pl.ylim()[0], pl.ylim()[1] * 1.1])
props = dict(boxstyle='round', facecolor = 'white', edgecolor = "r", ls = '--', 
             alpha=0.5)
pl.annotate (f"m = {m}", (0.05, 0.92), xycoords = "axes fraction",  
              va='bottom', ha = 'left',bbox = props) 
pl.ylabel("$k_i$ (normalised)")
pl.xlabel("Nodes")
pl.savefig(f"Testing/PAassign_m{m}")
pl.show()

del m, T, G, i
del x, original, obtained, ktot_orig, ktot

#%% TEST RA-ASSIGN
# Initialise and randomise the Graph
m = 1
T = 200 - 2*m-1
G = BAnetwork(m)
G.grow(T)
original = G.get_klist() *1
# Test
T = 100000
for i in range(T):
    G.RAassign_static(m)
obtained = G.get_klist()

x = np.arange(0, len(original))
kmin_orig = min(original);  kav_orig = np.mean(original)
kmin = min(obtained);       kav = np.mean(obtained)
ktot_orig = sum(original); ktot = sum(obtained)


# Plot results of test
pl.figure(tight_layout = True)
x = np.arange(0, len(original))
ktot_orig = sum(original)
pl.step(x, original / ktot_orig, label = "Original", c="b")
ktot = sum(obtained)
pl.step(x, obtained / ktot, label = "Final", c="darkred")
pl.legend()
pl.ylim([ pl.ylim()[0], pl.ylim()[1] * 1.1])
props = dict(boxstyle='round', facecolor = 'white', edgecolor = "r", ls = '--', 
             alpha=0.5)
pl.annotate (f"m = {m}", (0.05, 0.92), xycoords = "axes fraction",  
              va='bottom', ha = 'left',bbox = props) 
pl.ylabel("$k_i$ (normalised)")
pl.xlabel("Nodes")
pl.savefig(f"Testing/RAassign_m{m}")
pl.show()

del m, T, G, i
del x, original, obtained
del kmin_orig, kmin, kav_orig, kav
del props


#%% DIAMETER & CLUSTERING
m = 10
T = np.array([5e3, 1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5])
M = 5
mode = "PAalt"
try:
    C, Cstd = np.load(f"Testing/clustering_m{m}.npy")
except:
    Glist = []
    C = []
    Cstd = [] 
    for Ti in T:
        c = []
        for i in tqdm(range(M)):
            Ti = int(Ti)
            Gi = BAnetwork(m, mode)
            Gi.grow(Ti - Gi.getN(), m = m, mode = mode)
            Gi = Gi.networkX()
            c.append(nx.average_clustering(Gi))
        C.append(np.mean(c))
        Cstd.append(np.std(c))
    
    np.save(f"Testing/clustering_m{m}.npy", [C, Cstd], allow_pickle = True)
    
pl.figure("Clustering")
C = np.array(C)
Cstd = np.array(Cstd)
pl.errorbar(T, C, Cstd, fmt =  ".", capsize = 4, label = f"m = {m}")
N = np.linspace(min(T)-100, max(T)+100)
pl.plot(N, avgC(m, N, 2*m+1), c = "black", ls = "--", label = "Theory")
pl.ylabel(r"$\langle C \rangle$")
pl.xlabel("$N$")
pl.xscale("log")
pl.yscale("log")
pl.legend()
pl.savefig(f"Testing/Clustering_m{m}.pdf")





