# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 19:05:55 2022

@author: Stefano
"""
from network_functions import *
import numpy as np

from timeit import default_timer as timer 
from tqdm import tqdm


#############################################################################
# DEFINE HERE YOUR DATA
mlist = [1, 10, 25, 50, 100]
Tlist = [1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5, 1e6]
M     = 50
mode = "PA"
r     = "r"
seed  = 0 #0 means no seed for the way functions were written
############################################################################


def run_save (mlist, Tlist, M, mode, r = 0, seed = 0):
    for m in mlist:
        print(f"Working with m = {m}")
        if r == "r":
            ri = int(m/2)
        else:
            ri = r
        for T in Tlist:
            print(f"Working with T = {T}")
            
            N = T*1
            if mode == "PA" or mode == "RA" or "RP" in mode:
                T = T - 2*m-1
            elif mode == "PAalt" or mode == "RAalt":
                T = T - m - 1
            #Prepare to Save Data
            try:
                avklist_old_mT = np.load(f"{mode}/avklist_m{m}_N{N}.npy", 
                                      allow_pickle = True)
                avklist_mT = avklist_old_mT.tolist()
            except FileNotFoundError:
                avklist_mT = []
                
            # Make M realisations and save them
            for realis in tqdm(range(M), "Realisations"):
                if hasattr(m, "__len__"):
                    G = BAnetwork(max(m), mode = mode)
                else:
                    G = BAnetwork(m, mode = mode)
                G.grow(T = T, m = m, mode = mode, r = ri, seed = seed)
                if not G.check(mode)[0]:
                    print(f"At {realis}th realisation check failed")
                    break
                else:
                    avklist_mT.append(G.get_klist()*1)
                    
            #Save list
            np.save(f"{mode}/avklist_m{m}_N{N}", avklist_mT, 
                    allow_pickle = True)


Tlist = [int(T) for T in Tlist]
run_save(mlist, Tlist, M, mode, r, seed)
del mlist, Tlist, M, mode, r, seed


#############################################################################
# DEFINE HERE YOUR DATA
mlist = [1, 10]
Tlist = [1e4, 2.5e4, 5e4, 1e5]
M     = 10
mode = "RA"
r     = "r"
seed  = 0 #0 means no seed for the way functions were written
############################################################################
Tlist = [int(T) for T in Tlist]
run_save(mlist, Tlist, M, mode, r, seed)
del mlist, Tlist, M, mode, r, seed

##############################################################################

#############################################################################
# DEFINE HERE YOUR DATA
mlist = [25]
Tlist = [1e5, 2.5e5, 5e5, 1e6]
M     = 50
mode = "PAalt"
r     = "r"
seed  = 0 #0 means no seed for the way functions were written
############################################################################
Tlist = [int(T) for T in Tlist]
run_save(mlist, Tlist, M, mode, r, seed)
del mlist, Tlist, M, mode, r, seed


#############################################################################
# DEFINE HERE YOUR DATA
mlist = [50, 100]
Tlist = [1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5, 1e6]
M     = 50
mode = "PAalt"
r     = "r"
seed  = 0 #0 means no seed for the way functions were written
############################################################################
Tlist = [int(T) for T in Tlist]
run_save(mlist, Tlist, M, mode, r, seed)
del mlist, Tlist, M, mode, r, seed


#############################################################################
# DEFINE HERE YOUR DATA
mlist = [1, 10, 25, 50, 100]
Tlist = [1e4, 2.5e4, 5e4, 1e5, 2.5e5, 5e5, 1e6]
M     = 50
mode = "RAalt"
r     = "r"
seed  = 0 #0 means no seed for the way functions were written
############################################################################
Tlist = [int(T) for T in Tlist]
run_save(mlist, Tlist, M, mode, r, seed)
del mlist, Tlist, M, mode, r, seed


            