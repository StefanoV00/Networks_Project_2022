# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 23:40:50 2022

@author: Stefano
"""
from network_functions import *
from network_analysis_functions import *
from network_theory import *
from network_plotf import *

import numpy as np
import pandas as pd

from scipy.stats import cumfreq, chisquare, kstest, ks_1samp, ks_2samp
from scipy.stats import kstwo
from scipy.stats import anderson_ksamp, cramervonmises, norm
from timeit import default_timer as timer 


#%%
def statistics(m, N, mode, scale1 = 1.2, scale2 = 1.2, 
               logcut = 0.75, cumcut = 0.75, wrongtheory = 0, avg = False, 
               chi2 = False, KSall = True):
    """
    Perform Statistical Tests on Data

    Parameters
    ----------
    m : int
    N : int
    mode : str
    scale1 : float
        Scale for logbinning
    scale2 : float
        Scale for logbinning of flattened data
    logcut : float
        Fraction considered for Chi test of body (rest goes with tail)
        It is a log-like definition.
        Alternatively, value of k at which to cut.
    cumcut : float
        Fraction considered for KS test of body (rest goes with tail)
        It is NOT a log-lik definition.
        Alternatively, value of k at which to cut.
    wrongtheory : TYPE, optional
        DESCRIPTION. The default is 1.
    avg : Bool
        If True, do also avg for logbinning, otherwise exact count only
    KSbd : Bool
        Do KS test diversified for body and tail

    Raises
    ------
    Exception
        "N has length, not ok!"
    Exceptin
        "m has length bigger than 2"
        "m has length but is not a tuple"

    Returns
    -------
    None.

    """

    n = logcut
    # CHECKS
    if hasattr(N, "__len__"):
        raise Exception("N has length, not ok!")
    if isinstance(m, tuple):
        if len(m) <= 2:
            m = np.mean(m)
        else:
            raise Exception("m has length bigger than 2")
    elif hasattr(m, "__len__"):
        raise Exceptin("m has length but is not a tuple")
     
    #EXTRACT (AND PLOT) DATA
    N = N
    klists = np.load(f"{mode}/avklist_m{m}_N{N}.npy")
    
    klists_flat = np.sort(klists.flatten()) #also flattens
    
    kbinned, kstd, kcentres = load_logbin(m , N, mode, scale1, 
                                          truecount = True, pad = True)
    pkstuff, [cdfk, cdfk_std, ks] = k_plot(m , N, mode, scale1, 
                                            wrongtheory = wrongtheory, 
                                            logbinned = True, 
                                            cumulative = True)
    if n > 1:
        n = np.where(kcentres >= n)[0]
        if hasattr(n, "__len__"):
            try:
                n = n[0]
            except IndexError:
                pass
    
    code = mode[:2]
    mypk = globals()[f"{code}_pk"]
    mycdfk = globals()[f"{code}_cdfk"]
    
    theory  = mypk(kcentres, m)
    if cumcut <= 1:
        Lcum = int (len(ks)*cumcut)
    else:
        print(f"KS: k_body_tail is {cumcut}")
        print(f"KS: kmax is {klists_flat[-1]}")
        Lcum = cumcut - m #the index is kcut - m
        
    if KSall == 1:
        # FOR KS TEST
        kcut = ks[Lcum]
        l = int(len(klists_flat))
        lbody = int(len(klists_flat[klists_flat<=kcut]))
        ltail = l - lbody
        cumtheory  = mycdfk(ks, m)
        delta = abs(cdfk - cumtheory)
        D = max(delta)
        Dbody = max(delta[:Lcum]) / cdfk[Lcum-1]
        Dtail = max(delta[Lcum:])
        Dtail_ren = Dtail / (cdfk[-1] - cdfk[Lcum-1])
        # Dbody = max(cdfk[:Lcum] - cumtheory[:Lcum]) / cdfk[Lcum-1]
        # Dtail = max(cdfk[Lcum:] - cumtheory[Lcum:])
        # Dtail_ren = max(cdfk[Lcum:] - cumtheory[Lcum:]) / (cdfk[-1] - cdfk[Lcum-1])
        print("KOLMOGOROV - SMIRNOV TEST")
        print(f"{ks[np.where(delta == D)[0]]}")
        print(f"Whole ({l} Datapoints): \
              \n    -D = {D}\
              \n    -p = {kstwo.sf(D, l)}")
        print(f"Body ({lbody} Datapoints): \
              \n    -D = {Dbody}\
              \n    -p = {kstwo.sf(Dbody, l)}")
        print(f"Tail ({ltail} Datapoints): \
              \n    -D = {Dtail}\
              \n    -p = {kstwo.sf(Dtail, ltail)}")
        print(f"Tail Renormalised({ltail} Datapoints): \
              \n    -D = {Dtail_ren}\
              \n    -p = {kstwo.sf(Dtail_ren, ltail)}")
        print()
    
    elif KSall == 2:
        # FOR KS TEST
        kcut = ks[Lcum]
        l = int(len(klists[0]))
        lbody = int(len(klists[0][klists[0]<=kcut]))
        ltail = l - lbody
        cumtheory  = mycdfk(ks, m)
        delta = abs(cdfk - cumtheory)
        D = max(delta)
        Dbody = max(delta[:Lcum]) / cdfk[Lcum-1]
        Dtail = max(delta[Lcum:])
        Dtail_ren = Dtail / (cdfk[-1] - cdfk[Lcum-1])
        print("KOLMOGOROV - SMIRNOV TEST")
        print(f"{ks[np.where(delta == D)[0]]}")
        print(f"Whole ({l} Datapoints): \
              \n    -D = {D}\
              \n    -p = {kstwo.sf(D, l)}")
        print(f"Body ({lbody} Datapoints): \
              \n    -D = {Dbody}\
              \n    -p = {kstwo.sf(Dbody, l)}")
        print(f"Tail ({ltail} Datapoints): \
              \n    -D = {Dtail}\
              \n    -p = {kstwo.sf(Dtail, l)}")
        print(f"Tail Renormalised({ltail} Datapoints): \
              \n    -D = {Dtail_ren}\
              \n    -p = {kstwo.sf(Dtail_ren, l)}")
        print()
    # if "PA" in mode:
    #     theory  = mypk(kcentres, m)
    #     if cumcut <= 1:
    #         Lcum = int (len(ks)*cumcut)
    #     else:
    #         print(f"KS: k_body_tail is {cumcut}")
    #         print(f"KS: kmax is {klists_flat[-1]}")
    #         Lcum = cumcut - m #the index is kcut - m
    #     if KSall:
    #         # FOR KS TEST
    #         cumtheory  = mycdfk(ks, m)
    #         D = max(cdfk - cumtheory)
    #         Dbody = max(cdfk[:Lcum] - cumtheory[:Lcum]) / cdfk[Lcum-1]
    #         Dtail = max(cdfk[Lcum:] - cumtheory[Lcum:]) / (cdfk[-1] - cdfk[Lcum-1])
    #     else:
    #         D = 0
    #         Dbody = 0
    #         Dtail = 0
            
        
    # if mode == "RA":
    #     theory = RA_pk(kcentres, m)
    #     cumtheory = RA_cdfk(ks, m)
    #     # just to mantain same function
    #     theory1 = RA_pk(kcentres, m)
    #     if KSall:
    #         #FOR KS TEST
    #         D = max(cdfk - cumtheory)
    #         if cumcut <= 1:
    #             Lcum = int (len(cumtheory)*cumcut)
    #         else:
    #             Lcum = cumcut - m
    #         Lcum = int (len(cdfk)*cumcut)
    #         Dbody = max(cdfk[:Lcum] - cumtheory[:Lcum]) / (cdfk[Lcum-1])
    #         Dtail = max(cdfk[Lcum:] - cumtheory[Lcum:]) / (cdfk[-1] - cdfk[Lcum-1])
    #     else:
    #         D = 0
    #         Dbody = 0
    #         Dtail = 0
        
    del pkstuff, cdfk, cdfk_std
    
    if chi2:
        ###########################################################################
        #CHI2 TEST
        ###########################################################################
        # Find bin-sizes associated with pk
        smax = np.amax(klists)
        jmax = np.ceil(np.log(smax)/np.log(scale1))
        binedges = scale1 ** np.arange(1,jmax + 1)
        binedges = np.unique(binedges.astype('uint64'))
        while len(binedges) > len(kcentres) + 1:
            binedges = binedges[1:]
        binsizes = binedges[1:] - binedges[:-1]
    
        observed = kbinned
        expected = np.array([sum(mypk (np.arange(max(binedges[i], m), binedges[i+1]), m))
                      * N for i in range(len(binsizes))])
        doChi(observed, expected, "(kbinned vs vs sum_bin_p(k))", mode)
        
        if avg:
            expected = theory * binsizes * N
            doChi(observed, expected, "(kbinned vs avgp(k)*ds)", mode)
        
        # BODY
        if n <= 1:
            L = int(n*len(kbinned))
        else:
            L = n
        observed = kbinned[:L]
        print(f"CHI SQUARED: k_body_tail is {kcentres[L]}")
        print(f"CHI SQUARED: kmax is {klists_flat[-1]}")
        expected = np.array([sum(mypk (np.arange(max(binedges[i], m), binedges[i+1]), m) )
                      * N for i in range(L)])
        print(f"kmax = {L}")
        doChi(observed, expected, f"(kbinned[{n}] body vs sum_bin_p(k))",
              mode)
        
        if avg:
            expected = theory[:L] * binsizes[:L] * N
            doChi(observed, expected, f"(kbinned[{n}] body vs avgp(k)*ds)",
                  mode)
        
        # TAIL
        observed = kbinned[L:]
        expected = np.array([sum(mypk (np.arange(max(binedges[i], m), binedges[i+1]), m) )
                      * N for i in range(L, len(kbinned))])
        print(f"kmin = {L}")
        doChi(observed, expected, f"(kbinned[{n}:] tail vs sum_bin_p(k))",
              mode)
        
        if avg:
            expected = theory[L:] * binsizes[L:] * N
            doChi(observed, expected, f"(kbinned[{n}:] tail vs avgp(k)*ds)", 
                  mode)
    
    
        ##########################################################################
        # FLATTENED
        print("\n#########################################\nFLATTENED")
        pk_centres_flat, pk_flat, binedges = logbin(klists_flat, scale = scale2, 
                                                    truecount = True, pad = True,
                                                    edges = True)
        # jmax = np.ceil(np.log(smax)/np.log(scale2))
        # binedges = scale2 ** np.arange(1,jmax + 1)
        # binedges = np.unique(binedges.astype('uint64'))
        # while len(binedges) > len(pk_centres_flat) + 1:
        #     binedges = binedges[1:]
        binsizes = binedges[1:] - binedges[:-1]
         
        if mode == "PA":
            theoryflat  = mypk(pk_centres_flat, m)
        # new one
        NM = len(klists_flat)
        observed = pk_flat
        expected = np.array([sum(mypk (np.arange(max(binedges[i], m), binedges[i+1]), m) )
                      * NM for i in range(len(binsizes))])
        # expected = []
        # for i in range(len(binsizes)):
        #     expected.append(NM*(mycdfk(binedges[i+1]-1, m) - mycdfk(binedges[i], m)))
        # expected = np.array(expected)
        doChi(observed, expected, "(kflat binned vs sum_bin_p(k))",
              mode)
        
        if avg:
            observed = pk_flat
            expected = theoryflat * binsizes * NM
            doChi(observed, expected, "(kflat binned vs avgp(k)*ds)",
                  mode)
        
        # BODY
        if n <= 1:
            L = int(n*len(kbinned))
        else:
            L = n
        observed = pk_flat[:L]
        expected = np.array([sum(mypk (np.arange(max(binedges[i], m), binedges[i+1]), m) )
                      * NM for i in range(L)])
        doChi(observed, expected, f"(kflat binned[:{n}] body vs sum_bin_p(k))",
              mode)
        
        if avg:
            expected = theoryflat[:L] * binsizes[:L] * NM
            doChi(observed, expected, f"(kflat binned[:{n}] body vs avgp(k)*ds)",
                  mode)
        
        # TAIL
        observed = pk_flat[L:]
        expected = np.array([sum(mypk (np.arange(max(binedges[i], m), binedges[i+1]), m) )
                      * NM for i in range(L, len(pk_flat))])
        if not avg: last = True
        else: last = False 
        doChi(observed, expected, f"(kflat binned[{n}:] tail vs sum_bin_p(k))",
              mode, last)
        
        if avg:
            expected = theoryflat[L:] * binsizes[L:] * NM
            doChi(observed, expected, 
                  f"(kflat binned[{n}:] tail vs avgp(k)*ds)", mode, True)
    
    
    ###########################################################################
    #KOLMOGORV-SMIRNOFF
    ###########################################################################
    y = doKS (mode, m, klists_flat, D)
    print("\n FOR BODY")
    doKS (mode, m, klists_flat[klists_flat<ks[Lcum]], Dbody, y[y<ks[Lcum]])
    print("\n FOR TAIL")
    doKS (mode, m, klists_flat[klists_flat>=ks[Lcum]], Dtail, y[y>=ks[Lcum]])
    
    ##########################################################################
    #ANDERSON DARLING TEST
    ##########################################################################
    doAD (klists_flat, y, mode)
    
    # ##########################################################################
    # #CRAMER-VON MISES TEST
    # ##########################################################################
    #doCvM(mode, m, klists_flat)





#%% BEHIND THE SCENES
def doChi(observed, expected, string, mode = "PA", last = False):
    ddof = 0
    chi2, pvalue     = chisquare(observed, expected, ddof)
    chi2_reduc = chi2 / (len(observed) - ddof)
    #columns = ["Hypothesis", "Reduced \u03c7^2", "pvalue"]
    print("\nRESULTS OF \u03c7^2 TEST" + string)
    print("Chi2:  ", chi2_reduc, "   Pvalue:  ", pvalue)
    print(f"Out of {len(observed)} bins, \
    the minimum bin value was {round(min(observed), 5)}, and")
    print(f"- {len(np.where(observed <  5)[0])} bins had count smaller than  5")
    print(f"- {len(np.where(observed < 10)[0])} bins had count smaller than 10")
    print(f"- {len(np.where(observed < 20)[0])} bins had count smaller than 20")
    if last:
        print("According to scipy.stats.chisquare documentation, \
              \na typical rule is that all of the observed and expected freqs \
              \nshould be at least 5.\
              \nAccording to [3], the total number of samples is recommended \
              \nto be greater than 13.")

         
            
def doKS (mode, m, klists_flat, D, y = 0):
    NM = len(klists_flat)
    mypk = globals()[f"{mode[:2]}_pk"]
    mycdfk = globals()[f"{mode[:2]}_cdfk"]

    KS, KSpvalue = ks_1samp(klists_flat, mycdfk, args = (m,), mode = "exact")
    print("\nRESULTS OF KOLMOGORV-SMIRNOFF TEST (Data vs CDF)")
    print(f"(Flatten array, N*M = {NM})")
    print("KS:  ", KS)
    print("My KS: ", D)
    print("Pvalue: ", KSpvalue)
    if not hasattr( y, "__len__"):
        ks2 = np.arange(min(klists_flat), max(klists_flat) + int(1e6))
        p  = mypk(ks2, m)
        p /= sum(p)
        y = np.random.choice(ks2, size = NM, p = p)
    KS, KSpvalue = ks_2samp(klists_flat, y, mode = "exact")
    print("\nRESULTS OF KOLMOGORV-SMIRNOFF TEST (Data vs Simulated)")
    print(f"(Flatten array, N*M = {NM})")
    print("KS:  ", KS)
    print("My KS: ", D)
    print("Pvalue: ", KSpvalue)
    return y
    

def doAD (klist_flat, y, mode):
    y = np.array(y)
    NM = len(klist_flat)
    mypk = globals()[f"{mode[:2]}_pk"]
    mycdfk = globals()[f"{mode[:2]}_cdfk"]
    
    AD1, AD1list, ADp1 = anderson_ksamp([klist_flat, y])
    print("\nRESULTS OF ANDERSON-DERLING TEST")
    print(f"(Flatten array, N*M = {NM})")
    columns = ["Hypothesis","Anderson-Darling stat","AD pvalue"]
    content = ["Hypothesis ",   AD1,   ADp1]
    print("Critical values: ", "25%, 10%, 5%, 2.5%, 1%, 0.5%, 0.1%")
    print("Hypothesis : ", AD1list)
    print(columns)
    print(content)
    print("")


def doCvM (mode, m, klists_flat):
    NM = len(klists_flat)
    mypk = globals()[f"{mode[:2]}_pk"]
    mycdfk = globals()[f"{mode[:2]}_cdfk"]
    CM1, CMpvalue1 = cramervonmises(klists_flat, mycdfk, args = (m,))
    columns = ["Hypothesis", "Cramer-Von Mises", "CM pvalue"]
    content = ["Hypothesis ",   CM1,   CMpvalue1]
    print("\nRESULTS OF CRAMER-VON MISES TEST")
    print(pd.DataFrame(content, columns = columns))
    