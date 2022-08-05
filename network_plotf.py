# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 20:20:11 2022

@author: Stefano
"""
from network_theory import *
from network_analysis_functions import *

import numpy as np
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit
from scipy.stats import chisquare
from uncertainties import ufloat

#%% DEGREE DISTRIBUTION   
####################################################
##############################################################################
def k_plot(m, N, mode, scale,
           mscale = False, theory = True, wrongtheory = False, collapse = False, 
           logbinned = True, cumulative = False, zipf = False):
    """
    Import and Plot 1 or more klists. m and N canNOT be both lists.

    Parameters
    ----------
    m : int or list(int)
        
    N : int or list(int)
        
    mode : str
    
    scale : float>1
        Scaling factor for logbinning.
        
    mscale : Bool
        If True, collapse data for PA dividing by 2m(m+1).
    theory : Bool
        If True, plot theoretical best estimate.
    wrongtheory : Bool
        If True, plot theoretical worst estimate for PA case.
    collapse : Bool
        If True, plot data collapse.
    logbinned : Bool
        If True, logbin and plot logbinned distribution
    cumulative : Bool
        If True, plot cumulative functions
    zipf : Bool
        If True, plot zipf rank plot

    Returns
    -------

    result : TYPE
        ONLY IF m AND N ARE BOTH SCALARS:\n
        [[pk, pkstd, pkcentres], [cdfk, cdfk_std, ks], [ranked_avg, ranked_std, r]]
        where each of the three is returned iff logbinned, cumulative or zipf 
        are True respectively. 

    """
    if hasattr(m, "__len__") and hasattr(N, "__len__"):
        if not isinstance(m, tuple) and len(m)==2:
            raise Exception("Trust me, you don't want both m and N to be lists.")
    
    if hasattr(m, "__len__") and isinstance(m, list):
        if logbinned:
            for mi in m[:-1]:
                if hasattr(mi, "__len__"):
                    mi = int(np.mean(mi))
                loop_t = 0 if mscale else theory
                loop_wt = 0 if mscale else wrongtheory
                load_bin_plot(m = mi, N = N, mode = mode, scale =  scale, 
                              theory = loop_t, wrongtheory = loop_wt, mscale = mscale,
                              collapse = collapse,
                              logbinned = logbinned, cumulative = False, zipf = False,
                              dont = True, save = False)
            kmin = np.amin(m) if mscale else 0
            if hasattr(m[-1], "__len__"):
                mi = int(np.mean(m[-1]))
            else:
                mi = m[-1]
            load_bin_plot(m = mi, N = N, mode = mode, scale =  scale, 
                           theory = theory, wrongtheory = wrongtheory, mscale = mscale,
                           collapse = collapse,
                          logbinned = logbinned, cumulative = False, zipf = False,
                          dont = False, save = False, kmin = kmin)
            pl.savefig(f"{mode}/pk_m{m}_N{N}_mscale{mscale}.pdf")
    
        if cumulative:
            for mi in m[:-1]:
                if hasattr(mi, "__len__"):
                    mi = int(np.mean(mi))
                load_bin_plot(m = mi, N = N, mode = mode, scale =  scale, 
                              theory = theory, wrongtheory = wrongtheory, 
                              mscale = mscale,
                              collapse = collapse,
                              logbinned = False, cumulative = True, zipf = False,
                              dont = True, save = False)
            if hasattr(m[-1], "__len__"):
                mi = int(np.mean(m[-1]))
            else:
                mi = m[-1]
            load_bin_plot(m = mi, N = N, mode = mode, scale =  scale, 
                           theory = theory, wrongtheory = True, mscale = mscale,
                              collapse = collapse,
                          logbinned = False, cumulative = True, zipf = False,
                          dont = False, save = False)
            pl.savefig(f"{mode}/cdfk_m{m}_N{N}.pdf")
            
        if zipf:
            for mi in m[:-1]:
                if hasattr(mi, "__len__"):
                    mi = int(np.mean(mi))
                load_bin_plot(m = mi, N = N, mode = mode, scale =  scale, 
                              theory = theory, wrongtheory = wrongtheory,
                              mscale = mscale,
                              collapse = collapse,
                              logbinned = False, cumulative = False, zipf = True,
                              dont = True, save = False)
            if hasattr(m[-1], "__len__"):
                mi = int(np.mean(m[-1]))
            else:
                mi = m[-1]
            load_bin_plot(m = mi, N = N, mode = mode, scale =  scale, 
                          theory = theory, wrongtheory = wrongtheory, 
                          mscale = mscale,
                          collapse = collapse,
                          logbinned = False, cumulative = False, zipf = True,
                          dont = False, save = False)
            pl.savefig(f"{mode}/zipf_m{m}_N{N}.pdf")

    # For many Ns for same m 
    elif hasattr(N, "__len__"): 
        # if isinstance(m, tuple) and len(m)==2:
        #     mi = int(np.mean(m))
        mi = m
        if logbinned:
            for  Ti in N[:-1]:
                loop_t = 0 if mscale else theory
                loop_wt = 0 if mscale else wrongtheory
                load_bin_plot_m(m = mi, N = Ti, mode = mode, scale =  scale, 
                              theory = loop_t, wrongtheory = loop_wt, mscale = mscale,
                              collapse = collapse,
                              logbinned = logbinned, cumulative = False, zipf = False,
                              dont = True, save = False)
            
            load_bin_plot_m(m = mi, N = N[-1], mode = mode, scale =  scale, 
                           theory = theory, wrongtheory = wrongtheory, mscale = mscale,
                           collapse = collapse,
                          logbinned = logbinned, cumulative = False, zipf = False,
                          dont = False, save = False)
            pl.savefig(f"{mode}/pk_m{m}_N{N}_mscale{mscale}.pdf")
    
        if cumulative:
            for  Ti in N[:-1]:
                load_bin_plot_m(m = mi, N = Ti, mode = mode, scale =  scale, 
                               theory = theory, wrongtheory = wrongtheory, mscale = mscale,
                               collapse = collapse,
                              logbinned = False, cumulative = True, zipf = False,
                              dont = True, save = False)
            load_bin_plot_m(m = mi, N = N[-1], mode = mode, scale =  scale, 
                          theory = theory, wrongtheory = True, mscale = mscale,
                          collapse = collapse,
                          logbinned = False, cumulative = True, zipf = False,
                          dont = False, save = False)
            pl.savefig(f"{mode}/cdfk_m{m}_N{N}.pdf")
            
        if zipf:
            for Ti in N[:-1]:
                load_bin_plot_m(m = m, N = Ti, mode = mode, scale =  scale, 
                               theory = theory, wrongtheory = wrongtheory, mscale = mscale,
                              collapse = collapse,
                              logbinned = False, cumulative = False, zipf = True,
                              dont = True, save = False)
            load_bin_plot_m(m = mi, N = N[-1], mode = mode, scale =  scale, 
                           theory = theory, wrongtheory = wrongtheory, mscale = mscale,
                              collapse = collapse,
                          logbinned = False, cumulative = False, zipf = True,
                          dont = False, save = False)
            pl.savefig(f"{mode}/zipf_m{m}_N{N}.pdf")
    
    # None is a list
    elif isinstance(m, tuple) and len(m)==2:
        mi = int(np.mean(m))
        result = load_bin_plot(m = m, N = N, mode = mode, scale =  scale, 
                      theory = theory, wrongtheory = wrongtheory, 
                      mscale = mscale,
                      logbinned = logbinned, cumulative = cumulative,
                      zipf = zipf,
                      dont = False, save = False)
        pl.savefig(f"{mode}/zipf_m{m}_N{N}.pdf")
        return result
        
    # None is a list
    else:
        result = load_bin_plot(m = m, N = N, mode = mode, scale =  scale, 
                               theory = theory, wrongtheory = wrongtheory, 
                               mscale = mscale,
                               logbinned = logbinned, cumulative = cumulative,
                               zipf = zipf,
                               dont = False, save = True)
        return result



def load_bin_plot (m = 1, N = 100000, mode = "PA", scale = 1., 
                   theory = True, wrongtheory = True, mscale = False,
                   collapse = False, 
                   logbinned = True, cumulative = False, zipf = False,
                   dont = False, save = True, kmin = 0):
    """
    The function imports the klist saved as f'{mode}//avklist_m{m}_N{N}.npy', 
    then bins the data insid, finaly plots and saves such plot.
    If multiple realisation are done, it averages them.
    
    Parameters
    ----------
    First, the parameters with which data was simulated:
        mode : "PA" or "RA" or "PR" or "RP"
        m : int
        N : int
    
    scale : float
        Scaling factor. Default is 1., i.e. normal binning.
    
    theory : Bool
        If True (default) then plot also theoretical expectations.
        
    wrongtheory : Bool
        If True then plot also wrng p(k) theoretical expectations.
    
    mscale : Bool
        If True, scale the PA p(k) so that it is k only dependent (not m)
        
    logbinned : Bool, default True
        If True, logbin and plot data. 
        
    cumulative : Bool, default False
        If True, find and plot cdf.
        
    zipf : Bool, default False
        If True rank (biggest first) and plot the k_is.
    
    dont : Bool, default False
        If dont, do NOT plot:
            - label of theory plots
            - box with value of N
    
    kmin : int
        Set a minimum k for the range for the theoretical results, if non 0.

    Returns
    -------
    result : TYPE
        [[pk, pkstd, pkcentres], [cdfk, cdfk_std, ks], [ranked_avg, ranked_std, r]]
        where each of the three is returned iff logbinned, cumulative or zipf 
        are True respectively. 
    """
    klist = np.load(f"{mode}/avklist_m{m}_N{N}.npy")
 
    if kmin == 0:
        kmin = np.amin(klist)
    sqrtM = np.sqrt(len(klist))
    kmax = np.amax(klist)
    kmax_long = kmax + 10
    ks = np.arange(kmin, kmax_long)
    
    #pk = logbin(avklist, 1.1, start = 0)
    #pk2 = your_logbin(avklist, 1.1, zeros = True)
    # ks = []
    # for ki in pk[3]:
    #     if ki>m:
    #         ks.append(ki)
    # ks = np.array(ks)
    
    #Get Time string right
    ex = np.log10(N)
    if ex == int(ex):
        ex = int(ex)
        note = f"N = $10^{ex}$"
    else:
        ex10 = int(ex)
        n = int(np.rint(10**(ex-ex10)))
        note = f"N = {n}" +r"$\times$" + f" $10^{ex10}$"
        
    if isinstance(m, tuple) and len(m)==2:
        m = np.mean(m)
        
    result = []
    if mode == "PA" or mode == "PAalt":
        if logbinned:
            
            pk, pkstd, pkcentres = multi_logbin(klist, scale)
            fig = pl.figure(f"{mode}_p(k)_logbin")
            
            if mscale:
                mscaling = 2 * m * (m+1)
            else:
                mscaling = 1
                
            if collapse:
                ks_effect        = ks / (N)**collapse
                pkcentres_effect = pkcentres / (N)**collapse
                ks_col = ks * (ks+1) * (ks+2)
                pk_col = pkcentres * (pkcentres + 1) * (pkcentres + 2)
            else:
                ks_effect = ks
                pkcentres_effect = pkcentres
                ks_col = 1.
                pk_col = 1.
                
            pl.errorbar(pkcentres_effect, pk/mscaling*pk_col, 
                        yerr = pkstd/mscaling*pk_col/sqrtM, 
                        fmt = ".", capsize = 4, 
                        label = f"Data, m = {m}")
            pl.xscale("log"); pl.yscale("log")

            if not collapse:
                if theory:
                    theory2 = PA_pk(ks, m, mscale)
                    if dont:
                        pl.plot(ks_effect, theory2, ls = "--", c = "black")
                    else:
                        if not mscale:
                            label = r"$p(k) = \frac{2m(m+1)}{k(k+1)(k+2)}$"
                        else:
                            label = r"$\tilde{p}(k) = \frac{1}{k(k+1)(k+2)}$"
                        pl.plot(ks_effect, theory2, ls = "--", c = "black",
                                label = label)
                if wrongtheory:
                    theory1 = PA_pk1(ks, m, mscale)
                    if dont:
                        pl.plot(ks_effect, theory1, ls = "--", c = "blue")
                    else:
                        if not mscale:
                            label = r"$p(k) \propto k^{-3}$"
                        else:
                            label = r"$\tilde{p}(k) = k^{-3}$"
                        pl.plot(ks_effect, theory1, ls = "--", c = "blue",
                                label = label)
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props) 
                pl.xlabel(r"$k$")
                pl.ylabel(r"$p(k)$")
            pl.legend(loc = "upper right")
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_pk_m{m}_N{N}.pdf")
            result.append([pk, pkstd, pkcentres])
            
        #PA
        if cumulative:
            cdfk, cdfk_std = multi_cumfreq(klist, ks)
            fig = pl.figure(f"{mode}_cdf(k)")
            pl.errorbar(ks, cdfk, cdfk_std/sqrtM, fmt = ".", capsize = 4, 
                        label = f"Data, m = {m}")
            
            if wrongtheory:
                pass
                # cumtheory1 = PA_cdfk1(ks, m)
                # if dont:
                #     pl.plot(ks, cumtheory1, ls = "--")
                # else:
                #     pl.plot(ks, cumtheory1, ls = "--",
                #             label = r"$F_{k_i}(k) \propto k^{-3}$")
            if theory:
                cumtheory2 = PA_cdfk(ks, m)
                if dont:
                    pl.plot(ks, cumtheory2, ls = "--", c = "black", zorder = 4)
                else:
                    label = r"$F_{k_i}(k) = 1-\frac{m^2+m}{(k+1)(k+2)}$"
                    pl.plot(ks, cumtheory2, ls = "--", c = "black", zorder = 4,
                            label = label)
            
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.1, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props) 
                pl.xscale("log")#; pl.yscale("log")
                pl.xlabel("$k$"); pl.ylabel(r"$F_{k_i}(k)$")
            pl.legend(loc = "lower right")
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_cdfk_m{m}_N{N}.pdf")
            result.append([cdfk, cdfk_std, ks])
        
       
        #PA
        if zipf:
            ranked_avg, ranked_std, r = zipf_f(klist)
            fig = pl.figure(f"{mode}_zipf(k)")
            pl.errorbar(r, ranked_avg, ranked_std/sqrtM, fmt =".", capsize = 4, 
                        label = f"Data, m = {m}")
            pl.xlabel("Rank r")
            pl.ylabel("$k_i$")
            pl.xscale("log")
            pl.yscale("log")
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props) 
            pl.legend()
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_zipf_m{m}_N{N}.pdf")
            result.append([ranked_avg, ranked_std, r])
            
    ##########################################################################    
    elif mode == "RA" or mode == "RAalt":
        if logbinned:
            pk, pkstd, pkcentres = multi_logbin(klist, scale)
            fig = pl.figure(f"{mode}_p(k)_logbin")
            pl.errorbar(pkcentres, pk, yerr = pkstd/sqrtM, fmt = ".", capsize = 4, 
                        label = f"Data, m = {m}")
            if theory:
                pktheory = RA_pk(ks, m)
                if dont:
                    pl.plot(ks, pktheory, ls = "--", c = "black", zorder = 4)
                elif not dont:
                    #theory = RA_pk(pkcentres, m)
                    label = r"$p(k) = \frac{1}{1+m} (\frac{m}{1+m})^{k-m}$"
                    
                    #pl.plot(pkcentres, theory, ls = "--", c = "black", zorder = 4, 
                    pl.plot(ks, pktheory, ls = "--", c = "black", zorder = 4, 
                            label = label)
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props) 
                pl.yscale("log")
                if logbinned == "log":
                    pl.xscale("log")
                pl.xlabel(r"$k_i$")
                pl.ylabel(r"$p(k)$")
            pl.legend(loc = "upper right")
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_pk_m{m}_N{N}.pdf")
            result.append([pk, pkstd, pkcentres])
        
        
        #RA
        if cumulative:
            cdfk, cdfk_std = multi_cumfreq(klist, ks)
            fig = pl.figure(f"{mode}_cdf(k)")
            pl.errorbar(ks, cdfk, cdfk_std/sqrtM, fmt = ".", capsize = 4, 
                        label = f"Data, m = {m}")
            if theory:
                cumtheory = RA_cdfk(ks, m)
                if not dont:
                    pl.plot(ks, cumtheory, ls = "--", c = "black")
                elif dont:
                    label = r"$F_{k_i}(k) = 1-(\frac{m}{1+m})^{k-m}$"
                    pl.plot(ks, cumtheory, ls = "--", c = "black",
                            label = label, zorder=3)
            pl.xscale("log"); #pl.yscale("log")
            pl.xlabel("$k$"); pl.ylabel(r"$F_{k_i}(k)$")
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.1, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props) 
            pl.legend(loc = "lower right")
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_cdfk_m{m}_N{N}.pdf")
            result.append([cdfk, cdfk_std, ks])
        
        #RA
        if zipf:
            ranked_avg, ranked_std, r = zipf_f(klist)
            fig = pl.figure(f"{mode}_zipf(k)")
            pl.errorbar(r, ranked_avg, ranked_std/sqrtM, fmt =".", capsize = 4, 
                        label = f"Data, m = {m}")
            pl.xlabel("Rank r")
            pl.ylabel("$k_i$")
            pl.xscale("log")
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props) 
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_zipf_m{m}_N{N}.pdf")
            result.append([ranked_avg, ranked_std, r])           
     
    ##########################################################################
    if "RP" in mode:
        # RP
        if logbinned:
            pk, pkstd, pkcentres = multi_logbin(klist, scale)
            fig = pl.figure(f"{mode}_p(k)_logbin")
            
            if mscale:
                mscaling = 3 * m * (3*m+2) / 2
            else:
                mscaling = 1
                
            if collapse:
                ks_effect        = ks / (N)**collapse
                pkcentres_effect = pkcentres / (N)**collapse
                ks_col = (ks+m) * (ks+m+1) * (ks+m+2)
                pk_col = pkcentres * (pkcentres + 1) * (pkcentres + 2)
            else:
                ks_effect = ks
                pkcentres_effect = pkcentres
                ks_col = 1.
                pk_col = 1.
                
            pl.errorbar(pkcentres_effect, pk/mscaling*pk_col, 
                        yerr = pkstd/mscaling*pk_col/sqrtM, 
                        fmt = ".", capsize = 4, 
                        label = f"Data, m = {m}")
            pl.xscale("log"); pl.yscale("log")

            if not collapse:
                if theory:
                    theory2 = RP_pk(ks, m)
                    if dont:
                        pl.plot(ks_effect, theory2, ls = "--", c = "black", 
                                zorder = 4)
                    else:
                        if not mscale:
                            label = r"$p(k) = \frac{3}{2}\frac{m(3m+2)}{(k+m)(k+m+1)(k+2)}$"
                        else:
                            label = r"$\tilde{p}(k) = \frac{1}{k(k+1)(k+2)}$"
                        pl.plot(ks_effect, theory2, ls = "--", c = "black",
                                label = label, zorder = 4)
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props) 
                pl.xlabel(r"$k$")
                pl.ylabel(r"$p(k)$")
            pl.legend(loc = "upper right")
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_pk_m{m}_N{N}.pdf")
            result.append([pk, pkstd, pkcentres])
            
        #RP
        if cumulative:
            cdfk, cdfk_std = multi_cumfreq(klist, ks)
            fig = pl.figure(f"{mode}_cdf(k)")
            pl.errorbar(ks, cdfk, cdfk_std/sqrtM, fmt = ".", capsize = 4, 
                        label = f"Data, m = {m}")
            if theory:
                cumtheory = RP_cdfk(ks, m)
                if dont:
                    pl.plot(ks, cumtheory, ls = "--", c = "black", zorder = 4)
                else:
                    label = r"$F_{k_i}(k) = 1-\frac{3}{4}\frac{m(3m+2)}{(k+m+1)(k+m+2)}$"
                    pl.plot(ks, cumtheory, ls = "--", c = "black", zorder = 4,
                            label = label)
            
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.1, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props) 
                pl.xscale("log")#; pl.yscale("log")
                pl.xlabel("$k$"); pl.ylabel(r"$F_{k_i}(k)$")
            pl.legend(loc = "lower right")
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_cdfk_m{m}_N{N}.pdf")
            result.append([cdfk, cdfk_std, ks])
        
        #RP
        if zipf:
            ranked_avg, ranked_std, r = zipf_f(klist)
            fig = pl.figure(f"{mode}_zipf(k)")
            pl.errorbar(r, ranked_avg, ranked_std/sqrtM, fmt =".", capsize = 4, 
                        label = f"Data, m = {m}")
            pl.xlabel("Rank r")
            pl.ylabel("$k_i$")
            pl.xscale("log")
            pl.yscale("log")
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props) 
            pl.legend()
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_zipf_m{m}_N{N}.pdf")
            result.append([ranked_avg, ranked_std, r])
            
    
            
    return result




###############################################################################
def load_bin_plot_m (m = 1, N = 100000, mode = "PA", scale = 1., 
                   theory = True, wrongtheory = True, mscale = False, 
                   collapse = False,
                    logbinned = True, cumulative = False, zipf = False,
                    dont = False, save = True, kmin = 0):
    """
    The function imports the klist saved as f'{mode}//avklist_m{m}_N{N}.npy', 
    then bins the data insid, finaly plots and saves such plot.
    If multiple realisation are done, it averages them.
    
    Parameters
    ----------
    First, the parameters with which data was simulated:
        mode : "PA" or "RA" or "PR" or "RP"
        m : int
        N : int
    
    scale : float
        Scaling factor. Default is 1., i.e. normal binning.
    
    theory : Bool
        If True (default) then plot also theoretical expectations.
        
    wrongtheory : Bool
        If True then plot also wrng p(k) theoretical expectations.
    
    mscale : Bool
        If True, scale the PA p(k) so that it is k only dependent (not m)
        
    logbinned : Bool, default True
        If True, logbin and plot data. 
        
    cumulative : Bool, default False
        If True, find and plot cdf.
        
    zipf : Bool, default False
        If True rank (biggest first) and plot the k_is.
    
    dont : Bool, default False
        If dont, do NOT plot:
            - label of theory plots
            - box with value of N
    
    kmin : int
        Set a minimum k for the range for the theoretical results, if non 0.

    Returns
    -------
    result : TYPE
        [[pk, pkstd, pkcentres], [cdfk, cdfk_std], [ranked_avg, ranked_std, r]]
        where each of the three is returned iff logbinned, cumulative or zipf 
        are True respectively. 
    """
    try:
        klist = np.load(f"{mode}/avklist_m{m}_N{N}.npy")
    except FileNotFoundError:
        klist = np.load(f"{mode}/avklist_m{m}_N{N}.npy")
    if kmin == 0:
        kmin = np.amin(klist)
    kmax = np.amax(klist)
    kmax_long = kmax + 10
    ks = np.arange(kmin, kmax_long)
    
    M = len(klist)
    sqrtM = np.sqrt(M)
    
    
    # Get time string right
    ex = np.log10(N)
    if ex == int(ex):
        ex = int(ex)
        note_T = f"N = $10^{ex}$"
    else:
        ex10 = int(ex)
        n = round(np.rint(10**(ex-ex10)), 1)
        if n == 2:
            n = 2.5
        note_T = f"N = {n}" +r"$\times$" + f" $10^{ex10}$"
    
    #Get m string right
    note = f"m = {m}"
    
    if isinstance(m, tuple) and len(m)==2:
        m = np.mean(m)
        
    result = []
    if "PA" in mode:
            
        if logbinned:
            pk, pkstd, pkcentres = multi_logbin(klist, scale)
            fig = pl.figure(f"{mode}_p(k)_logbin")
            if mscale:
                mscaling = 2 * m * (m+1)
            else:
                mscaling = 1
                
            if collapse:
                ks_effect        = ks / (N)**collapse
                pkcentres_effect = pkcentres / (N)**collapse
                ks_col = ks * (ks+1) * (ks+2)
                pk_col = pkcentres * (pkcentres + 1) * (pkcentres + 2)
            else:
                ks_effect = ks
                pkcentres_effect = pkcentres
                ks_col = 1
                pk_col = 1
                
            pl.errorbar(pkcentres_effect, pk/mscaling * pk_col, 
                        yerr = pkstd/mscaling * pk_col/sqrtM, ls ="--" ,
                        fmt = ".", capsize = 4, label = "Data, " + note_T)
            pl.xscale("log"); pl.yscale("log")
            
            if not collapse:
                if theory:
                    theory2 = PA_pk(ks, m, mscale)
                    if dont:
                        pl.plot(ks_effect, theory2, ls = "--", c = "black")
                    else:
                        if not mscale:
                            label = r"$p(k) = \frac{2m(m+1)}{k(k+1)(k+2)}$"
                        else:
                            label = r"$\tilde{p}(k) = \frac{1}{k(k+1)(k+2)}$"
                        pl.plot(ks_effect, theory2, ls = "--", c = "black",
                                label = label)
                if wrongtheory:
                    theory1 = PA_pk1(ks, m, mscale)
                    if dont:
                        pl.plot(ks_effect, theory1, ls = "--", c = "blue")
                    else:
                        if not mscale:
                            label = r"$p(k) \propto k^{-3}$"
                        else:
                            label = r"$\tilde{p}(k) = k^{-3}$"
                        pl.plot(ks_effect, theory1, ls = "--", c = "blue",
                                label = label)
            
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props) 
                if not collapse:
                    pl.xlabel(r"$k$")
                    pl.ylabel(r"$p(k)$")
                else:
                    pl.xlabel(r"$k/N^{D_N}$")
                    pl.ylabel(r"$p(k)/p_{th}(k)$")
            pl.legend()
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_pk_m{m}_N{N}.pdf")
            result.append([pk, pkstd, pkcentres])
            
            
        #PA
        if cumulative:
            cdfk, cdfk_std = multi_cumfreq(klist, ks)
            fig = pl.figure(f"{mode}_cdf(k).pdf")
            pl.errorbar(ks_effect, cdfk, cdfk_std/sqrtM, fmt = ".", capsize = 4, 
                        label = "Data, " + note_T)
            
            if not collapse:
                if wrongtheory:
                    pass
                    # cumtheory1 = PA_cdfk1(ks, m)
                    # if dont:
                    #     pl.plot(ks, cumtheory1, ls = "--")
                    # else:
                    #     pl.plot(ks, cumtheory1, ls = "--",
                    #             label = r"$F_{k_i}(k) \propto k^{-3}$")
                if theory:
                    cumtheory2 = PA_cdfk(ks, m)
                    if dont:
                        pl.plot(ks_effect, cumtheory2, ls = "--", c = "black", 
                                zorder = 4)
                    else:
                        label = r"$F_{k_i}(k) = 1-\frac{m^2+m}{(k+1)(k+2)}$"
                        pl.plot(ks_effect, cumtheory2, ls = "--", c = "black", 
                                zorder = 4,
                                label = label)

            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props) 
                pl.xscale("log"); pl.yscale("log")
                pl.xlabel("$k$"); pl.ylabel(r"$F_{k_i})(k)$")
            pl.legend()
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_cdfk_m{m}_N{N}.pdf")
            result.append([cdfk, cdfk_std])
        
        #PA
        if zipf:
            ranked_avg, ranked_std, r = zipf_f(klist)
            fig = pl.figure(f"{mode}_zipf(k)")
            pl.errorbar(r, ranked_avg, ranked_std/sqrtM, fmt =".", capsize = 4, 
                        label = "Data, " + note_T)
            pl.xlabel("Rank r")
            pl.ylabel("$k_i$")
            pl.xscale("log")
            pl.yscale("log")
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props)  
            pl.legend()
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_zipf_m{m}_N{N}.pdf")
            result.append([ranked_avg, ranked_std, r])
            
    ###########################################################################   
    elif "RA" in mode:
        if logbinned:
            pk, pkstd, pkcentres = multi_logbin(klist, scale)
            fig = pl.figure(f"{mode}_p(k)_logbin")
            pl.errorbar(pkcentres, pk, yerr = pkstd/sqrtM, fmt = ".", capsize = 4, 
                        label = "Data, " + note_T)
            if theory:
                theory = RA_pk(ks, m)
                if dont:
                    pl.plot(ks, theory, ls = "--", c = "black", zorder = 4)
                elif not dont:
                    #theory = RA_pk(pkcentres, m)
                    label = r"$p(k) = \frac{1}{1+m} (\frac{m}{1+m})^{k-m}$"
                    
                    #pl.plot(pkcentres, theory, ls = "--", c = "black", zorder = 4, 
                    pl.plot(ks, theory, ls = "--", c = "black", zorder = 4, 
                            label = label)
            pl.legend()
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props)  
                pl.yscale("log")
                if logbinned == "log":
                    pl.xscale("log")
                pl.xlabel(r"$k_i$")
                pl.ylabel(r"$p(k)$")
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_pk_m{m}_N{N}.pdf")
            result.append([pk, pkstd, pkcentres])
        
        #RA
        if cumulative:
            cdfk, cdfk_std = multi_cumfreq(klist, ks)
            fig = pl.figure(f"{mode}_cdf(k)")
            pl.errorbar(ks, cdfk, cdfk_std/sqrtM, fmt = ".", capsize = 4, 
                        label = "Data, " + note_T)
            if theory:
                cumtheory = RA_cdfk(ks, m)
                if not dont:
                    pl.plot(ks, cumtheory, ls = "--", c = "black")
                elif dont:
                    label = r"$F_{k_i}(k) = 1-(\frac{m}{1+m})^{k-m}$"
                    pl.plot(ks, cumtheory, ls = "--", c = "black",
                            label = label, zorder=3)
            pl.legend()
            pl.xscale("log"); #pl.yscale("log")
            pl.xlabel("$k$"); pl.ylabel("$p(k)$")
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props)  
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_cdfk_m{m}_N{N}.pdf")
            result.append([cdfk, cdfk_std, ks])
        
        #RA
        if zipf:
            ranked_avg, ranked_std, r = zipf_f(klist)
            fig = pl.figure(f"{mode}_zipf(k)")
            pl.errorbar(r, ranked_avg, ranked_std/sqrtM, fmt =".", capsize = 4, 
                        label = "Data, " + note_T)
            pl.xlabel("Rank r")
            pl.ylabel("$k_i$")
            pl.xscale("log")
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props)  
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_zipf_m{m}_N{N}.pdf")
            result.append([ranked_avg, ranked_std, r])
    
    
    ###########################################################################
    if "RP" in mode:
            
        if logbinned:
            pk, pkstd, pkcentres = multi_logbin(klist, scale)
            fig = pl.figure(f"{mode}_p(k)_logbin")
            if mscale:
                mscaling = 3 * m * (3*m+2) / 2
            else:
                mscaling = 1
                
            if collapse:
                ks_effect        = ks / (N)**collapse
                pkcentres_effect = pkcentres / (N)**collapse
                ks_col = (ks+m) * (ks+m+1) * (ks+m+2)
                pk_col = pkcentres * (pkcentres + 1) * (pkcentres + 2)
            else:
                ks_effect = ks
                pkcentres_effect = pkcentres
                ks_col = 1
                pk_col = 1
                
            pl.errorbar(pkcentres_effect, pk/mscaling * pk_col, 
                        yerr = pkstd/mscaling * pk_col/sqrtM, ls ="--" ,
                        fmt = ".", capsize = 4, label = "Data, " + note_T)
            pl.xscale("log"); pl.yscale("log")
            
            if not collapse:
                if theory:
                    theory2 = RP_pk(ks, m, mscale)
                    if dont:
                        pl.plot(ks_effect, theory2, ls = "--", c = "black")
                    else:
                        if not mscale:
                            label = r"$p(k) = \frac{3}{2}\frac{m(3m+2)}{(k+m)(k+m+1)(k+2)}$"
                        else:
                            label = r"$\tilde{p}(k) = \frac{1}{k(k+1)(k+2)}$"
                        pl.plot(ks_effect, theory2, ls = "--", c = "black",
                                label = label)
            
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props) 
                if not collapse:
                    pl.xlabel(r"$k$")
                    pl.ylabel(r"$p(k)$")
                else:
                    pl.xlabel(r"$k/N^{D_N}$")
                    pl.ylabel(r"$p(k)/p_{th}(k)$")
            pl.legend()
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_pk_m{m}_N{N}.pdf")
            result.append([pk, pkstd, pkcentres])
            
        #RP
        if cumulative:
            cdfk, cdfk_std = multi_cumfreq(klist, ks)
            fig = pl.figure(f"{mode}_cdf(k).pdf")
            pl.errorbar(ks_effect, cdfk, cdfk_std/sqrtM, fmt = ".", capsize = 4, 
                        label = "Data, " + note_T)
            
            if not collapse:
                if wrongtheory:
                    pass
                    # cumtheory1 = PA_cdfk1(ks, m)
                    # if dont:
                    #     pl.plot(ks, cumtheory1, ls = "--")
                    # else:
                    #     pl.plot(ks, cumtheory1, ls = "--",
                    #             label = r"$F_{k_i}(k) \propto k^{-3}$")
                if theory:
                    cumtheory2 = RP_cdfk(ks, m)
                    if dont:
                        pl.plot(ks_effect, cumtheory2, ls = "--", c = "black", 
                                zorder = 4)
                    else:
                        label = r"$F_{k_i}(k) = 1-\frac{3}{4}\frac{m(3m+2)}{(k+m+1)(k+m+2)}$"
                        pl.plot(ks_effect, cumtheory2, ls = "--", c = "black", 
                                zorder = 4,
                                label = label)

            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props) 
                pl.xscale("log"); pl.yscale("log")
                pl.xlabel("$k$"); pl.ylabel(r"$F_{k_i})(k)$")
            pl.legend()
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_cdfk_m{m}_N{N}.pdf")
            result.append([cdfk, cdfk_std])
        
        #RP
        if zipf:
            ranked_avg, ranked_std, r = zipf_f(klist)
            fig = pl.figure(f"{mode}_zipf(k)")
            pl.errorbar(r, ranked_avg, ranked_std/sqrtM, fmt =".", capsize = 4, 
                        label = "Data, " + note_T)
            pl.xlabel("Rank r")
            pl.ylabel("$k_i$")
            pl.xscale("log")
            pl.yscale("log")
            if not dont:
                props = dict(boxstyle='round', facecolor='white',edgecolor="r", 
                             ls = '--', alpha=0.5)
                pl.annotate (note, (0.05, 0.05), xycoords = "axes fraction", 
                              va='bottom', ha = 'left', bbox=props)  
            pl.legend()
            fig.tight_layout()
            if save:
                pl.savefig(f"{mode}/plot_zipf_m{m}_N{N}.pdf")
            result.append([ranked_avg, ranked_std, r])
            
    return result


#%%MAXIMUM DEGREE  
####################################################
##############################################################################
def plot_k1_N(m, Tlist, mode = "PA", theory = [1,2,3,4],
              dont = False, fit = True):
    """
    Parameters
    ----------
    m : int
    Tlist : ndarray
    mode : str
        The default is "PA".
    theory : theory options to plot
        DESCRIPTION. The default is [1,2,3,4].

    Returns
    -------
    [amplitude, exponent], [a_std, e_std]

    """
    kmax = []
    kstd = []
    alltheory = []
    alltags = []
    
    if mode == "PA" or mode == "RA":
        N = np.array(Tlist) + 2*m + 1
        Ns = np.linspace(min(N)-100, max(N) + 100 )
        m0 = 2
    elif mode == "PAalt" or mode == "RAalt" or "RP" in mode:
        N = np.array(Tlist) + m + 1
        Ns = np.linspace(min(N)-100, max(N) + 100 )
        m0 = 1
    
    M = []
    ###########################################################################
    # Get Experimental and Theoretical Maximum
    for i, Ti in enumerate(Tlist):
        klists = np.load(f"{mode}/avklist_m{m}_N{int(Ti)}.npy")
        kmaxs_T = np.amax(klists, axis = 1)
        kmax.append(np.mean(kmaxs_T))
        kstd.append(np.std(kmaxs_T))
        theory_ind = []
        M.append(len(klists))
        if i == 0:  
            for j in theory:
                try:
                    f = globals()[f"{mode[:2]}_kmax{j}"]
                    t, tag = f(Ns, m, True, m0)
                    theory_ind.append(t)
                    alltags.append(tag)
                except KeyboardInterrupt:
                    raise KeyboardInterrupt()
                except:
                    pass
            alltheory.append(np.array(theory_ind))
        #alltheory = np.transpose(alltheory)  
    alltheory = alltheory[0]
    M = np.array(M)
    sqrtM = np.sqrt(M)
    ###########################################################################
    kmax = np.array(kmax)
    kstd = np.array(kstd)
    # Fit Data
    if "PA" in mode:
        def fit1 (x, a, b):
            return a * x ** b
        (a, b), fit_cov = curve_fit(fit1, N, kmax, [m, 0.5], sigma = kstd, 
                                    absolute_sigma = True)
        # def fit1 (x, a, b):
        #     return b * x + a
        # (a, b), fit_cov = curve_fit(fit1, np.log(N), np.log(kmax), [m, 0.5], 
        #                             sigma = kstd / kmax, 
        #                             absolute_sigma = True)
        # can set bounds and constraints
        astd, bstd = np.sqrt(np.diag(fit_cov))
        print(f"Fit of k1, for m = {m}")
        print(f" - Amplitude was {a} +/- {astd}")
        print(f" - Exponent was  {b} +/- {bstd}")
    
    if "RA" in mode:
        def fit1 (x, a, b):
            return a * np.log(x) + b
        (a, b), fit_cov = curve_fit(fit1, N, kmax, [m, 2*m], sigma = kstd, 
                                    absolute_sigma = True)
        # def fit1 (x, a, b):
        #     return b * x + a
        # (a, b), fit_cov = curve_fit(fit1, np.log(N), np.log(kmax), [m, 0.5], 
        #                             sigma = kstd / kmax, 
        #                             absolute_sigma = True)
        # can set bounds and constraints
        astd, bstd = np.sqrt(np.diag(fit_cov))
        print(f"Fit of k1, for m = {m}")
        print(f" - Slope was {a} +/- {astd} vs {m}")
        print(f" - Offset was  {b} +/- {bstd} vs {m*(m0 - np.log((m0+m+1)/2))}" )
    
    if "RP" in mode:
        def fit1 (x, a, b, c):
            return a * x ** b + c * m
        (a, b, c), fit_cov = curve_fit(fit1, N, kmax, [m, 0.5, -m], sigma = kstd, 
                                    absolute_sigma = True)
        # def fit1 (x, a, b):
        #     return b * x + a
        # (a, b), fit_cov = curve_fit(fit1, np.log(N), np.log(kmax), [m, 0.5], 
        #                             sigma = kstd / kmax, 
        #                             absolute_sigma = True)
        # can set bounds and constraints
        astd, bstd, cstd = np.sqrt(np.diag(fit_cov))
        print(f"Fit of k1, for m = {m}")
        print(f" - Amplitude was {a} +/- {astd}")
        print(f" - Exponent was  {b} +/- {bstd}")
        print(f" - Offset was  {c} +/- {cstd}")
        
    ###########################################################################
    # Plot
    # if pl.fignum_exists(1):
    #     dont = True
    fig = pl.figure("Largest k")
    pl.errorbar(N, kmax, kstd, fmt = ".", capsize = 4, 
                label = f"Data, m = {m}", c = "black")
    if "PA" in mode:
        if not dont and fit:
            b_u = ufloat(b, bstd)
            pl.plot(Ns, fit1(Ns, a, b), ls = "--", 
                    label = f"Fit: $D_N$ = {b_u}", c = "black")
        else:
            pl.plot(Ns, fit1(Ns, a, b), ls = "--", c = "black")
    elif "RA" in mode:
        if not dont and fit:
            a_u = ufloat(a, astd)
            pl.plot(Ns, fit1(Ns, a, b), ls = "--", 
                    label = f"Slope fit: {a_u}", c = "black")
        else:
            pl.plot(Ns, fit1(Ns, a, b), ls = "--", c = "black")
    elif "RP" in mode:
        if not dont and fit:
            b_u = ufloat(b, bstd)
            pl.plot(Ns, fit1(Ns, a, b, c), ls = "--", 
                    label = f"Fit: $D_N$ = {b_u}", c = "black")
        else:
            pl.plot(Ns, fit1(Ns, a, b, c), ls = "--", c = "black")
    
    c = ["blue", "red", "darkorange", "green"]
    for i in range(len(alltheory)):
        zorder = 3
        if i == 0:
            zorder = 4
        if not dont:
            pl.plot(Ns, alltheory[i], ls = "--", #c = c[i],
                    label = "$k_1$ = " + alltags[i], zorder = zorder)
        else:
            pl.plot(Ns, alltheory[i], ls = "--", #c = c[i], 
                    zorder = zorder)
    
    pl.xlabel("$N$")
    pl.ylabel("$k_1$")
    pl.xscale("log")
    if "PA" or "RP" in mode:
        pl.yscale("log")
    pl.legend()
    pl.grid(alpha = 0.4)
    fig.tight_layout()
    pl.savefig(f"{mode}/plot_k1_m{m}_N{Tlist}.pdf")
    if "PA" not in mode:
        a, b, astd, bstd = None, None, None, None
    return np.array([a, b]), np.array([astd, bstd])


#############################################################################
#
#_____________________________________________________________________________
def plot_k1_m(mlist, N, mode = "PA", theory = [1,2,3], dont = False):
    kmax = []
    kstd = []
    alltheory = []
    alltags = []
    
    if mode == "PA" or mode == "RA":
        ms = np.linspace(min(mlist), max(mlist) + 1 )
    
    ###########################################################################
    # Get Experimental and Theoretical Maximum
    for i, m in enumerate(mlist):
        try:
            klists = np.load(f"{mode}/avklist_m{m}_N{int(N)}.npy")
        except FileNotFoundError:
            klists = np.load(f"{mode}/avklist_m{m}_N{int(N)}.npy")
        
        kmaxs_m = np.amax(klists, axis = 1)
        kmax.append(np.mean(kmaxs_m))
        kstd.append(np.std(kmaxs_m))
        theory_ind = []
        if i == 0:
            for j in theory:
                try:
                    f = globals()[f"{mode}_kmax{j}"]
                    t, tag = f(N, ms, True)
                    theory_ind.append(t)
                    alltags.append(tag)
                except KeyboardInterrupt:
                    raise KeyboardInterrupt()
                except:
                    pass
            alltheory.append(np.array(theory_ind))
        #alltheory = np.transpose(alltheory)  
    alltheory = alltheory[0]
    
    ###########################################################################
    kmax = np.array(kmax)
    kstd = np.array(kstd)
    # Fit Data
    if "PA" in mode:
        def fit1 (x, a, b):
            return a * x ** b
        (a, b), fit_cov = curve_fit(fit1, mlist, kmax, [np.sqrt(N), 0.5], 
                                    sigma = kstd, 
                                    absolute_sigma = True)
        # def fit1 (x, a, b):
        #     return b * x + a
        # (a, b), fit_cov = curve_fit(fit1, np.log(N), np.log(kmax), [m, 0.5], 
        #                             sigma = kstd / kmax, 
        #                             absolute_sigma = True)
        # can set bounds and constraints
        astd, bstd = np.sqrt(np.diag(fit_cov))
        print(f"Fit of k1, for N = {N}")
        print(f" - Amplitude was {a} +/- {astd}")
        print(f" - Exponent was  {b} +/- {bstd}")
    
    ###########################################################################
    # Plot
    # if pl.fignum_exists(1):
    #     dont = True
    fig = pl.figure()
    pl.errorbar(mlist, kmax, kstd/sqrtM, fmt = ".", capsize = 4, 
                label = f"Data, N = {N}", c = "black")
    if "PA" in mode:
        if not dont:
            b_u = ufloat(b, bstd)
            pl.plot(ms, fit1(ms, a, b), ls = "--", 
                    label = f"Fit: $D_m$ = {b_u}", c = "black")
        else:
            pl.plot(ms, fit1(ms, a, b), ls = "--", c = "black")
    
    c = ["blue", "red", "darkorange", "green"]
    for i in range(len(alltheory)):
        if not dont:
            pl.plot(ms, alltheory[i], ls = "--", c = c[i],
                    label = "$k_1$ = " + alltags[i])
        else:
            pl.plot(ms, alltheory[i], ls = "--", c = c[i])
    
    pl.xlabel("$m$")
    pl.ylabel("$k_1$")
    pl.xscale("log")
    if "PA" in mode:
        pl.yscale("log")
    pl.legend()
    fig.tight_layout()
    pl.savefig(f"{mode}/plot_k1_m{mlist}_N{N}.pdf")
    if "PA" not in mode:
        a, b, astd, bstd = None, None, None, None
    return np.array([a, b]), np.array([astd, bstd])


###############################################################################ù
#
#_____________________________________________________________________________
def k1_fit(mlist, Tlist, mode):
    mlist = np.array(mlist)
    Tlist = np.array(Tlist)
    
    kmax = []
    kstd = []
    flat = []
    
    if "PA" in mode:
        
        def all_fit(x, a, Dm, DN):
            m, N = x
            return a * m ** Dm * N ** DN
            
        for m in mlist:
            for N in Tlist:
                flat.append([m, N])
                try:
                    klists = np.load(f"{mode}/avklist_m{m}_N{int(N)}.npy")
                except FileNotFoundError:
                    klists = np.load(f"{mode}/avklist_m{m}_N{int(N)}.npy")
                kmaxs_m = np.amax(klists, axis = 1)
                kmax.append(np.mean(kmaxs_m))
                kstd.append(np.std(kmaxs_m))
        flat = np.array(flat)
        flat = np.transpose(flat)
                
        (a, Dm, DN), cov = curve_fit(all_fit, flat, kmax, p0 = [1, 0.5, 0.5],
                               sigma = kstd, absolute_sigma = True)
        astd, Dmstd, DNstd = np.sqrt(np.diag(cov))
        
        expected = []
        for m in mlist:
            for N in Tlist:
                expected.append(all_fit([m, N], a, Dm, DN))
        ddof = 3
        chi2, p = chisquare(kmax, expected, ddof)
        chi2_reduc = chi2 / (len(kmax)-1-ddof)
        print(f"Fit of k1:")
        print(f"- Amplitude was  {a} +/- {astd}")
        print(f"- m-Exponent was {Dm} +/- {Dmstd}")
        print(f"- N-Exponent was {DN} +/- {DNstd}")
        print(f"- Chi2red = {chi2_reduc},     pvalue = {p}")
        
    
    if "RA" in mode:
        
        def all_fit(x, a, Dm, DN):
            m, N = x
            return a * m ** Dm * N ** DN
            
        for m in mlist:
            for N in Tlist:
                flat.append([m, N])
                try:
                    klists = np.load(f"{mode}/avklist_m{m}_N{int(N)}.npy")
                except FileNotFoundError:
                    klists = np.load(f"{mode}/avklist_m{m}_N{int(N)}.npy")
                kmaxs_m = np.amax(klists, axis = 1)
                kmax.append(np.mean(kmaxs_m))
                kstd.append(np.std(kmaxs_m))
        flat = np.array(flat)
        flat = np.transpose(flat)
                
        (a, Dm, DN), cov = curve_fit(all_fit, flat, kmax, p0 = [1, 0.5, 0.5],
                               sigma = kstd, absolute_sigma = True)
        astd, Dmstd, DNstd = np.sqrt(np.diag(cov))
        
        expected = []
        for m in mlist:
            for N in Tlist:
                expected.append(all_fit(m, N, a, Dm, DN))
        ddof = 3
        chi2, p = chisquare(kmax, expected, ddof)
        chi2_reduc = chi2 / (len(observed)-1-ddof)
        print(f"Fit of k1:")
        print(f"- Amplitude was  {a} +/- {astd}")
        print(f"- m-Exponent was {Dm} +/- {Dmstd}")
        print(f"- N-Exponent was {DN} +/- {DNstd}")
        print(f"- Chi2red = {chi2_reduc},     pvalue = {pvalue}")
        
        return (a, Dm, DN), (astd, Dmstd, DNstd)
    
    
    if "RP" in mode:
        
        def all_fit(x, a, Dm, DN, c, d):
            m, N = x
            return a * m ** Dm * N ** DN - c * m + d
            
        for m in mlist:
            for N in Tlist:
                flat.append([m, N])
                try:
                    klists = np.load(f"{mode}/avklist_m{m}_N{int(N)}.npy")
                except FileNotFoundError:
                    klists = np.load(f"{mode}/avklist_m{m}_N{int(N)}.npy")
                kmaxs_m = np.amax(klists, axis = 1)
                kmax.append(np.mean(kmaxs_m))
                kstd.append(np.std(kmaxs_m))
        flat = np.array(flat)
        flat = np.transpose(flat)
                
        (a, Dm, DN, c, d), cov = curve_fit(all_fit, flat, kmax, p0 = [1, 0.5, 0.5, 10, 10],
                               sigma = kstd, absolute_sigma = True)
        astd, Dmstd, DNstd, cstd, dstd = np.sqrt(np.diag(cov))
        
        expected = []
        for m in mlist:
            for N in Tlist:
                expected.append(all_fit([m, N], a, Dm, DN, c, d))
        ddof = 5
        chi2, p = chisquare(kmax, expected, ddof)
        chi2_reduc = chi2 / (len(kmax)-1-ddof)
        print(f"Fit of k1:")
        print(f"- Amplitude was  {a} +/- {astd}")
        print(f"- m-Exponent was {Dm} +/- {Dmstd}")
        print(f"- N-Exponent was {DN} +/- {DNstd}")
        print(f"- m-Offset    was {c} +/- {cstd}")
        print(f"- d-Offset    was {d} +/- {dstd}")
        print(f"- Chi2red = {chi2_reduc},     pvalue = {p}")
                
                

