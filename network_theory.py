# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 18:32:31 2022

@author: Stefano
"""
import numpy as np

#%% PRFERENTIAL ATTACHMENT
#################################################
def PA_pk1 (k, m, mscale = False):
    """
    Power Law decay of p(k).

    Parameters
    ----------
    m : int    
    k : ndarray
        k values   
    g : float, optional
        Gamma coefficient, s.t. p(k)=Ak^(-3). The default is 3.

    Returns
    -------
    pk1 : ndarray
        p(k) values.
    """
    if hasattr(k, "__len__"):
        k = np.array(k, dtype = float)
        if not mscale:
            A = 2 * m **3 / (2 + m)
            pk = A * k ** (-3)
        else:
            pk = k ** (-3)
        return np.where(k>=m, pk, 0)
    else:
        k = float(k)
        if k<m:
            return 0
        if not mscale:
            A = 2 * m **3 / (2 + m)
            pk = A * k ** (-3)
        else:
            pk = k ** (-3)
        return pk


def PA_cdfk1 (k, m, g = 3):
    """
    Cumulative density function of p(k).

    Parameters
    ----------
    m : int   
    k : ndarray
        k values
    g : float, optional
        Gamma coefficient, s.t. p(k)=Ak^(-3). The default is 3.

    Returns
    -------
    cdfk : ndarray
        Cumulative density function at k coordinates of p(k)=Ak^(-3).
    """
    if hasattr(k, "__len__"):
        k = np.array(k)
        cdfk = np.zeros(len(k))
        for i in range(len(k)):
            cdfk[i] = cdfk[i-1] + PA_pk1(k[i], m)
        return np.where(k>=m, cdfk, 0)
    else:
        cdfk[i] = cdfk[i-1] + PA_pk1(k[i], m)
        if k<m:
            return 0
        return cdfk

def PA_pk (k, m, mscale = False):
    """
    More precise p(k).

    Parameters
    ----------
    k : ndarray
        k values
    m : int
            
    Returns
    -------
    pk : ndarray
        p(k) values A/(k*(k+1)*(k+2)), A = 2m(m+1)
    """
    if hasattr(k, "__len__"):
        k = np.array(k)
        A = 2 * m * (m+1)
        if mscale:
            A = 1
        pk = A / k
        pk /= (k+1)
        pk /= (k+2)
        if not mscale:
            return np.where(k>=m, pk, 0)
        else:
            return pk
    else:
        if k<m and not mscale:
            return 0
        A = 2 * m * (m+1)
        if mscale:
            A = 1
        pk = A / k
        pk /= (k+1)
        pk /= (k+2)
        return pk


def PA_cdfk (k, m):
    """
    More precise cumulative density function of p(k).

    Parameters
    ----------
    m : int
        
    k : ndarray
        k values

    Returns
    -------
    cdfk : ndarray
        cumulative density function at k coordinates of 
        p(k) = A/(k*(k+1)*(k+2)), A = 2m(m+1)
        which is
        cdf(k) = 1 - (m^2+m)/((k+1)(k+2))
    """
    if hasattr(k, "__len__"):
        k = np.array(k)
        A = m * (m+1)
        d = (k+1) * (k+2)
        cdfk = 1 - A/d
        return np.where(k>=m, cdfk, 0)
    else:
        if k<m:
            return 0
        A =  m * (m+1)
        d = (k+1) * (k+2)
        cdfk = 1 - A/d
        return cdfk
    

def PA_kmax1(N, m, tag = False, m0 = 1):
    """
    k1 = k0 * sqrt(N/(N0))
    Parameters
    ----------
    N : int or ndarray
    m : int or ndarray
    Returns
    -------
    kmax : float or ndarray
    """
    lencheck = 0
    if hasattr(N, "__len__"):
        N = np.array(N)
        lencheck += 1 
    if hasattr(m, "__len__"):
        m = np.array(m)
        lencheck += 1 
    if lencheck == 2:
        n, M = np.meshgrid([m, N])
    kmax = (m * m0 +1)* np.sqrt( N / (m0*m+1) )
    if tag:
        return kmax, r"$k_{0} \times \sqrt{N/N_0}$"
    return kmax


def PA_kmax2(N, m, tag = False, m0 = 1):
    """
    k1 = k0 * sqrt(N/sqrt(N0))
    Parameters
    ----------
    N : int or ndarray
    m : int or ndarray
    Returns
    -------
    kmax : float or ndarray
    """
    lencheck = 0
    if hasattr(N, "__len__"):
        N = np.array(N)
        lencheck += 1 
    if hasattr(m, "__len__"):
        m = np.array(m)
        lencheck += 1 
    if lencheck == 2:
        n, M = np.meshgrid([m, N])
    kmax = (m * m0 +1)* np.sqrt( N / np.sqrt(m0*m+1) )
    if tag:
        return kmax, r"$k_{0} \times \sqrt{N/ \sqrt{N_0} }$"
    return kmax

def PA_kmax3(N, m, tag = False, m0 = 1):
    """
    k1 = k0 * sqrt(N/ (m+1) )
    Parameters
    ----------
    N : int or ndarray
    m : int or ndarray
    Returns
    -------
    kmax : float or ndarray
    """
    lencheck = 0
    if hasattr(N, "__len__"):
        N = np.array(N)
        lencheck += 1 
    if hasattr(m, "__len__"):
        m = np.array(m)
        lencheck += 1 
    if lencheck == 2:
        n, M = np.meshgrid([m, N])
    kmax = (m * m0 +1) * np.sqrt( 2*N / ((m * m0 +1)) )
    if tag:
        return kmax, r"$k_{0} \times \sqrt{2N/N_0}$"
    return kmax


def PA_kmax4(N, m, tag = False, m0 = 1):
    """
    k1 = sqrt(N * m * (m+1) )
    Parameters
    ----------
    N : int or ndarray
    m : int or ndarray
    Returns
    -------
    kmax : float or ndarray
    """
    lencheck = 0
    if hasattr(N, "__len__"):
        N = np.array(N)
        lencheck += 1 
    if hasattr(m, "__len__"):
        m = np.array(m)
        lencheck += 1 
    if lencheck == 2:
        n, M = np.meshgrid([m, N])
    kmax = np.sqrt( N * m * (m+1) )
    if tag:
        return kmax, r"$\sqrt{Nm(m+1)}$"
    return kmax

# def PA_kmax4(N, m, tag = False):
#     """
#     k1 = cbrt(2 * N * m * (m+1) )
#     Parameters
#     ----------
#     N : int or ndarray
#     m : int or ndarray
#     Returns
#     -------
#     kmax : float or ndarray
#     """
#     lencheck = 0
#     if hasattr(N, "__len__"):
#         N = np.array(N)
#         lencheck += 1 
#     if hasattr(m, "__len__"):
#         m = np.array(m)
#         lencheck += 1 
#     if lencheck == 2:
#         n, M = np.meshgrid([m, N])
#     kmax = np.cbrt( 2 * N * m * (m+1) )
#     if tag:
#         return kmax, r"$\sqrt[3]{2Nm(m+1)}$"
#     return kmax
    

def avgD (N):
    return np.log(N) / np.log(np.log(N))

def avgC (m, N, N0):
    A = (m+1)**2 * (np.log(N-N0))**2
    B = 8*m*np.log(N-N0) + 8*m
    D = 8 * (m-1) * (6*m**2+8*m+3) * (N-N0)
    return 6*m**2 * (A-B) / D  

def triangles (m, N):
    A = m * (m-1) * (m+1)
    return A/48 * (np.log(N))**3


    
    
#%% RANDOM ASSIGNMENT
#################################################
def RA_pk (k, m):
    """
    Parameters
    ----------
    k : ndarray
        k values
    m : int
            
    Returns
    -------
    pk : ndarray
        p(k) values A/(k*(k+1)*(k+2)), A = 2m(m+1)
    """
    if hasattr(k, "__len__"):
        k = np.array(k)
        A = 1 / (m+1)
        B = A * m
        pk = A * B**(k-m)
        return np.where(k>=m, pk, 0) 
    else:
        if k<m:
            return 0
        A = 1 / (m+1)
        B = A * m
        pk = A * B**(k-m)
        return pk 


def RA_cdfk (k, m):
    """
    Parameters
    ----------
    k : ndarray
        k values
    m : int
            
    Returns
    -------
    pk : ndarray
        p(k) values A/(k*(k+1)*(k+2)), A = 2m(m+1)
    """
    if hasattr(k, "__len__"):
        k = np.array(k)
        A = m / (m+1)
        pk = A**(k-m+1)
        return 1 - np.where(k>=m, pk, 1) 
    else:
        if k<m:
            return 0
        A = m / (m+1)
        pk = A**(k-m+1)
        return 1 - pk 
    

def RA_kmax1(N, m, tag = False, m0 = 1):
    """
    k1 = k0 + m * ln( N/(N0) )
    Parameters
    ----------
    N : int or ndarray
    m : int or ndarray
    Returns
    -------
    kmax : float or ndarray
    """
    lencheck = 0
    if hasattr(N, "__len__"):
        N = np.array(N)
        lencheck += 1 
    if hasattr(m, "__len__"):
        m = np.array(m)
        lencheck += 1 
    if lencheck == 2:
        n, M = np.meshgrid([m, N])
    kmax = m0*m + m * np.log( N / (m0*m+1) )
    if tag:
        return kmax, r"$ k_0 + m \ln (N/ {N_0})$"
    return kmax


def RA_kmax2(N, m, tag = False, m0 = 1):
    """
    k1 = m + m * ln( N/sqrt(N0) )
    Parameters
    ----------
    N : int or ndarray
    m : int or ndarray
    Returns
    -------
    kmax : float or ndarray
    """
    lencheck = 0
    if hasattr(N, "__len__"):
        N = np.array(N)
        lencheck += 1 
    if hasattr(m, "__len__"):
        m = np.array(m)
        lencheck += 1 
    if lencheck == 2:
        n, M = np.meshgrid([m, N])
    kmax = m0*m + m * np.log( N / np.sqrt(m0*m+1) )
    if tag:
        return kmax, r"$ k_0 + m \ln (N/ \sqrt{N_0})$"
    return kmax


def RA_kmax3(N, m, tag = False, m0 = 1):
    """
    k1 = m0 + m * ln( 2N/N0 )
    Parameters
    ----------
    N : int or ndarray
    m : int or ndarray
    Returns
    -------
    kmax : float or ndarray
    """
    lencheck = 0
    if hasattr(N, "__len__"):
        N = np.array(N)
        lencheck += 1 
    if hasattr(m, "__len__"):
        m = np.array(m)
        lencheck += 1 
    if lencheck == 2:
        n, M = np.meshgrid([m, N])
    kmax = m0*m + m * np.log( 2*N / (m0*m+1) )
    if tag:
        return kmax, r"$ k_0 + m \ln (2N/N_0)$"
    return kmax


def RA_kmax4(N, m, tag = False, m0 = 1):
    """
    k1 = m + ln(N) / ln(1 + 1/m)
    Parameters
    ----------
    N : int or ndarray
    m : int or ndarray
    Returns
    -------
    kmax : float or ndarray
    """
    lencheck = 0
    if hasattr(N, "__len__"):
        N = np.array(N)
        lencheck += 1 
    if hasattr(m, "__len__"):
        m = np.array(m)
        lencheck += 1 
    if lencheck == 2:
        n, M = np.meshgrid([m, N])
    kmax = m + np.log(N) / np.log(1 + 1/m)
    if tag:
        return kmax, r"$ m + \frac {\ln(N)}{\ln(1+1/m)} $"
    return kmax


#%% EXISTING VERTICES ASSIGNMENT
#################################################

def RP_pk(k, m, r = 0.5):
    """
    Parameters
    ----------
    k : ndarray
        k values
    m : int
            
    Returns
    -------
    pk : ndarray
        p(k) values A/((k+m)*(k+m+1)*(k+m+2)), A = 3m(3m+2)/2
    """
    if r == m/2 or r == 0.5:  
        r = int(m/2)
        if hasattr(k, "__len__"):
            A = 3 * m * (3*m+2) / 2
            b = k+m
            pk = A / b /(b+1) /(b+2)
            return np.where(k>=r, pk, 0) 
        else:
            if k< m/2:
                return 0
            A = 3 * m * (3*m+2) / 2
            b = k+m
            pk = A / b /(b+1) /(b+2)
            return pk 


def RP_cdfk(k, m, r = 0.5):
    """
    Parameters
    ----------
    k : ndarray
        k values
    m : int
            
    Returns
    -------
    pk : ndarray
        cdfk(k) values A/(k*(k+1)*(k+2)), A = 2m(m+1)
    """
    if r == m/2 or r == 0.5:  
        r = int(m/2)
        if hasattr(k, "__len__"):
            k = np.array(k)
            A = 3 * m * (3*m+2) / 4
            b = k+m
            cdfk = 1 - A / (b+1) /(b+2)
            return np.where(k>=r, cdfk, 0) 
        else:
            if k< m/2:
                return 0
            A = 3 * m * (3*m+2) / 4
            b = k+m
            cdfk = 1 - A / (b+1) /(b+2)
            return cdfk 
   
        
def RP_kmax1(N, m, tag = False, m0 = 1):
    """
    For r = m/2
    Parameters
    ----------
    N : int or ndarray
    m : int or ndarray
    Returns
    -------
    kmax : float or ndarray
    """
    lencheck = 0
    if hasattr(N, "__len__"):
        N = np.array(N)
        lencheck += 1 
    if hasattr(m, "__len__"):
        m = np.array(m)
        lencheck += 1 
    if lencheck == 2:
        n, M = np.meshgrid([m, N])
    kmax = (1 + m0) * m * np.sqrt(N/(m0*m+1)) - m
    if tag:
        return kmax, r"$ (m+k_0) \sqrt{ N/N_0} - m $"
    return kmax


def RP_kmax2(N, m, tag = False, m0 = 1):
    """
    For r = m/2
    Parameters
    ----------
    N : int or ndarray
    m : int or ndarray
    Returns
    -------
    kmax : float or ndarray
    """
    lencheck = 0
    if hasattr(N, "__len__"):
        N = np.array(N)
        lencheck += 1 
    if hasattr(m, "__len__"):
        m = np.array(m)
        lencheck += 1 
    if lencheck == 2:
        n, M = np.meshgrid([m, N])
    kmax = (1 + m0) * m * np.sqrt( 2 * N/(m0*m+1)) - m
    if tag:
        return kmax, r"$ (m+k_0) \sqrt{2N/N_0} - m $"
    return kmax

def RP_kmax3(N, m, tag = False, m0 = 1):
    """
    For r = m/2
    Parameters
    ----------
    N : int or ndarray
    m : int or ndarray
    Returns
    -------
    kmax : float or ndarray
    """
    lencheck = 0
    if hasattr(N, "__len__"):
        N = np.array(N)
        lencheck += 1 
    if hasattr(m, "__len__"):
        m = np.array(m)
        lencheck += 1 
    if lencheck == 2:
        n, M = np.meshgrid([m, N])
    kmax = (1 + m0) * m * np.sqrt(N/np.sqrt(m0*m+1)) - m
    if tag:
        return kmax, r"$ (m+k_0) \sqrt{N/ \sqrt{N_0}} - m $"
    return kmax


def RP_kmax4(N, m, tag = False, m0 = 1):
    """
    For r = m/2
    Parameters
    ----------
    N : int or ndarray
    m : int or ndarray
    Returns
    -------
    kmax : float or ndarray
    """
    lencheck = 0
    if hasattr(N, "__len__"):
        N = np.array(N)
        lencheck += 1 
    if hasattr(m, "__len__"):
        m = np.array(m)
        lencheck += 1 
    if lencheck == 2:
        n, M = np.meshgrid([m, N])
    kmax = np.sqrt(3 / 4 * N * m * (3*m+2))
    if tag:
        return kmax, r"$ \sqrt { \frac{3}{4} N m (3m+2)} $"
    return kmax










