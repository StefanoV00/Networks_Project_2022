# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:59:48 2022

@author: Stefano
"""
import numpy as np
import networkx as nx
import random as rand

from tqdm import tqdm
from numba import njit
from timeit import default_timer as timer 

class BAnetwork ():
    """
    My Class for modelling a Barabasi Albert Network.
    """
    counter = 0 #counter of instances
    
    def __init__(self, m = 1, mode = "auto", Ninit = 0, Einit = 0, kinit = 0,
                 seed = float("nan")):
        
        if m != int(m):
            raise Exception ("Heyyy, m must be an integer!")
        else:
            m = int(m)
            self._m = m
        
        if not np.isnan(seed):
            rand.seed(seed)
        
        if mode == "auto" or mode == "Auto" or mode == "PA" or mode == "RA" \
            or "RP" in mode:
            #produce the smallest N graph with m edges per node (all connected)
            self._N = int(2 * m + 1)
            self._E = int(self._N * (self._N - 1) / 2)
            self._adjlist = [ [i for i in range(self._N) if i!=j] 
                              for j in range(self._N) ]
            self._klist = [k for k in [self._N - 1]*self._N]
            self._attachlist = list(range(self._N))*2*self._m
            self._eset = set()
            for i in range(self._N):
                for j in self._adjlist[i]:
                    if j>i:
                        self._eset.add((i,j))
        
        elif mode == "PAalt" or mode == "RAalt":
            #produce the smallest N graph with m edges per node (all connected)
            self._N = int(m + 1)
            self._E = int(self._N * (self._N - 1) / 2)
            self._adjlist = [ [i for i in range(self._N) if i!=j] 
                              for j in range(self._N) ]
            self._klist = [k for k in [self._N - 1]*self._N]
            self._attachlist = list(range(self._N))*self._m
        
        elif mode == "RP" or mode == "RPmulti" or mode == "RPalt":
            #produce a graph with m edges per node and N = 2*(2*m+1)
            self._N = int(2 * m + 1)
            self._E = int(self._N * m)
            
            G = BAnetwork(m, "auto")
            G.grow(T = int(2*m+1), m = m)
            
            self._N = G.getN()
            self._E = G.getE()
            self._adjlist = G.get_adjlist()
            self._klist = G.get_klist().tolist()
            self._attachlist = G.get_attachlist()
            self._eset = set()
            for i in range(self._N):
                for j in self._adjlist[i]:
                    if j>i:
                        self._eset.add((i,j))
        
        else:
            raise Exception("no mode given")
         
        self.check()
        self._t = 0
        self._NEinit = [self._N * 1, self._E * 1]
        self.counter += 1
        


    def __str__ (self):
        return "My Class for modelling a Barabasi Albert Network"
    
    def check(self, mode = "PA"):
        """
        If adjlist and elist don't agree with each other, or if two sublists
        in the adjlist don't agree with each other, an exception is returned.
        
        Return
        ----------
        
        check : list
            If everything went alright, it will be [True], otherwise
            [False, str1, str2, ...].
        """
        #check for coherence between edgelist and adjacency list
        a = 0
        e = 0
        check = []
        N = self._N
        Nadj = len(self._adjlist)
        Nk = len(self._klist)
        if N != Nadj:
            check.append("len(adjlist) is not N !")
        if N != Nk:
            check.append("len(klist) is not N !")
        if Nadj != Nk:
            check.append("len(adjlist) != len(klist) !")
        for i in range(N):
            if len(self._adjlist[i]) != self._klist[i]:
                check.append("Degree ki doens't agree with len(adjlist_i) !")
        ktot = sum(self._klist)
        if ktot != 2*self._E:
            check.append("sum(klist) != 2E !!")
        if 0 in self._klist:
            check.append("There is an empty node!!!!")
        
        if mode != "RA":
            if max(self._attachlist) != N -1:
                check.append("max(attachlist) is different than N -1 !")
            La = len(self._attachlist)
            if La != ktot:
                check.append("sum(klist) != len(attachlist) !!")
            if La !=  2*self._E:
                check.append("len(attachlist) != 2E !!")
        if len(check) == 0:
            check = [True]
        else:
            check = [False, check]
        return check
                
    
    
    ########################################################################
    # PROPER NETWORK METHODS
    def add (self, m = 0, mode = "PA", r = 0, seed = 0):
        """
        Add new vertex, connect it with m old ones, update networks attributes.
        Also returns copy of updated network. 
        
        Parameters
        ----------
        m : int, default 0
            Number of edges departing from new node. The default is 0, meaning
            self._m is used as m.  
        mode : str or int, default "PA"
            Defines how the choice of which nodes to connect the new one to is 
            made. The options are:\n
                -"PA" or 1, for preferential attachment (default).\n
                -"RA" or 2, for random attachment.\n
                -"PR" or 3.1, for existing nodes attachment (see r).\n
                -"RP" or 3.2, for existing nodes attachment, version 2.\n
                -"RPalt" or 3.3, for RP alternative version. \n
                -"RPmulti" or 3.4, for RP multigraph version. \n
        r : int, default 0
           -For r of new edges, connect one end of the new edge to the new 
            node and the other to an existing i chosen with\n
            - probability propto k_i for "PR".\n
            - equal probability for "RP".\n
           -For remaining (m − r) of the edges connect both ends to 2
            existing nodes, each chosen with probability:
            - equal probability for "PR".\n
            - probability propto k_i for "RP".\n
        seed : int
            For debugging purposes.
        """
        
        if hasattr(m, "__len__"):
            m = rand.randint(m[0], m[1])
        elif callable(m):
            m = m()
        elif m == 0: 
            m = self._m
        self._t += 1
        
        if "PA" in mode:
            self.PAassign(m, seed)
            
        elif "RA" in mode:
            self.RAassign(m, seed)
            
        elif "PR" in mode:
            self.PRassign(m, r, seed)
            
        elif mode == "RP":
            self.RPassign(m, r, seed)
        
        elif mode == "RPalt":
            self.RPassign_alt(m, r, seed)
            
        elif mode == "RPmulti":
            self.RPassign_multi(m, r, seed)
        
        else:
            raise ValueError("The given mode isn't any of:\
                                \n -'PA'\
                                \n -'RA'\
                                \n -'PR'\
                                \n -'RP'\
                                \n -'RPalt'\
                                \n -'RPmulti'")
        
        
    def grow(self, T = 100, m = 0, mode = "PA", r = 0, seed = 0):
        """
        Perform addition of new vertex T times.
        
        Parameters
        ----------
        m : int or 2,tuple, default 0 
            Number of edges departing from new node. The default is 0, meaning
            self._m is used as m. If 2,tuple, then choose randomly uniformly 
            between m[0] and m[1], both included. 
        mode : str or int, default "PA"
            Defines how the choice of which nodes to connect the new one to is 
            made. The options are:\n
                -"PA" or 1, for preferential attachment (default).\n
                -"RA" or 2, for random attachment.\n
                -"PR" or 3.1, for existing nodes attachment (see r).\n
                -"RP" or 3.2, for existing nodes attachment, version 2.\n
                -"RPalt" or 3.3, for RP alternative version. \n
                -"RPmulti" or 3.4, for RP multigraph version. \n
        r : int, default 0
           -For r of new edges, connect one end of the new edge to the new 
            node and the other to an existing i chosen with\n
            - probability propto k_i for "PR".\n
            - equal probability for "RP".\n
           -For remaining (m − r) of the edges connect both ends to 2
            existing nodes, each chosen with\n
            - equal probability for "PR".\n
            - probability propto k_i for "RP".\n
        seed : int, default 0
            For debugging purposes. If 0, not used.
        """

        for t in range(T):
            self.add(m, mode, r, seed)
    
    def networkX (self):
        """
        Return networkx copy of graph.
        """
        # elist = []
        # for i in range(self._N):
        #     for j in self._adjlist[i]:
        #         if j>i:
        #             elist.append([i,j])
        new_adjlist = dict()
        for i in range(self._N):
            new_adjlist[f"{i}"] = {}
            for j in self._adjlist[i]:
                new_adjlist[f"{i}"][f"{j}"] = {}
        G = nx.Graph(new_adjlist)
        return G
    
    def draw(self, **kwds):
        G = self.networkX(**kwds)
        nx.draw(G)
    
    def draw_networkx(self, pos=None, arrows=None, with_labels=True, **kwds):
        G = self.networkX()
        nx.draw_networkx(G, pos, arrows, with_labels, **kwds)
        

    
    ########################################################################
    #ALL THE STANDARD GET, SET AND ADD METHODS FOR THE ATTRIBUTES
    def get_NEinit (self):
        return 1* self._NEinit 
    
    def getm (self):
        return 1* self._m
    
    def getN (self):
        return 1* self._N  
    
    def getE (self):
        return 1* self._E 
     
    def get_attachlist (self):
        return 1* self._attachlist
    
    def get_adjlist (self):
        return 1* self._adjlist
    
    def get_klist (self):
        return np.array(self._klist)  
    
    def get_counter (self):
        return 1* self.counter  
    
    def gett (self):
        return 1* self._t  
    
    def setm (self, m):
        self._m = m
        
    def set_adjlist (self, al):
        self._adjlist = al
        self._N = len(al)
        
    def sett (self, T):
        self._t = T
        
    # def get_elist(self):
    #     return 1* self._elist
    
    # def set_elist (self, el):
    #     self._elist = el
    #     self._E = len(el)    
    
    # def add_edge(self, edge):
    #     if edge not in self._elist:
    #         self._elist.append(edge)
    #         self._adjlist[edge[0]].append(edge[1])
    #         self._adjlist[edge[1]].append(edge[0])
    
    # def add_edges_from(self, el):
    #     for edge in el:
    #         if edge not in self._elist:
    #             self._elist.append(edge)
    #             self._adjlist[edge[0]].append(edge[1])
    #             self._adjlist[edge[1]].append(edge[0])



    ##########################################################################
    #ASSIGNMENT FUNCTIONS FOR NETWORK
    def PAassign(self, m = 0, seed = 0):
        if seed:
            rand.seed(seed)
        chosen = []
        count = 0
        while count < m:
            r = rand.random()
            ind = self._attachlist[int(2 * self._E * r)]
            if ind not in chosen:
                chosen.append(ind) 
                #Update Old Nodes' Info
                #try:
                self._attachlist.append(ind)
                self._adjlist[ind].append(self._N)
                self._klist[ind] += 1
                # except:
                #     print("r = ", r)
                #     print("i = ", ind, "as 2E = ", 2*E)
                #     print("2E =", sum(klist), " = ", len(attachlist))
                #     print(N, " = ", len(adjlist), " = ", len(klist))
                #     raise Exception ("Booh!")
                count += 1
        #Update With New Node's Info
        self._attachlist.extend( [self._N] * m)
        self._adjlist.append(chosen)
        self._klist.append(m)
        self._N += 1
        self._E += m
    
    def RAassign(self, m=0, seed = 0):
        if seed:
            rand.seed(seed)
        chosen = []
        count = 0
        while count < m:
            r = rand.random()
            ind = int(self._N * r)
            if ind not in chosen:
                chosen.append(ind) 
                #Update Old Nodes' Info
                #self._attachlist.append(ind)
                self._adjlist[ind].append(self._N)
                self._klist[ind] += 1
                count += 1
        #Update With New Node's Info
        #self._attachlist.extend( [self._N]*m )
        self._adjlist.append(chosen)
        self._klist.append(m)
        self._N += 1
        self._E += m
    
    
    def PRassign(self, m = 0, r = 0, seed = 0):
        if seed:
            rand.seed(seed)
        # First PA attachment of r edges to new node
        chosen = set()
        count = 0
        while count < r:
            rf = rand.random()
            ind = self._attachlist[int(2 * self._E * rf)]
            if ind not in chosen:
                count += 1
                chosen.add(ind) 
                #Update Old Nodes' Info
                self._attachlist.append(ind)
                self._adjlist[ind].append(self._N)
                self._klist[ind] += 1
                self._eset.add((self._N, ind))
        #Update With New Node's Info
        self._attachlist.extend([self._N]*r)
        self._adjlist.append(chosen)
        self._klist.append(r)
        # Then RA attachment of m-r between existing nodes
        while count < m:
            r1 = rand.random(); ind1 = int(self._N * r1)
            r2 = rand.random(); ind2 = int(self._N * r2)
            while ind1 == ind2:
                r2 = rand.random(); ind2 = int(self._N * r2)
            if (ind1, ind2) not in self._eset and (ind2, ind1) not in self._eset: 
                count += 1
                #Update Old Nodes' Info
                self._attachlist.extend([ind1, ind2])
                self._adjlist[ind1].append(ind2)
                self._adjlist[ind2].append(ind1)
                self._klist[ind1] += 1
                self._klist[ind2] += 1
                self._eset.add((ind1, ind2))
        #Update With New Node's Info (part 2)
        self._N += 1
        self._E += m
                
    
    def RPassign(self, m=0, r = 0, seed=0):
        if seed:
            rand.seed(seed)
        # First RA attachment of r edges to new node
        chosen = set()
        count = 0
        while count < r:
            rf = rand.random()
            ind = int(self._N * rf)
            if ind not in chosen:
                chosen.add(ind) 
                #Update Old Nodes' Info
                self._attachlist.append(ind)
                self._adjlist[ind].append(self._N)
                self._klist[ind] += 1
                self._eset.add((self._N, ind))
                count += 1
        #Update With New Node's Info
        self._attachlist.extend([self._N]*r)
        self._adjlist.append(list(chosen))
        self._klist.append(r)
        # Then PA attachment of m-r between existing nodes
        #chosen = set()
        while count < m:
            r1= rand.random() 
            r2= rand.random() 
            ind1 = self._attachlist[int(2 * self._E * r1)]
            ind2 = self._attachlist[int(2 * self._E * r2)]
            while ind2 == ind1:
                r2 = rand.random()
                ind2 = self._attachlist[int(2 * self._E * r2)]
            if (ind1, ind2) not in self._eset and (ind2, ind1) not in self._eset: 
                count += 1
                #Update Old Nodes' Info
                self._attachlist.extend([ind1, ind2])
                self._adjlist[ind1].append(ind2)
                self._adjlist[ind2].append(ind1)
                self._klist[ind1] += 1
                self._klist[ind2] += 1
                self._eset.add((ind1, ind2))
        #Update With New Node's Info (part 2)
        self._N += 1
        self._E += m  

    
    def RPassign_alt(self, m=0, r = 0, seed=0):
        if seed:
            rand.seed(seed)
        # First RA attachment of r edges to new node
        chosen = set()
        count = 0
        while count < r:
            rf = rand.random()
            ind = int(self._N * rf)
            if ind not in chosen:
                chosen.add(ind) 
                #Update Old Nodes' Info
                self._attachlist.append(ind)
                self._adjlist[ind].append(self._N)
                self._klist[ind] += 1
                self._eset.add((self._N, ind))
                count += 1
        #Update With New Node's Info
        self._attachlist.extend([self._N]*r)
        self._adjlist.append(list(chosen))
        self._klist.append(r)
        # Then PA attachment of m-r between existing nodes
        #chosen = set()
        while count < m:
            r1= rand.random() 
            r2= rand.random() 
            ind1 = self._attachlist[int(2 * self._E * r1)]
            while self._klist[ind] >= self._N - 1:
                ind1 = self._attachlist[int(2 * self._E * r1)]
            ind2 = self._attachlist[int(2 * self._E * r2)]
            while ind2 == ind1 or (ind1, ind2) in self._eset \
            or (ind2, ind1) in self._eset:
                r2 = rand.random()
                ind2 = self._attachlist[int(2 * self._E * r2)] 
            count += 1
            #Update Old Nodes' Info
            self._attachlist.extend([ind1, ind2])
            self._adjlist[ind1].append(ind2)
            self._adjlist[ind2].append(ind1)
            self._klist[ind1] += 1
            self._klist[ind2] += 1
            self._eset.add((ind1, ind2))
        #Update With New Node's Info (part 2)
        self._N += 1
        self._E += m
        
    
    def RPassign_multi(self, m=0, r = 0, seed=0):
        if seed:
            rand.seed(seed)
        # First RA attachment of r edges to new node
        chosen = set()
        count = 0
        while count < r:
            rf = rand.random()
            ind = int(self._N * rf)
            if ind not in chosen:
                chosen.add(ind) 
                #Update Old Nodes' Info
                self._attachlist.append(ind)
                self._adjlist[ind].append(self._N)
                self._klist[ind] += 1
                #self._eset.add((self._N, ind))
                count += 1
        #Update With New Node's Info
        self._attachlist.extend([self._N]*r)
        self._adjlist.append(list(chosen))
        self._klist.append(r)
        # Then PA attachment of m-r between existing nodes
        #chosen = set()
        while count < m:
            r1= rand.random() 
            r2= rand.random() 
            ind1 = self._attachlist[int(2 * self._E * r1)]
            ind2 = self._attachlist[int(2 * self._E * r2)]
            while ind2 == ind1:
                r2 = rand.random()
                ind2 = self._attachlist[int(2 * self._E * r2)] 
            count += 1
            #Update Old Nodes' Info
            self._attachlist.extend([ind1, ind2])
            self._adjlist[ind1].append(ind2)
            self._adjlist[ind2].append(ind1)
            self._klist[ind1] += 1
            self._klist[ind2] += 1
            #self._eset.add((ind1, ind2))
        #Update With New Node's Info (part 2)
        self._N += 1
        self._E += m
        
        
        
    #######################################################################
    #"STATIC" ASSIGNMENT FUNCTIONS FOR NETWORK
    # Useuful for debugging and testing, they do NOT update anything but
    # the klist. 
    # As they don't upload E, the only elements considered to be picked are
    # the "orginal" ones, no matter how many times you run it.
    def PAassign_static(self, m = 0, seed = 0):
        if seed:
            rand.seed(seed)
        chosen = []
        count = 0
        while count < m:
            r = rand.random()
            ind = self._attachlist[int(2 * self._E * r)]
            if ind not in chosen:
                chosen.append(ind) 
                #Update Old Nodes' Info
                #try:
                self._klist[ind] += 1
                # except:
                #     print("r = ", r)
                #     print("i = ", ind, "as 2E = ", 2*E)
                #     print("2E =", sum(klist), " = ", len(attachlist))
                #     print(N, " = ", len(adjlist), " = ", len(klist))
                #     raise Exception ("Booh!")
                count += 1
        
    
    def RAassign_static(self, m=0, seed = 0):
        if seed:
            rand.seed(seed)
        chosen = []
        count = 0
        while count < m:
            r = rand.random()
            ind = int(self._N * r)
            if ind not in chosen:
                chosen.append(ind) 
                #Update Old Nodes' Info
                self._klist[ind] += 1
                count += 1
    
    
    def PRassign_static(self, m = 0, r = 0, seed = 0):
        if seed:
            rand.seed(seed)
        # First PA attachment of r edges to new node
        chosen = []
        count = 0
        while count < r:
            rf = rand.random()
            ind = self._attachlist[int(2 * self._E * rf)]
            if ind not in chosen:
                count += 1
                chosen.append(ind) 
                #Update Old Nodes' Info
                self._klist[ind] += 1
        # Then RA attachment of m-r between existing nodes
        while count < m:
            r1 = rand.random(); ind1 = int(self._N * r1)
            r2 = rand.random(); ind2 = int(self._N * r2)
            while ind1 == ind2:
                r2 = rand.random(); ind2 = int(self._N * r2)
            if ind1 not in self._adjlist[ind2]: 
                count += 1
                #Update Old Nodes' Info
                self._klist[ind1] += 1
                self._klist[ind2] += 1
                
    
    def RPassign_static(self, m=0, r = 0, seed=0):
        if seed:
            rand.seed(seed)
        # First RA attachment of r edges to new node
        chosen = []
        count = 0
        while count < r:
            rf = rand.random()
            ind = int(self._N * rf)
            if ind not in chosen:
                chosen.append(ind) 
                #Update Old Nodes' Info
                self._klist[ind] += 1
                count += 1
        # Then PA attachment of m-r between existing nodes
        chosen = []
        while count < m:
            r1= rand.random() 
            r2= rand.random() 
            ind1 = self._attachlist[int(2 * self._E * r1)]
            ind2 = self._attachlist[int(2 * self._E * r2)]
            while ind2 == ind1:
                r2 = rand.random()
                ind2 = self._attachlist[int(2 * self._E * r2)]
            if ind1 not in self._adjlist[ind2]: 
                count += 1
                #Update Old Nodes' Info
                self._klist[ind1] += 1
                self._klist[ind2] += 1