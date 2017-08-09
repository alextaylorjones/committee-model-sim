#!/usr/bin/python
CAN_PLOT = True
import networkx as nx
import re
import pickle
import collections
import math
import numpy as np
import random as rand
import copy

if (CAN_PLOT):
    import matplotlib.pyplot as plt

import scipy
from itertools import combinations as comb
from io import open

#constants
SEED=42

run_id = unicode(np.random.rand())
np.random.seed(SEED)

def mkdir_p(mypath):
    u'''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError, exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
#**** Graph building helper functions ****#

#degree k, rewiring probability p
def get_watts_strogatz_graph(nodeNum,k,p):   
    G = nx.watts_strogatz_graph(nodeNum,k,p,seed=SEED)
    for e in G.edges():
        G[e[0]][e[1]][u'weight'] = 1
    return G

# Build g
def get_undirected_ER_graph(nodeNum, p):
    G = nx.erdos_renyi_graph(nodeNum,p,seed=SEED)
    for e in G.edges():
        G[e[0]][e[1]][u'weight'] = 1
    return G

#Generate tree with powerlaw degree distribution
def get_powerlaw_tree_graph(nodeNum,gamma):
    G = nx.random_powerlaw_tree(nodeNum,gamma,seed=SEED)
    for e in G.edges():
        G[e[0]][e[1]][u'weight'] = 1
    return G


#Graph helper functions

def draw_graph_helper(H,positionFlag=u"spring",drawNodeLabels=True,drawEdgeLabels=False,pos=None):
    print "Edges:",H.edges() 
    if (positionFlag.startswith(u"spring") and pos==None):
        pos=nx.spring_layout(H,iterations=20)
    if (positionFlag.startswith(u"random") and pos==None):
        pos=nx.random_layout(H)
    plt.figure(figsize=(20,20))
    
    nx.draw_networkx_nodes(H,pos,node_color=u'k',alpha=0.3, node_shape=u'o')
    nx.draw_networkx_edges(H,pos)
    if (drawNodeLabels):
        nx.draw_networkx_labels(H,pos,fontsize=14)
    labels = nx.get_edge_attributes(H,u'weight')
    labelsNonZero = {}
    for l in list(labels.keys()):
        if labels[l] > 0.0001:
            labelsNonZero[l] = labels[l]
    if (drawEdgeLabels):
        nx.draw_networkx_edge_labels(H,pos,edge_labels=labelsNonZero)
    
    plt.show()
    return pos


#Construct a graph whose edge weights are the likelihood of an awareness relationship, given the contact graph G
#under the JS spline metric model
# --> This is a symetric measure,for unweighted graphs 
"""
def construct_awareness_from_contact_graph(G):
    H = G.copy()
    #print "Edges",H.edges()
    for pair in comb(G.nodes(),2): #get pairs
        x = pair[0]
        y = pair[1]
       
        metric = 0.0
        if (y in G[x]):#if y a neighor of x (and visa versa)
            metric = 0.5
            
        #JS Metric
        xNbr = set(G[x])
        yNbr = set(G[y])
        
        #if one has no neighbors
        if (len(xNbr) == 0 or len(yNbr) == 0):
            H.add_edge(x,y,weight=metric)
            continue
            
        mutualNbrs = xNbr.intersection(yNbr)
        
        #special case of isolated dyad
        if (xNbr == set([y]) and yNbr == set([x])):
            metric = 1.0
        else:
            metric = metric + (0.5*len(mutualNbrs)/float(len(xNbr.union(yNbr))))
        
        #set as edge weight in new graph
        if (y not in H[x] and metric > 0.0):
            H.add_edge(x,y)
        if (metric > 0.0):
            H[x][y]['weight'] = metric

            assert(H[x][y]['weight'] >= 0 and H[x][y]['weight'] <= 1.0)
    
    return H
"""

def construct_awareness_from_contact_graph(G):
    H = G.copy()
    #print "Edges",H.edges()
    for pair in comb(G.nodes(),2): #get pairs
        x = pair[0]
        y = pair[1]
       
        metric = 0.0
        if (y in G[x]):#if y a neighor of x (and visa versa)
            metric = 0.5
            
        #JS Metric
        xNbr = set(G[x])
        yNbr = set(G[y])
        
        #if one has no neighbors
        if (len(xNbr) == 0 or len(yNbr) == 0):
            H.add_edge(x,y,weight=metric)
            continue
            
        mutualNbrs = xNbr.intersection(yNbr)
        
        #special case of isolated dyad
        if (xNbr == set([y]) and yNbr == set([x])):
            metric = 1.0
        else:
            metric = metric + 0.5*(1 - AWARENESS_COEFFICIENT_ALPHA*math.exp(-1.0*AWARENESS_COEFFICIENT_BETA*len(mutualNbrs)))
        assert(metric >= 0.0 and metric <= 1.0) 
        #set as edge weight in new graph
        if (y not in H[x] and metric > 0.0):
            H.add_edge(x,y)
        if (metric > 0.0):
            H[x][y][u'weight'] = metric

            assert(H[x][y][u'weight'] >= 0 and H[x][y][u'weight'] <= 1.0)
    
        
    return H

#Calculate the expected number of neighbors of S in V(G), not including S itself
def get_exp_coverage(G,S):
    #Coverage probabilities of
    #print "Finding coverage of S = ",S

    covg = [0.0 for n in G.nodes()]
   
    for n in S:
        #Get neighbors
        for b in G.neighbors(n):
            #If not in nodeSet or in S, ignore
            if (b in S):
                continue
            #If in nodeSet and already covered by some node, update coverage
            covg[b] = 1 - (1-covg[b])*(1-G[n][b][u'weight'])
                
    #return the sum of all of the coverage weights

    return sum(covg)
def get_t_step_opt_committee(G_contact,t,k,closure_param):
    G = G_contact
    #Find larger committee
    committee,_ = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),2*k-1,1)
    committee = list(committee)

    #Choose which amount of committee overlap between steps will lead to coverage in the resulting graph
    max_covg = 0
    max_committee =[]
    max_overlap = -1
    f = math.factorial
    n = len(committee)
    num_comb = f(n)/ f(n-k)/f(k)

    #for i,tmp_committee in enumerate(comb(committee,k)):
    for x in xrange(k+1):
        tmp_committee = committee[x:x+k]
        covg = []
        for y in xrange(sample_count):
            H = G.copy()
            #go t steps after alteration
            c_total = 0
            for _ in xrange(t):
                #Perform closure with overlapping portion of large committee
                H = committee_closure_augmentation(H,tmp_committee,closure_param)
                #Calculate coverage of first committee in resulting awareness network
                tmp_committee,c = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(H),k,1)
                c_total += c
            
            covg.append(c_total)

        #average coverage
        covg = sum(covg)/float(len(covg))
        if (covg > max_covg):
            max_covg = covg
            max_committee = tmp_committee
        print (u"Committee #%i out of %i processed "%(x,(k)))

    #Calculate actual coverage using max_committee
    covg = get_exp_coverage(construct_awareness_from_contact_graph(G),max_committee)
    print (u"In %i-step process, max overlap was "%t,max_overlap)
    #Select max committee as committee
    return set(max_committee),covg


def get_exp_coverage2(G,S):

    #get neighbor set
    nbrs = []
    for v in S:
        nbrs[0:0] = list(G[v].keys())
    nbrs = set(nbrs)

    n = len(G)
    covg = np.zeros(n)

    #Make log weights

    log_weight = np.zeros((n,n))
 
    for i in S:
        for j in list(G[i].keys()):
            if (j not in S):
                if (G[i][j][u'weight'] == 1.0):
                   #Coverage is guaranteed
                   log_weight[i,j] = -float(u'inf') 
                else:
                    log_weight[i,j] += math.log(1.0 - G[i][j][u'weight'])
        
    #Column sums
    col_sums = np.sum(log_weight,0)

    cm_sum = np.sum(np.exp(col_sums))

    return n - cm_sum
    
# Find approximation max coverage set using k-greedy method, with t total nodes in solution
def greedy_expected_max_coverage_set(G,t,k):
    #Debug
    assert(t < len(G.nodes()))
    
        #solution
    soln = set()
           
    #node set
    remaining_nodes = set(G.nodes())
    
    #Construct how many nodes to select at each step
    k_vals = [k for _ in xrange(int(math.floor(float(t)/k)))]
    if (t - sum(k_vals) > 0):
        k_vals.append(t - sum(k_vals))
    #Debug
    assert(sum(k_vals) == t)
    
    for sub_k in k_vals:
        
        #Get value of current (partial solution)
        if (len(soln) > 0):
            current_covg = get_exp_coverage(G,soln)
        else:
            current_covg = 0.0
           
        max_sym_diff = 0.0
        best_seen_soln = set()
        remaining_nodes = remaining_nodes.difference(soln)
        
        #print "\n Selecting next ", sub_k, "nodes from set ", remaining_nodes
        for nodeset in comb(remaining_nodes,sub_k): #For all k combinations of unselected nodes
            #calculate expected coverage with the addition of nodeset to the current solution
            sym_diff = get_exp_coverage(G,soln.union(nodeset))

            #print "Coverage of ", soln.union(nodeset), "is ", sym_diff
            if (sym_diff > max_sym_diff):
                #print "Using ", nodeset, "as current best"
                max_sym_diff = sym_diff
                best_seen_soln = nodeset
        #print "Chose node(s) ", list(best_seen_soln)
        #Add found nodes to solution set
        #Debug
        assert(len(set(best_seen_soln).intersection(soln)) == 0)
        soln = soln.union(set(best_seen_soln))
        
    #Return coverage plus the number of committee members
    return soln,(max_sym_diff+t)

## Close each triplet with at least one node in committee according to threshold
## Close all triads which have one edge in committee

def get_overlap_committee(G,k,p1,p2):
    #Get committee pt1 -high coverage core
    if (p1 > 0):
        committee1,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),p1,1)
        committee = set(committee1)
    else:
        committee = set()
        committee1 = set()
        covg = 0

    #print("Coverage of graph is ",covg, " by high coverage core of size ",p1)

    #print("Committee 1:", committee1)
    
    #Get contact diversity
    committee2= set()

    filledFlag = False
    while (filledFlag == False):

        committee2 = []
        #Get neighbors of committee1
        nbrs = []
        for c in committee1:
            nbrs = nbrs + list(G[c].keys())
        nbrs = set(nbrs)
        remaining = set(G.nodes()).difference(nbrs)

        #if we fail, override this flag
        filledFlag = True
        total_nbrs_size = 0

        for i in xrange(1,p2+1):
            last_node = None
            max_nbrs_size = -1
            #print("Size of remaining contact neighborhood is ",len(remaining), "out of ",len(G.nodes())," after ", i," members of diversity set")
            for x in set(G.nodes()).difference(committee):
                nbrs_size = len(set(list(G[x].keys())).intersection(remaining))
                if (nbrs_size > max_nbrs_size):
                    max_nbrs_size = nbrs_size
                    last_node = x
            total_nbrs_size += max_nbrs_size

            print u"max neighbors in iteration ",i, u" is ",max_nbrs_size
            #if (total_nbrs_size/float(i) < MIN_DIVERSITY_THRESHOLD*len(G.nodes())):
            #if (max_nbrs_size < MIN_DIVERSITY_THRESHOLD*len(G.nodes())):
            if (max_nbrs_size < 1):
                print (u"Average covered neighbors was ",total_nbrs_size/float(i), u" from ", i, u" diversity nodes less than required", (MIN_DIVERSITY_THRESHOLD*len(G.nodes())))
                filledFlag = False
                #resize committees
                p1 = p1 + 1
                p2 = p2 - 1 

                #get new hig coverage committee committee
                #print(("Diversity committee failed to reach threshold at iteration ",i," out of ", p2, "getting new high coverage committee with ", p1, "total nodes, then getting new diversity committee with ",p2, "nodes"))
                committee1,_ = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),p1,1)
                #reset committee
                committee = set(committee1)
                #print(("Committee 1:",committee1))
                #print(("Committee 2:",committee2))
                break


            remaining = remaining.difference(list(G[last_node].keys()))
            committee2.append(last_node)

            committee = committee.union(set([last_node]))

        #print("found committee 2:",committee2)

        committee2 = set(committee2)
    print (u"Committee 1:",committee1)
    print (u"Committee 2:",committee2)
    #join committees
    committee = committee1.union(committee2)
    assert(len(committee) == (p1 + p2))
    u"""
    if (p3 > 0): #save time when p3 set is not used
        #Get one and two hop neighborhoods of committee pt1 and pt2
        one_hop_neighbors = []
        two_hop_neighbors = []
        for c in committee:
            one_hop_neighbors = one_hop_neighbors + list(G[c].keys())
            for d in list(G[c].keys()):
                two_hop_neighbors = two_hop_neighbors + list(G[d].keys())

        #rename and make sets to make unique
        one_hop_neighbors = set(one_hop_neighbors)
        two_hop_neighbors = set(two_hop_neighbors).difference(one_hop_neighbors)

        #Find set of size p3 which overlaps the two hop neighborhood as much as possible
        committee3 = []
        
        for i in range(p3):
            #initialize coverage
            covg = [0 for _ in G.nodes()]
            print(("In glue set, two hop neighborhood is size ",len(two_hop_neighbors), "after ",i,"memebers of glue set"))

            #Find nodes outside the committee
            for v in set(G.nodes()).difference(committee):
                #find the number of neighbors of these nodes which are inside the two_hop_neighborhood
                covg[v] = len(set(G[v].keys()).intersection(two_hop_neighbors))


            max_covg = max(covg)
            #If glue is not useful enough
            if (max_covg < MIN_GLUE_COVG_THRESHOLD*len(G.nodes())):
                break

            max_covg_node = covg.index(max_covg)       
     
            committee3.append(max_covg_node)

            #add max coverage node
            committee = committee.union(set([max_covg_node]))

            #remove covered nodes in 2hop neighborhood by new node
            
            two_hop_neighbors = two_hop_neighbors.difference(set(list(G[max_covg_node].keys())))
        
        if (len(committee3) == p3):
            print(("In glue set, two hop neighborhood is size ",len(two_hop_neighbors), "after ",p3,"memebers of glue set"))
        else:
            print("Glue set did not pass threshold, adding random nodes instead")
            r = p3 - len(committee3)
            #Get unique random set outside committee
            c = np.random.choice(list(set(G.nodes()).difference(committee)),r)
            while (len(c) != len(set(c))):
                c = np.random.choice(list(set(G.nodes()).difference(committee)),r)
                

            committee = committee.union(set(c))
                
    """                    
    assert(len(committee) == k)
    covg = get_exp_coverage(construct_awareness_from_contact_graph(G),committee)
 
    return committee,covg,p2

def committee_closure_augmentation(G,committee,closure_threshold):
    H = G.copy()
    #print "Closing committee", committee 
    for x,y in comb(committee,2):
        #add committee-committee edge
        H.add_edge(x,y,weight=1.0)
        
    for x,y in comb(committee,2):
        #Close triads which have two committee members
        for n in set(H[x].keys()).union(list(H[y].keys())).difference(set([x,y])):
            if (n not in G[y]): #then unclosed triad, missing edge n-y
                r = rand.random()
                if (r<closure_threshold):
                    H.add_edge(n,y,weight=1)
                    print "New edge", n,y
            if (y not in G[x]): #unclosed triad missing edge n-x
                r = rand.random()
                if (r<closure_threshold):
                    H.add_edge(n,x,weight=1)
                    print "New edge", n,x

    u"""
    #Close triads in which the center is a neighbor 
    nbrs = []
    for c in committee:
        nbrs = nbrs + list(G[c].keys())
    nbrs = set(nbrs)

    for n in nbrs:
        #one neighor inside the committee
        for x in set(G[n]).intersection(committee):
            for y in set(G[n]).intersection(nbrs):
                r = rand.random()
                if (r<closure_threshold):
                    H.add_edge(x,y,weight=1)
    """

    return H 


def get_distribution_coverage_time(G_init,k,alpha,closure_param,trials=100,max_tries=100,alg=u"greedy",draw_freq=0,show_ecc=False):
    time_dist = []
    num_nodes = len(G_init.nodes())
  
     
    committee_coverage = []
    max_committee_coverage = []
    committees = []
    graphs = []
    
    for t in xrange(trials):
        G = G_init.copy()
        graphs.append([])
        committees.append([])
        committee_coverage.append( [])
        max_committee_coverage.append([])

        pos = None
        tries = 0
            
        if (alg.startswith(u"greedy-diverse")):
            #Greedy coverage algorithm
            _,max_covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
            committee,_ = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k-1,1)

            
            #get nbrs
            nbrs = []
            for c in committee:
                nbrs = nbrs + list(G[c].keys())
            nbrs = set(nbrs)
            remaining = set(G.nodes()).difference(nbrs)

            last_node = None
            max_nbrs_size = -1

            for x in set(G.nodes()).difference(committee):
                nbrs_size = len(set(list(G[x].keys())).intersection(remaining))
                if (nbrs_size > max_nbrs_size):
                    max_nbrs_size = nbrs_size
                    last_node = x


            committee = committee.union(set([last_node]))
            covg = get_exp_coverage(G,committee)
            
            committee_coverage[-1].append(covg/float(num_nodes))
            max_committee_coverage[-1].append(max_covg/float(num_nodes))
            committees[-1].append(committee)


            print u"T=0"
            if (draw_freq != 0):
                H = construct_awareness_from_contact_graph(G)
                for i,j in H.edges():
                    if (G[i][j][u'weight'] < 0.25):
                        H.remove_edge(i,j)
                pos = draw_graph_helper(H,u"spring",pos)

            graphs[-1].append(G)
            
                            #if showing eccentricities
            if (show_ecc):
                ecc_dict = nx.eccentricity(G)
                plt.hist(list(ecc_dict.values()),bins=range(max(ecc_dict.values())+2))
                plt.title(u"Iteration 0 of Greedy Committee Formation:\n Contact Network Eccentricity Distribution")
                plt.show()
            
            while (covg < alpha*num_nodes and tries < max_tries):
                G = committee_closure_augmentation(G,committee,closure_param)
                graphs[-1].append(G)
            
                #print "T=%i" % (tries+1)
                
                #draw according to frequency

                if (draw_freq != 0 and (tries+1) % draw_freq == 0):
                    H = construct_awareness_from_contact_graph(G)
                    for i,j in H.edges():
                        if (G[i][j][u'weight'] < 0.25):
                            H.remove_edge(i,j)
                    pos = draw_graph_helper(H,u"spring",pos)
                
                #if showing eccentricities
                if (show_ecc):
                    ecc_dict = nx.eccentricity(G)
                    plt.hist(list(ecc_dict.values()),bins=range(max(ecc_dict.values())+2))
                    plt.title(u"Iteration %i of Greedy Committee Formation:\n Contact Network Eccentricity Distribution" % (tries+1))
                    plt.show()
                
                tries += 1
             
                #Greedy coverage algorithm
                _,max_covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
                committee,_ = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k-1,1)

                
                #get nbrs
                nbrs = []
                for c in committee:
                    nbrs = nbrs + list(G[c].keys())
                nbrs = set(nbrs)
                remaining = set(G.nodes()).difference(nbrs)

                last_node = None
                max_nbrs_size = -1

                for x in set(G.nodes()).difference(committee):
                    nbrs_size = len(set(list(G[x].keys())).intersection(remaining))
                    if (nbrs_size > max_nbrs_size):
                        max_nbrs_size = nbrs_size
                        last_node = x


                committee = committee.union(set([last_node]))
                covg = get_exp_coverage(G,committee)
                
                committees[-1].append(committee)
                committee_coverage[-1].append(covg/float(num_nodes))
                max_committee_coverage[-1].append(max_covg/float(num_nodes))
        elif (alg.startswith(u"greedy")):
            #Greedy coverage algorithm
            committee,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
            committee_coverage[-1].append(covg/float(num_nodes))
            max_committee_coverage[-1].append(covg/float(num_nodes))
            committees[-1].append(committee)


            print u"T=0"
            if (draw_freq != 0):
                pos = draw_graph_helper(G,u"spring",pos)
            graphs[-1].append(G)
            
                            #if showing eccentricities
            if (show_ecc):
                ecc_dict = nx.eccentricity(G)
                plt.hist(list(ecc_dict.values()),bins=range(max(ecc_dict.values())+2))
                plt.title(u"Iteration 0 of Greedy Committee Formation:\n Contact Network Eccentricity Distribution")
                plt.show()
            
            while (covg < alpha*num_nodes and tries < max_tries):
                G = committee_closure_augmentation(G,committee,closure_param)
                graphs[-1].append(G)
            
                #print "T=%i" % (tries+1)
                
                #draw according to frequency
                if (draw_freq != 0 and (tries+1) % draw_freq == 0):
                    pos = draw_graph_helper(G,u"spring",pos)
                
                #if showing eccentricities
                if (show_ecc):
                    ecc_dict = nx.eccentricity(G)
                    plt.hist(list(ecc_dict.values()),bins=range(max(ecc_dict.values())+2))
                    plt.title(u"Iteration %i of Greedy Committee Formation:\n Contact Network Eccentricity Distribution" % (tries+1))
                    plt.show()
                
                tries += 1
             
                committee,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
                committees[-1].append(committee)
                committee_coverage[-1].append(covg/float(num_nodes))
                max_committee_coverage[-1].append(covg/float(num_nodes))
        elif (alg.startswith(u"diversity")):
             #recover step size
            #overlap-k1-k2-k3
            p1 = int((re.split(u"-",alg))[1])
            p2 = int((re.split(u"-",alg))[2])

            assert(p1 + p2 == k) 

            _,max_covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
            max_committee_coverage[-1].append(max_covg/float(num_nodes))
            
            #Get high coverage core of size p1 
            #diversifying set of size p2
            #and "glue" set of size p3
            committee,covg,num_c2 = get_overlap_committee(G,k,p1,p2)
            

            committee_coverage[-1].append(covg/float(num_nodes))

            committees[-1].append(committee)


            print u"T=0"
            if (draw_freq != 0):
                pos = draw_graph_helper(G,u"spring",pos=pos)


            graphs[-1].append(G)
            
                            #if showing eccentricities
            if (show_ecc):
                ecc_dict = nx.eccentricity(G)
                plt.hist(list(ecc_dict.values()),bins=range(max(ecc_dict.values())+2))
                plt.title(u"Iteration 0 of Greedy Committee Formation:\n Contact Network Eccentricity Distribution")
                plt.show()
            
            while (covg < alpha*num_nodes and tries < max_tries):
                G = committee_closure_augmentation(G,committee,closure_param)
                graphs[-1].append(G)
            
                #print "T=%i" % (tries+1)
                
                #draw according to frequency
                if (draw_freq != 0 and (tries+1) % draw_freq == 0):
                    pos = draw_graph_helper(G,u"spring",pos=pos)

                #if showing eccentricities
                if (show_ecc):
                    ecc_dict = nx.eccentricity(G)
                    plt.hist(list(ecc_dict.values()),bins=range(max(ecc_dict.values())+2))
                    plt.title(u"Iteration %i of Greedy Committee Formation:\n Contact Network Eccentricity Distribution" % (tries+1))
                    plt.show()
                
                tries += 1

                _,max_covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
                max_committee_coverage[-1].append(max_covg/float(num_nodes))
                
                committee,covg,num_c2 = get_overlap_committee(G,k,p1,p2)

                committee_coverage[-1].append(covg/float(num_nodes))

                committees[-1].append(committee)

        elif (alg.startswith(u"step")):
            #t-step coverage algorithm

            #recover step size
            step_sz = (re.split(u"-",alg))[1]
            step_sz = int(step_sz)
            assert(step_sz > 0) 

            #Get first committee
            committee,covg = get_t_step_opt_committee(G,step_sz,k,closure_param)

            committee_coverage[-1].append(covg/float(num_nodes))
            #calc max coverage
            _,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
            max_committee_coverage[-1].append(covg/float(num_nodes))
            committees[-1].append(committee)


            print u"T=0"
            if (draw_freq != 0):
                pos = draw_graph_helper(G,u"spring",pos)
            graphs[-1].append(G)
            
                            #if showing eccentricities
            if (show_ecc):
                ecc_dict = nx.eccentricity(G)
                plt.hist(list(ecc_dict.values()),bins=range(max(ecc_dict.values())+2))
                plt.title(u"Iteration 0 of Greedy Committee Formation:\n Contact Network Eccentricity Distribution")
                plt.show()
            
            while (covg < alpha*num_nodes and tries < max_tries):
                G = committee_closure_augmentation(G,committee,closure_param)
                graphs[-1].append(G)
            
                #print "T=%i" % (tries+1)
                
                #draw according to frequency
                if (draw_freq != 0 and (tries+1) % draw_freq == 0):
                    pos = draw_graph_helper(G,u"spring",pos)
                
                #if showing eccentricities
                if (show_ecc):
                    ecc_dict = nx.eccentricity(G)
                    plt.hist(list(ecc_dict.values()),bins=range(max(ecc_dict.values())+2))
                    plt.title(u"Iteration %i of Greedy Committee Formation:\n Contact Network Eccentricity Distribution" % (tries+1))
                    plt.show()
                
                tries += 1
            
                if (tries % (step_sz+1) == 0):
                    committee,covg = get_t_step_opt_committee(G,step_sz,k,closure_param)
                else:
                    committee,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
                committees[-1].append(committee)
                committee_coverage[-1].append(covg/float(num_nodes))

                _,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
                max_committee_coverage[-1].append(covg/float(num_nodes))
        elif (alg.startswith(u"random")):
            #Calculate coverage of greedily selected committee
            print "Starting random committee selection"
            _,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
            
            max_committee_coverage[-1].append(covg/float(num_nodes))

 
            #random committee selection
            committee = np.random.choice(G.nodes(),k)
            
            #ensure uniqueness
            while (len(committee) != len(set(committee))):
                 committee = np.random.choice(G.nodes(),k)
            
            committees[-1].append(committee)
                    
            covg = get_exp_coverage(construct_awareness_from_contact_graph(G),committee)


            committee_coverage[-1].append(covg/float(num_nodes))
           
            if (draw_freq != 0):
                H = construct_awareness_from_contact_graph(G)
                for i,j in H.edges():
                    if (H[i][j][u'weight'] < 0.25):
                        H.remove_edge(i,j)
                pos = draw_graph_helper(H,u"spring",pos)



                
            graphs[-1].append(G)
            
            #if showing eccentricities
            if (show_ecc):
                ecc_dict = nx.eccentricity(G)
                plt.hist(list(ecc_dict.values()),bins=range(max(ecc_dict.values())+2))
                plt.title(u"Iteration 0 of Random Committee Formation:\n Contact Network Eccentricity Distribution" )
                plt.show()
                    
            while (covg < alpha*num_nodes and tries < max_tries):
            
                #augment with random committee
                G = committee_closure_augmentation(G,committee,closure_param)
                

                graphs[-1].append(G)
                #Get coverage quality under greedy method
                _,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
            
                max_committee_coverage[-1].append(covg/float(num_nodes))               
                
                #draw according to frequency
                if (draw_freq != 0 and (tries+1) % draw_freq == 0):
                    #print "T=",tries+1
                    H = construct_awareness_from_contact_graph(G)
                    for i,j in H.edges():
                        if (H[i][j][u'weight'] < 0.25):
                            H.remove_edge(i,j)
                    pos = draw_graph_helper(H,u"spring")

                tries += 1
                
                #randomly choose committee
                committee = np.random.choice(G.nodes(),k)
                
                #ensure uniqueness
                while (len(committee) != len(set(committee))):
                     committee = np.random.choice(G.nodes(),k)

                committees[-1].append(committee)

                covg = get_exp_coverage(construct_awareness_from_contact_graph(G),committee)
                committee_coverage[-1].append(covg/float(num_nodes))
        
        if (covg < alpha*(len(G.nodes()))):
            print (u"ERROR: Coverage was %f, not high enough after Max tries= %i exceeded" % (covg,max_tries))
        #print "Finished at time ", tries, " with committee", committee, " with coverage ", covg
        time_dist.append(tries)
        
    return time_dist,graphs,committees,committee_coverage,max_committee_coverage
    
#Load data

CONTACT_COL = 4 #indexed from 0
AWARENESS_COL = 6 #indexed from 0

contact_graphs = []
awareness_graphs = []
data_files = []


from os import listdir as ld

#Load data files from data directory
for filename in ld(u"./data/"):
    if (filename.endswith(u".txt")):
        data_files.append(open(u"./data/%s" % filename))

    


#construct graphs
for data_file in data_files:
    contact_graph = nx.Graph()
    awareness_graph = nx.Graph()
    
    #read lines
    for i,line in enumerate(data_file):
        if (len(line.split()) != 10):
            print (u"",line, u"line",i,u" not long enough", data_file)
            continue
        if ((line.split())[CONTACT_COL] == u'1'):
            #print "int", (line.split())[CONTACT_COL] 
            contact_graph.add_edge(int((line.split())[0]),int((line.split())[1]),weight=1.0)
        if ((line.split())[AWARENESS_COL] == u'1'):
            awareness_graph.add_edge(int((line.split())[0]),int((line.split())[1]),weight=1.0)
    print (u"Adding graphs ", data_file.name, u"with ", len(contact_graph.nodes()),u" nodes and ", len(contact_graph.edges()), u" edges")
    name = re.split(ur'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', data_file.name)[3]
    print (u"Downloaded ",name)
    contact_graphs.append((contact_graph,name))
    awareness_graphs.append((awareness_graph,name))

#Check graphs
for g,name in contact_graphs:
    for x,y in g.edges():
        assert(x in g.nodes() and x in list(g[y].keys()) and y in list(g[x].keys()))

            
#close files
for data_file in data_files:
    data_file.close()


# In[8]:

u"""#Draw graphs
for g,name in contact_graphs:
    print "Name: ",name
    draw_graph_helper(g,"spring")
"""


# In[9]:


#standard data
single_contact_networks_list = [(g,name) for g,name in contact_graphs]


u"""
# In[14]:

G_list = []

#List of graphs to try


#two graphs together
for G,H in comb(contact_graphs,2):

    
    S = nx.convert_node_labels_to_integers(G[0],first_label=0)
    R = nx.convert_node_labels_to_integers(H[0],first_label=-1*len(H[0].nodes()))
     F = nx.compose(S,R)
    G_list.append((F, "%s - merged %s" % (G[1], H[1])))
    
    #make sure no overlap
    for v in S.nodes():
        assert(set(S[v].keys()).intersection(set(R.nodes())) == set())
    
    for w in R.nodes():
        assert(set(R[w].keys()).intersection(set(S.nodes())) == set())

#make deepcopy of list
combined_contact_networks_list = copy.deepcopy(G_list)
na

"""


def plot_comp_results(plt_name,all_data,metric_name,plt_type=u'avg',saveName=None):


    print (u"\n\b Plotting metric %s"%metric_name)

    if (plt_type.startswith(u'avg')):
 
        fig = plt.figure()
        fig.set_size_inches(6,6) 
       
        for alg_name in alg_list: 
            print (u"Plotting for alg",alg_name, u" tracking metric",metric_name)
            
            #Iterate through each kind of data
            records = all_data[alg_name][metric_name]
            
            #print("Data to plot:",records)
            data_avg = []
            #print ("\n data",records)
            for record in records:
                #Store records in data_avg
                for i,entry in enumerate(record):
                    if (i+1 > len(data_avg)):
                        data_avg.append([entry])
                        assert(len(data_avg) == i+1)
                    else:
                        data_avg[i].append(entry)

            #Average entries
            to_plot = []
            for entry in data_avg:
                to_plot.append(float(sum(entry))/len(entry))

            #Plot data average
            print (u"Ploting data",metric_name,u" for alg",alg_name,u":",to_plot)
            plt.plot(range(len(to_plot)),to_plot,label=alg_name)
            
        plt.title(u"Graph: %s" % plt_name)
        plt.legend()
        plt.xlabel(u"Iteration Count")
        plt.ylabel(u"Average Value of %s" % metric_name)
        
        if (saveName != None):
            mkdir_p(u"plots/%s"%run_id)
            print (u"Saving run %s"%run_id)
            plt.savefig(u"plots/%s/%s.png"%(run_id,saveName),bbox_inches=u'tight')
        #plt.show()

def list_local_bridges(G):
    local_bridges = []
    for v in G.nodes():
        nbrs = list(G[v].keys())
        twoHopNbrs = []
        for w in nbrs:
            twoHopNbrs = twoHopNbrs + list(G[w].keys())
        
        for n in nbrs:
            if (n not in twoHopNbrs):
                local_bridges.append((v,n))
    return local_bridges 
#Find closure time on real graphs



#G_list = single_contact_networks_list

def get_example_network(ID):
    if (ID == 0):
        G = nx.Graph()
        G.add_edge(1,3)
        G.add_edge(3,-3)
        G.add_edge(1,4)
        G.add_edge(4,-4)
        G.add_edge(1,5)
        G.add_edge(5,-5)
        G.add_edge(1,7)
        G.add_edge(7,-7)
        G.add_edge(2,8)
        G.add_edge(8,-8)
        G.add_edge(2,9)
        G.add_edge(9,-9)
        G.add_edge(2,10)
        G.add_edge(2,11)
        G.add_edge(2,12)
        G.add_edge(10,-10)
        G.add_edge(11,-11)
        G.add_edge(12,-12)
        G.add_edge(2,14)
        G.add_edge(14,-14)

        return G

G_list = [(get_example_network(0), "Example 0")]



#alg_list= ['overlap-3-1-0','overlap-1-3-0','overlap-2-1-1','overlap-4-0-0','random']
#alg_list = ['random']
alg_list = ['diversity-2-0']

metric_list = [u'contact_diameter',u'awareness_diameter' , u'avg_awareness_path', u'avg_contact_path',u'num_edges',u'coverage',  u'max_coverage',u'max_second_coverage',u'clustering',u'local_bridges',u'committee']

trials = 5
COMMITTEE_SZ = 2
COVERAGE_MIN = 1.0
CLOSURE_PARAM = 0.20
max_tries = 30

#for t-step lookahead
sample_count = 10
#for overlap
MIN_GLUE_COVG_THRESHOLD = 0.01
MIN_DIVERSITY_THRESHOLD = 0.02

#for awareness
AWARENESS_COEFFICIENT_ALPHA = 1.0
AWARENESS_COEFFICIENT_BETA = 0.05

print u"Starting simulation"
print (u"Coverage minimum fraction %.3f, committee size %i , closure prob %.2f, min glue threshold %.2f, min diversity threshold %.2f" % (COVERAGE_MIN,COMMITTEE_SZ,CLOSURE_PARAM,MIN_GLUE_COVG_THRESHOLD,MIN_DIVERSITY_THRESHOLD))

if (CAN_PLOT):
    plt.rcParams.update({u'font.size': 14})

#load results

#Store result
results = {}
fileLoad = False
#Load first file in results folder

for filename in ld(u"./results/"):
    if (filename.endswith(u"pickled")):
        f = open(u"results/%s"%filename,u'rb')
        results = pickle.load(f)
        f.close()
        fileLoad = True
        print (u"Loading results from file",filename)
        
        split_name_list = re.split(u"-",filename)
        #format results-%s-k=%i-closure=%.2f-sample_count=%i-pickled" %(name,COMMITTEE_SZ,CLOSURE_PARAM,sample_count)
        committee = split_name_list[2]
        COMMITTEE_SZ = int(re.split(u"=",committee)[1])
        
        closure = split_name_list[3]
        CLOSURE_PARAM = float(re.split(u"=",closure)[1])
    
        sample = split_name_list[4]
        CLOSURE_PARAM = int(re.split(u"=",sample)[1])

        for name in list(results.keys()):
            print (u"Using results with name ",name)
            for metric in metric_list:
                plot_comp_results(u"%s - k=%i - closure=%.2f - sample count %i" %(name,COMMITTEE_SZ,CLOSURE_PARAM,sample_count),results[name],metric,saveName=u"%s-%s"%(metric,name))

        
if (fileLoad == True):
    print u"Finished loading all files, exiting"
    quit()    

for z,graph_desc in enumerate(G_list):
    if (z not in [0]):
        continue 
    G,name = graph_desc
    print (u"G=%s has" %name, len(G.nodes()), u" nodes")
    G = nx.convert_node_labels_to_integers(G)

    data_set = []
    results[name] = {}

    #repeat process several times, adding all times to one list
    for alg in alg_list:
        print (u"Using %s method"%alg)
        t,graphs,committee,covg,max_covg = get_distribution_coverage_time(G,COMMITTEE_SZ,COVERAGE_MIN,CLOSURE_PARAM,trials,max_tries,alg=alg,draw_freq=5,show_ecc=False)
        data_set.append((t,graphs,covg,max_covg,committee,alg))


    #Post process graphs
    for data in data_set:
        #DEBUG 
        #copy time distribution of convergence
        time_dist = []
        print (u"Processing data for algorithm",alg)
        alg = data[-1] 
        results[name][alg] = {}

        results[name][alg][u'contact_diameter'] = [] 
        results[name][alg][u'awareness_diameter'] = [] 
        results[name][alg][u'avg_contact_path'] = [] 
        results[name][alg][u'avg_awareness_path'] = [] 
        results[name][alg][u'num_edges'] = [] 
        results[name][alg][u'coverage'] = [] 
        results[name][alg][u'max_coverage'] = [] 
        results[name][alg][u'max_second_coverage'] = [] 
        results[name][alg][u'clustering'] = [] 
        results[name][alg][u'local_bridges'] = [] 
        results[name][alg][u'committee'] = [] 


        #for x,record in enumerate(data):
        #Save times
        #times = data[0]
        #time_dist = time_dist + times
    
        #Record diameter trends
        graph_trials = data[1]

        #Record committees
        committee_lists = data[4]


        for y,trial in enumerate(graph_trials):
            contact_diameter = []
            awareness_diameter = []
            avg_contact_path = []
            avg_awareness_path = []
            edges = []
            clustering = []
            local_bridges = []
            degrees = []
            max_second_covg = []
            for i,graph in enumerate(trial):
                #print "Graph edges",len(graph.edges())
                contact_path_lengths_dict = nx.all_pairs_shortest_path_length(graph)
                awareness_path_lengths_dict = nx.all_pairs_shortest_path_length(construct_awareness_from_contact_graph(graph))

                contact_path_lengths = []
                awareness_path_lengths = []
                
                for k in list(contact_path_lengths_dict.keys()):
                    sp_dict = contact_path_lengths_dict[k]
                    for x in list(sp_dict.values()):
                        contact_path_lengths.append(x)

                for k in list(awareness_path_lengths_dict.keys()):
                    sp_dict = awareness_path_lengths_dict[k]
                    for x in list(sp_dict.values()):
                        awareness_path_lengths.append(x)


                #print "Contact paths", contact_path_lengths
                #print "Awareness paths paths", awareness_path_lengths

                cd = sorted(contact_path_lengths)[-2]

 
                contact_diameter.append(cd)
                awareness_diameter.append(max(contact_path_lengths))
                 
                avg_contact_path.append(sum(list(contact_path_lengths))/float(len(contact_path_lengths)))
                avg_awareness_path.append(sum(list(awareness_path_lengths))/float(len(awareness_path_lengths)))
            
                edges.append(len(graph.edges()))

                clustering.append(nx.average_clustering(graph))
                local_bridges.append(len(list_local_bridges(graph))/2)
                
                #Remove committee from graph and retry
                committee = committee_lists[y][i]
                #print "Coverage at time ",i,"of first set is ",covg, " and should be ",record[2][y][i]
                G = graph.copy() 
                for c in committee:
                    G.remove_node(c)
                G = nx.convert_node_labels_to_integers(G)
                _,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),COMMITTEE_SZ,1)

                #print(("Adding coverage value",covg))
                max_second_covg.append(covg/len(G))
                
                results[name][alg][u'contact_diameter'].append(contact_diameter)
                results[name][alg][u'awareness_diameter'].append(awareness_diameter)
                results[name][alg][u'avg_awareness_path'].append(avg_awareness_path)
                results[name][alg][u'avg_contact_path'].append(avg_contact_path)
                results[name][alg][u'num_edges'].append(edges)
                results[name][alg][u'clustering'].append(clustering)
                results[name][alg][u'local_bridges'].append(local_bridges)
                results[name][alg][u'max_second_coverage'].append(max_second_covg)
            

            #Record coverage
            results[name][alg][u'coverage'] = data[2]
            results[name][alg][u'max_coverage'] = data[3]

            #Record committee consistency
            committee_lists = data[4]
            for committee_list in committee_lists:
                committee_remain= []
                for i,committee in enumerate(committee_list):
                    if (i==0):
                        continue
                    prev_committee = committee_list[i-1]
                    current_committee = committee
                    #Calc percent of current committee which changes over one round
                    committee_remain.append( 1.0 - (len(set(current_committee).difference(set(prev_committee)))/float(COMMITTEE_SZ)) )
                    
                results[name][alg][u'committee'].append(committee_remain)

         
    print u"Results:"
    if (CAN_PLOT):
        #Plot data
        for metric in metric_list:
            plot_comp_results(u"%s - k=%i - closure=%.2f" %(name,COMMITTEE_SZ,CLOSURE_PARAM),results[name],metric,saveName=u"%s-%s"%(metric,name))


    #Save files
    mkdir_p(u"results")
    with open(u"results/results-%s-k=%i-closure=%.2f-sample_count=%i-pickled" %(name,COMMITTEE_SZ,CLOSURE_PARAM,sample_count),u'wb') as fp:
        print (u"Dumping results-%s-k=%i-closure=%.2f-sample_count=%i-pickled" %(name,COMMITTEE_SZ,CLOSURE_PARAM,sample_count))
        pickle.dump(results,fp)
       
    #plt.ylim([-0.1,1.1])
    #for alg in alg_list:
    #        plot_comp_results("%s - k=%i - closure=%.2f" %(name,COMMITTEE_SZ,CLOSURE_PARAM),results[name][alg]['max_second_coverage'],'(greedy) second',greedy_results[name]['alg']['max_coverage'][0],'(greedy) first','(Greedy) Max vs Alt Coverage',saveName="first-second-greedy-%s"%name)

#print("Done")
#plt.show()
"""
iterations = 30 #degree dist
frequency = 5
coverage_set_sz = 3
diversity_set_sz = 3

if __name__ == "__main__":
    
    # Plot degree distributions
    h = 2
    w = int(math.ceil(iterations/2/float(frequency)))+1

    for i,graph_desc in enumerate(G_list):
        #just show 2nd graph
        #if (i==0):
        #    continue

        G,name = graph_desc
        G = nx.convert_node_labels_to_integers(G)
        H = G.copy()
        G_r = G.copy()

        print(("\nShowing ",name))

        degrees = [[] for _ in range(iterations)]
        degrees_r = [[] for _ in range(iterations)]
        committee_remain = [[] for _ in range(iterations)]

        for t in range(trials):
            G = H.copy()
            G_r = H.copy()
            print("Running trial ",t)
            for i in range(iterations):
                #Get Greedy Committee and Update graph

                degrees[i] = degrees[i] + list(G.degree().values())
                degrees_r[i] = degrees_r[i] + list(G_r.degree().values())

                committee,_ = get_overlap_committee(G,COMMITTEE_SZ,coverage_set_sz,diversity_set_sz)
                #DEBUG 
                #committee,_ = greedy_expected_max_coverage_set(G,COMMITTEE_SZ,1)

        
                G = committee_closure_augmentation(G,committee,CLOSURE_PARAM)

                #Get random committee and update graph
                committee_r = np.random.choice(G_r.nodes(),COMMITTEE_SZ)
            
                #ensure uniqueness
                while (len(committee_r) != len(set(committee_r))):
                    committee_r = np.random.choice(G_r.nodes(),COMMITTEE_SZ)
                # update graph
                G_r = committee_closure_augmentation(G_r,committee_r,CLOSURE_PARAM)


        f, axarr = plt.subplots(w, h)
        f_r, axarr_r = plt.subplots(w, h)
        f.subplots_adjust(hspace=0.3)
        f_r.subplots_adjust(hspace=0.3)

        x = 0
        y = 0

        for i in range(iterations):
            if (i % frequency == 0):
                #Greedy
                degree_sequence=sorted(degrees[i], reverse=True) # degree sequence
                degreeCount=collections.Counter(degree_sequence)
                deg, cnt = list(zip(*list(degreeCount.items())))        

                axarr[x,y].bar(deg, cnt, width=0.80, color='g')
                axarr[x,y].set_xlim([0,len(G.nodes())])
                #axarr[x,y].set_ylim([0,len(G.nodes())])
                if (x==0 and y==0):
                    axarr[x,y].set_title("Coverage %i+Diversity %i (threshold=%.2f)- k=%i,p=%.2f %s -T=0 "%(coverage_set_sz,diversity_set_sz,MIN_DIVERSITY_THRESHOLD,COMMITTEE_SZ,CLOSURE_PARAM,name,))
                else:
                    axarr[x,y].set_title("T=%i"%i)

                #random
                degree_sequence_r=sorted(degrees_r[i], reverse=True) # degree sequence
                degreeCount_r=collections.Counter(degree_sequence_r)
                deg_r, cnt_r = list(zip(*list(degreeCount_r.items())))        
                axarr_r[x,y].bar(deg_r, cnt_r, width=0.80, color='r')
                axarr_r[x,y].set_xlim([0,len(G.nodes())])
                #axarr_r[x,y].set_xlim([0,len(G.nodes())])
                if (x==0 and y==0):
                    axarr_r[x,y].set_title("Rand k=%i,p=%.2f %s - T=%i"%(COMMITTEE_SZ,CLOSURE_PARAM,name,i))
                else:
                    axarr_r[x,y].set_title("T=%i"%i)

                y += 1 
                if (y >= h):
                    y=0
                    x += 1
        break    
    print("Done")

    #Show plot from one trial
    plt.show()
"""
