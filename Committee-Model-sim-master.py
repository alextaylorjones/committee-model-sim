#!/usr/bin/python

import networkx as nx
import collections
import math
import numpy as np
import random as rand
import copy
import matplotlib.pyplot as plt
from itertools import combinations as comb
#constants


#**** Graph building helper functions ****#

#degree k, rewiring probability p
def get_watts_strogatz_graph(nodeNum,k,p):   
    G = nx.watts_strogatz_graph(nodeNum,k,p,seed=SEED)
    for e in G.edges():
        G[e[0]][e[1]]['weight'] = 1
    return G

# Build g
def get_undirected_ER_graph(nodeNum, p):
    G = nx.erdos_renyi_graph(nodeNum,p,seed=SEED)
    for e in G.edges():
        G[e[0]][e[1]]['weight'] = 1
    return G

#Generate tree with powerlaw degree distribution
def get_powerlaw_tree_graph(nodeNum,gamma):
    G = nx.random_powerlaw_tree(nodeNum,gamma,seed=SEED)
    for e in G.edges():
        G[e[0]][e[1]]['weight'] = 1
    return G


#Graph helper functions

def draw_graph_helper(H,positionFlag="spring",drawNodeLabels=True,drawEdgeLabels=False,pos=None):
    
    if (positionFlag.startswith("spring") and pos==None):
        pos=nx.spring_layout(H,iterations=20)
    if (positionFlag.startswith("random") and pos==None):
        pos=nx.random_layout(H)
    plt.figure(figsize=(20,20))
    
    nx.draw_networkx_nodes(H,pos)
    nx.draw_networkx_edges(H,pos)
    if (drawNodeLabels):
        nx.draw_networkx_labels(H,pos,fontsize=12)
    
    labels = nx.get_edge_attributes(H,'weight')
    labelsNonZero = {}
    for l in labels.keys():
        if labels[l] > 0.0001:
            labelsNonZero[l] = labels[l]
            
    if (drawEdgeLabels):
        nx.draw_networkx_edge_labels(H,pos,edge_labels=labelsNonZero)
    
    plt.show()
    return pos


#Construct a graph whose edge weights are the likelihood of an awareness relationship, given the contact graph G
#under the JS spline metric model
# --> This is a symetric measure,for unweighted graphs 
def construct_awareness_from_contact_graph(G):
    H = G.copy()
    #print "Edges",H.edges()
    for pair in comb(G.nodes(),2): #get pairs
        x = pair[0]
        y = pair[1]
       
        metric = 0.0
        if (y in G[x].keys()):#if y a neighor of x (and visa versa)
            metric = 0.5
            
        #JS Metric
        xNbr = set(G[x])
        yNbr = set(G[y])
        
        #if they have no mutual neighbors
        if (len(xNbr) == 0 and len(yNbr) == 0):
            H.add_edge(x,y,weight=metric)
            continue
            
        mutualNbrs = xNbr.intersection(yNbr)
        
        #special case of isolated dyad
        if (xNbr == set([y]) and yNbr == set([x])):
            metric = 1.0
        else:
            metric = metric + len(mutualNbrs)/float(len(xNbr.union(yNbr)))
        
        #set as edge weight in new graph
        if (y not in H[x]):
            H.add_edge(x,y)
        H[x][y]['weight'] = metric
        
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
            covg[b] = 1 - (1-covg[b])*(1-G[n][b]['weight'])
                
    #return the sum of all of the coverage weights
    return sum(covg)
    
# Find approximation max coverage set using k-greedy method, with t total nodes in solution
def greedy_expected_max_coverage_set(G,t,k):
    #Debug
    assert(t < len(G.nodes()))
    
        #solution
    soln = set()
           
    #node set
    remaining_nodes = set(G.nodes())
    
    #Construct how many nodes to select at each step
    k_vals = [k for _ in range(int(math.floor(float(t)/k)))]
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
## Close all triads in committee
def committee_closure_augmentation(G,committee,closure_threshold):
    H = G.copy()
    #print "Closing committee", committee 
    for x,y in comb(committee,2):
        #add committee-committee edge
        H.add_edge(x,y,weight=1.0)
        
        #Close triads which have two committee members
        for n in set(G[x].keys()).union(G[y].keys()).difference(set([x,y])):
            
            if (n not in G[y]): #then unclosed triad, missing edge n-y
                r = rand.random()
                if (r<closure_threshold):
                    H.add_edge(n,y,weight=1)
            elif (y not in G[x]): #unclosed triad missing edge n-x
                r = rand.random()
                if (r<closure_threshold):
                    H.add_edge(n,x,weight=1)
    
    #Close triads which have one committee member
    for x in committee:
        for y in set(G[x].keys()).difference(set(committee)):
            for z in set(G[y].keys()).difference(set(committee)):
                if (z not in G[x].keys()): #if unclosed triad
                    r = rand.random()
                    if (r<closure_threshold):
                        H.add_edge(n,x,weight=1.0)
    return H
    

#Plot distribution of times at which given network is covered at alpha fraction by k nodes
def get_distribution_coverage_time(G_init,k,alpha,closure_param,trials=100,max_tries=100,alg="greedy",draw_freq=0,show_ecc=False):
    time_dist = []
    num_nodes = len(G_init.nodes())
   
    committee_coverage = []
    max_committee_coverage = []
    committees = []
    graphs = []
    
    for t in range(trials):
        G = G_init.copy()
        graphs.append([])
        committees.append([])
        committee_coverage.append( [])
        max_committee_coverage.append([])

        pos = None
        tries = 0
        if (alg.startswith("greedy")):
            #Greedy coverage algorithm
            committee,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
            committee_coverage[-1].append(covg)
            max_committee_coverage[-1].append(covg)
            committees[-1].append(committee)


            print "T=0"
            if (draw_freq != 0):
                pos = draw_graph_helper(G,"spring",pos)
            graphs[-1].append(G)
            
                            #if showing eccentricities
            if (show_ecc):
                ecc_dict = nx.eccentricity(G)
                plt.hist(ecc_dict.values(),bins=range(max(ecc_dict.values())+2))
                plt.title("Iteration 0 of Greedy Committee Formation:\n Contact Network Eccentricity Distribution")
                plt.show()
            
            while (covg < alpha*num_nodes and tries < max_tries):
                G = committee_closure_augmentation(G,committee,closure_param)
                graphs[-1].append(G)
            
                #print "T=%i" % (tries+1)
                
                #draw according to frequency
                if (draw_freq != 0 and (tries+1) % draw_freq == 0):
                    pos = draw_graph_helper(G,"spring",pos)
                
                #if showing eccentricities
                if (show_ecc):
                    ecc_dict = nx.eccentricity(G)
                    plt.hist(ecc_dict.values(),bins=range(max(ecc_dict.values())+2))
                    plt.title("Iteration %i of Greedy Committee Formation:\n Contact Network Eccentricity Distribution" % (tries+1))
                    plt.show()
                
                tries += 1
             
                committee,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
                committees[-1].append(committee)
                committee_coverage[-1].append(covg)
                max_committee_coverage[-1].append(covg)

       
                #print "Chose commmittee", committee, "with coverage ", covg
        elif (alg.startswith("random")):
            #Greedy coverage algorithm
            
            committee = np.random.choice(G.nodes(),k)
            
            #ensure uniqueness
            while (len(committee) != len(set(committee))):
                 committee = np.random.choice(G.nodes(),k)
            
            committees[-1].append(committee)
                    
            covg = get_exp_coverage(G,committee)


            committee_coverage[-1].append(covg)

            #Calculate coverage of greedily selected committee
            _,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
            
            max_committee_coverage[-1].append(covg)

            if (draw_freq != 0):
                #print "T=",0
                pos = draw_graph_helper(G,"spring",pos)
                
            graphs[-1].append(G)
            
            #if showing eccentricities
            if (show_ecc):
                ecc_dict = nx.eccentricity(G)
                plt.hist(ecc_dict.values(),bins=range(max(ecc_dict.values())+2))
                plt.title("Iteration 0 of Random Committee Formation:\n Contact Network Eccentricity Distribution" )
                plt.show()
                    
            while (covg < alpha*num_nodes and tries < max_tries):
            
                #augment with random committee
                G = committee_closure_augmentation(G,committee,closure_param)

                #Get coverage quality under greedy method
                _,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
            
                max_committee_coverage[-1].append(covg)
               
                
                #draw according to frequency
                if (draw_freq != 0 and (tries+1) % draw_freq == 0):
                    #print "T=",tries+1
                    pos = draw_graph_helper(G,"spring",pos)
                
                #if showing eccentricities
                if (show_ecc):
                    ecc_dict = nx.eccentricity(G)
                    plt.hist(ecc_dict.values(),bins=range(max(ecc_dict.values())+2))
                    plt.title("Iteration %i of Random Committee Formation:\n Contact Network Eccentricity Distribution" % (tries+1))
                    plt.show()
                
                graphs[-1].append(G)
                
                #draw_graph_helper(G,"spring")
                tries += 1
                
                #randomly choose committee
                committee = np.random.choice(G.nodes(),k)
                
                #ensure uniqueness
                while (len(committee) != len(set(committee))):
                     committee = np.random.choice(G.nodes(),k)

                committees[-1].append(committee)

                covg = get_exp_coverage(G,committee)
                committee_coverage[-1].append(covg)


        
        if (covg < alpha*(len(G.nodes()))):
            print "ERROR: Coverage was %f, not high enough after Max tries= %i exceeded" % (covg,max_tries)
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
for filename in ld("./data/"):
    if (filename.endswith(".txt")):
        data_files.append(open("./data/%s" % filename))

    


#construct graphs
for data_file in data_files:
    contact_graph = nx.Graph()
    awareness_graph = nx.Graph()
    
    #read lines
    for i,line in enumerate(data_file):
        if (len(line.split()) != 10):
            print line, "line",i," not long enough", data_file
            continue
        if ((line.split())[CONTACT_COL] == '1'):
            #print "int", (line.split())[CONTACT_COL] 
            contact_graph.add_edge(int((line.split())[0]),int((line.split())[1]),weight=1.0)
        if ((line.split())[AWARENESS_COL] == '1'):
            awareness_graph.add_edge(int((line.split())[0]),int((line.split())[1]),weight=1.0)
    print "Adding graphs ", data_file.name, "with ", len(contact_graph.nodes())," nodes and ", len(contact_graph.edges()), " edges"
    contact_graphs.append((awareness_graph,data_file.name))
    awareness_graphs.append((awareness_graph,data_file.name))
    
    
            
            
#close files
for data_file in data_files:
    data_file.close()


# In[8]:

"""#Draw graphs
for g,name in contact_graphs:
    print "Name: ",name
    draw_graph_helper(g,"spring")
"""


# In[9]:


#standard data
single_contact_networks_list = [(g,name) for g,name in contact_graphs]


"""
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

# In[48]:


#Take
def plot_comp_results(graph_name,data1,data1name,data2,data2name,plt_name,plt_type='avg'):

    all_data = [(data1,data1name),(data2,data2name)]
    plt.figure()

    #Aggregate data1 and data2
    if (plt_type.startswith('avg')):
        #Iterate through each kind of data
        for data,name in all_data:
 
            data_avg = []

            for record in data:
                #Store records in data_avg
                for i,entry in enumerate(record):
                    if (i+1 > len(data_avg)):
                        data_avg.append([entry])
                        assert(len(data_avg) == i+1)
                    else:
                        data_avg[i].append(entry)

            print data_avg
            to_plot = []
            #Average entries
            for entry in data_avg:
                to_plot.append(float(sum(entry))/len(entry))

            #Plot data average
            print name,to_plot
            plt.plot(range(len(to_plot)),to_plot,label=name)
        

    #Create legend from data name entries
    #plt.legend(handles=[ entry[1] for entry in all_data])
    plt.title("Graph: %s" % graph_name)
    plt.legend()
    plt.xlabel("Iteration Count")
    plt.ylabel("Average Value of %s" % plt_name)

    #plt.show()

def list_local_bridges(G):
    local_bridges = []
    for v in G.nodes():
        nbrs = G[v].keys()
        twoHopNbrs = []
        for w in nbrs:
            twoHopNbrs = twoHopNbrs + G[w].keys()
        
        for n in nbrs:
            if (n not in twoHopNbrs):
                local_bridges.append((v,n))
    return local_bridges 
#Find closure time on real graphs



#G_list = combined_contact_networks_list
G_list = single_contact_networks_list



"""
print "Starting simulation"
print "Coverage minimum fraction %.2f, committee size %i and closure prob %.2f" % (COVERAGE_MIN,COMMITTEE_SZ,CLOSURE_PARAM)



plt.rcParams.update({'font.size': 14})



rand_results = {}
greedy_results = {}

for i,graph_desc in enumerate(G_list):
  
    if (i==0):
        continue 
    print graph_desc
    G,name = graph_desc
    G = nx.convert_node_labels_to_integers(G)
    data_greedy = []
    data_rand = []
    
    #repeat process several times, adding all times to one list
    for c in range(repeats):
        print "Using greedy method"

        t,graphs,committee,covg,max_covg = get_distribution_coverage_time(G,COMMITTEE_SZ,COVERAGE_MIN,CLOSURE_PARAM,trials,max_tries=int(2.75*len(G.nodes()))/COMMITTEE_SZ,alg="greedy",draw_freq=0,show_ecc=False)

        data_greedy.append((t,graphs,covg,max_covg,committee))
        #data_greedy.append(tuple(t,graphs) for t,graphs in get_distribution_coverage_time(G,COMMITTEE_SZ,COVERAGE_MIN,CLOSURE_PARAM,trials,max_tries=N/COMMITTEE_SZ,alg="greedy",draw_freq=1))
        print "Using random method"
        t,graphs,committee,covg,max_covg =  get_distribution_coverage_time(G,COMMITTEE_SZ,COVERAGE_MIN,CLOSURE_PARAM,trials,max_tries=int(2.75*len(G.nodes()))/COMMITTEE_SZ,alg="random",draw_freq=0,show_ecc=False)
        data_rand.append((t,graphs,covg,max_covg,committee))

    #print "Data Greedy = " ,data_greedy
    
    #copy time distribution of convergence
    time_dist_greedy = []
    time_dist_rand = []
  
    rand_results[name] = {}
    greedy_results[name] = {}
    
   
    rand_results[name]['diameter'] = [] 
    rand_results[name]['num_edges'] = [] 
    rand_results[name]['coverage'] = [] 
    rand_results[name]['max_coverage'] = [] 
    rand_results[name]['clustering'] = [] 
    rand_results[name]['local_bridges'] = [] 
    rand_results[name]['committee'] = [] 

    greedy_results[name]['diameter'] = [] 
    greedy_results[name]['num_edges'] = [] 
    greedy_results[name]['coverage'] = [] 
    greedy_results[name]['max_coverage'] = [] 
    greedy_results[name]['clustering'] = [] 
    greedy_results[name]['local_bridges'] = [] 
    greedy_results[name]['committee'] = [] 

 
    #f = plt.figure("%s : greedy" % name)
    
    for record in data_greedy:
        #Save times
        times = record[0]
        time_dist_greedy = time_dist_greedy + times
        
        #Record diameter trends
        graph_trials = record[1]
        for trial in graph_trials:
            diameter = []
            edges = []
            clustering = []
            local_bridges = []
            degrees = []
            #print "Checking graphs ", trial, "for trial" ,graph_trials
            for graph in trial:
                #print "Graph edges",len(graph.edges())
                diameter.append(nx.diameter(graph))
                edges.append(len(graph.edges()))
                clustering.append(nx.average_clustering(graph))
                local_bridges.append(len(list_local_bridges(graph))/2)
                    
            greedy_results[name]['diameter'].append(diameter)
            greedy_results[name]['num_edges'].append(edges)
            greedy_results[name]['clustering'].append(clustering)
            greedy_results[name]['local_bridges'].append(local_bridges)

        #Record coverage
        greedy_results[name]['coverage'].append(record[2])
        greedy_results[name]['max_coverage'].append(record[3])

     
    print "Times recorded for greedy are ",time_dist_greedy
    
    #plt.show()

    
    #f = plt.figure("%s : random" % name)
    
    for record in data_rand:
        #Save times
        times = record[0]
        time_dist_rand = time_dist_rand + times
        
        #Record diameter trends
        graph_trials = record[1]
        for trial in graph_trials:
            diameter = []
            edges = []
            clustering = []
            local_bridges = []
            for graph in trial:
                diameter.append(nx.diameter(graph))
                edges.append(len(graph.edges()))
                clustering.append(nx.average_clustering(graph))
                local_bridges.append(len(list_local_bridges(graph))/2)

            rand_results[name]['diameter'].append(diameter)
            rand_results[name]['num_edges'].append(edges)
            rand_results[name]['clustering'].append(clustering)
            rand_results[name]['local_bridges'].append(local_bridges)

        #Record coverage
        rand_results[name]['coverage'].append(record[2])
        rand_results[name]['max_coverage'].append(record[3])

    print "Times recorded for random are ",time_dist_rand
    
    #plt.legend(handles=[line_greedy, line_rand])
    #plt.show()
   
    #print "Showing data raw"
    #print greedy_results[name]
    #print rand_results[name]
    

    plot_comp_results("%s - k=%i - closure=%.2f" %(name,COMMITTEE_SZ,CLOSURE_PARAM),greedy_results[name]['diameter'],'greedy',rand_results[name]['diameter'],'random','Diameter' )
    plot_comp_results("%s - k=%i - closure=%.2f" %(name,COMMITTEE_SZ,CLOSURE_PARAM),greedy_results[name]['num_edges'],'greedy',rand_results[name]['num_edges'],'random','Num Edges')
    plot_comp_results("%s - k=%i - closure=%.2f" %(name,COMMITTEE_SZ,CLOSURE_PARAM),greedy_results[name]['clustering'],'greedy',rand_results[name]['clustering'],'random','Average Contact Network Clustering Coefficient')
    plot_comp_results("%s - k=%i - closure=%.2f" %(name,COMMITTEE_SZ,CLOSURE_PARAM),greedy_results[name]['coverage'][0],'greedy',rand_results[name]['coverage'][0],'random','Committee Coverage')
    plot_comp_results("%s - k=%i - closure=%.2f" %(name,COMMITTEE_SZ,CLOSURE_PARAM),greedy_results[name]['max_coverage'][0],'greedy',rand_results[name]['max_coverage'][0],'random','Est. Max Committee Coverage')
    plot_comp_results("%s - k=%i - closure=%.2f" %(name,COMMITTEE_SZ,CLOSURE_PARAM),greedy_results[name]['local_bridges'],'greedy',rand_results[name]['local_bridges'],'random','Local Bridges')

    break

plt.show()
"""

if __name__ == "__main__":
    
    # Plot degree distributions
    trials = 5
    iterations = 30
    frequency = 5
    COMMITTEE_SZ = 8
    COVERAGE_MIN = 0.9999
    CLOSURE_PARAM = 0.1

    h = 2
    w = int(math.ceil(iterations/2/float(frequency)))+1

    for graph_desc in G_list:

        G,name = graph_desc
        G = nx.convert_node_labels_to_integers(G)
        H = G.copy()
        G_r = G.copy()

        print "\nShowing ",name

        degrees = [[] for _ in range(iterations)]
        degrees_r = [[] for _ in range(iterations)]

        for t in range(trials):
            G = H.copy()
            G_r = H.copy()

            for i in range(iterations):
                #Get Greedy Committee and Update graph
                committee,_ = greedy_expected_max_coverage_set(G,COMMITTEE_SZ,1)
                G = committee_closure_augmentation(G,committee,CLOSURE_PARAM)

                #Get random committee and update graph
                committee_r = np.random.choice(G_r.nodes(),COMMITTEE_SZ)
            
                #ensure uniqueness
                while (len(committee_r) != len(set(committee_r))):
                    committee_r = np.random.choice(G_r.nodes(),COMMITTEE_SZ)
                # update graph
                G_r = committee_closure_augmentation(G_r,committee_r,CLOSURE_PARAM)

                degrees[i] = degrees[i] + G.degree().values()
                degrees_r[i] = degrees_r[i] + G_r.degree().values()
       


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
                deg, cnt = zip(*degreeCount.items())        

                axarr[x,y].bar(deg, cnt, width=0.80, color='g')
                axarr[x,y].set_xlim([0,len(G.nodes())])
                #axarr[x,y].set_ylim([0,len(G.nodes())])
                if (x==0 and y==0):
                    axarr[x,y].set_title("Greedy k=%i,p=%.2f %s - %i"%(COMMITTEE_SZ,CLOSURE_PARAM,name,i))
                else:
                    axarr[x,y].set_title("%i"%i)

                #random
                degree_sequence_r=sorted(degrees_r[i], reverse=True) # degree sequence
                degreeCount_r=collections.Counter(degree_sequence_r)
                deg_r, cnt_r = zip(*degreeCount_r.items())        
                axarr_r[x,y].bar(deg_r, cnt_r, width=0.80, color='r')
                axarr_r[x,y].set_xlim([0,len(G.nodes())])
                #axarr_r[x,y].set_xlim([0,len(G.nodes())])
                if (x==0 and y==0):
                    axarr_r[x,y].set_title("Rand k=%i,p=%.2f %s - %i"%(COMMITTEE_SZ,CLOSURE_PARAM,name,i))
                else:
                    axarr_r[x,y].set_title("%i"%i)

                y += 1 
                if (y >= h):
                    y=0
                    x += 1
    print "Done"
    #Show plot from one trial
    plt.show()
          


