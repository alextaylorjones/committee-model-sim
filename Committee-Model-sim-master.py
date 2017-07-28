#!/usr/bin/python
import networkx as nx
import re
import collections
import math
import numpy as np
import random as rand
import copy
import matplotlib.pyplot as plt
from itertools import combinations as comb
#constants
SEED=4
np.random.seed(SEED)
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

def draw_graph_helper(H,positionFlag="spring",drawNodeLabels=False,drawEdgeLabels=False,pos=None):
    
    if (positionFlag.startswith("spring") and pos==None):
        pos=nx.spring_layout(H,iterations=20)
    if (positionFlag.startswith("random") and pos==None):
        pos=nx.random_layout(H)
    plt.figure(figsize=(20,20))
    
    nx.draw_networkx_nodes(H,pos,node_color='k',alpha=0.8, node_shape='o')
    nx.draw_networkx_edges(H,pos)
    if (drawNodeLabels):
        nx.draw_networkx_labels(H,pos,fontsize=12)
    labels = nx.get_edge_attributes(H,'weight')
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
                if (G[i][j]['weight'] == 1.0):
                   #Coverage is guaranteed
                   log_weight[i,j] = -float('inf') 
                else:
                    log_weight[i,j] += math.log(1.0 - G[i][j]['weight'])
        
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
## Close all triads which have one edge in committee

def committee_closure_augmentation(G,committee,closure_threshold):
    H = G.copy()
    #print "Closing committee", committee 
    for x,y in comb(committee,2):
        #add committee-committee edge
        H.add_edge(x,y,weight=1.0)
        
    for x,y in comb(committee,2):
        #Close triads which have two committee members
        for n in set(G[x].keys()).union(list(G[y].keys())).difference(set([x,y])):
            
            if (n not in G[y]): #then unclosed triad, missing edge n-y
                r = rand.random()
                if (r<closure_threshold):
                    H.add_edge(n,y,weight=1)
            elif (y not in G[x]): #unclosed triad missing edge n-x
                r = rand.random()
                if (r<closure_threshold):
                    H.add_edge(n,x,weight=1)
    return H 

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
            committee_coverage[-1].append(covg/float(num_nodes))
            max_committee_coverage[-1].append(covg/float(num_nodes))
            committees[-1].append(committee)


            print("T=0")
            if (draw_freq != 0):
                pos = draw_graph_helper(G,"spring",pos)
            graphs[-1].append(G)
            
                            #if showing eccentricities
            if (show_ecc):
                ecc_dict = nx.eccentricity(G)
                plt.hist(list(ecc_dict.values()),bins=list(range(max(ecc_dict.values())+2)))
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
                    plt.hist(list(ecc_dict.values()),bins=list(range(max(ecc_dict.values())+2)))
                    plt.title("Iteration %i of Greedy Committee Formation:\n Contact Network Eccentricity Distribution" % (tries+1))
                    plt.show()
                
                tries += 1
             
                committee,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
                committees[-1].append(committee)
                committee_coverage[-1].append(covg/float(num_nodes))
                max_committee_coverage[-1].append(covg/float(num_nodes))

        elif (alg.startswith("random")):
            #Greedy coverage algorithm
            
            committee = np.random.choice(G.nodes(),k)
            
            #ensure uniqueness
            while (len(committee) != len(set(committee))):
                 committee = np.random.choice(G.nodes(),k)
            
            committees[-1].append(committee)
                    
            covg = get_exp_coverage(G,committee)


            committee_coverage[-1].append(covg/float(num_nodes))
            #Calculate coverage of greedily selected committee
            _,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
            
            max_committee_coverage[-1].append(covg/float(num_nodes))
            committee_coverage[-1].append(covg/float(num_nodes))


            if (draw_freq != 0):
                #print "T=",0
                pos = draw_graph_helper(G,"spring",pos)
                
            graphs[-1].append(G)
            
            #if showing eccentricities
            if (show_ecc):
                ecc_dict = nx.eccentricity(G)
                plt.hist(list(ecc_dict.values()),bins=list(range(max(ecc_dict.values())+2)))
                plt.title("Iteration 0 of Random Committee Formation:\n Contact Network Eccentricity Distribution" )
                plt.show()
                    
            while (covg < alpha*num_nodes and tries < max_tries):
            
                #augment with random committee
                G = committee_closure_augmentation(G,committee,closure_param)

                #Get coverage quality under greedy method
                _,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),k,1)
            
                max_committee_coverage[-1].append(covg/float(num_nodes))               
                
                #draw according to frequency
                if (draw_freq != 0 and (tries+1) % draw_freq == 0):
                    #print "T=",tries+1
                    pos = draw_graph_helper(G,"spring",pos)
                
                #if showing eccentricities
                if (show_ecc):
                    ecc_dict = nx.eccentricity(G)
                    plt.hist(list(ecc_dict.values()),bins=list(range(max(ecc_dict.values())+2)))
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
                committee_coverage[-1].append(covg/float(num_nodes))


        
        if (covg < alpha*(len(G.nodes()))):
            print(("ERROR: Coverage was %f, not high enough after Max tries= %i exceeded" % (covg,max_tries)))
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
            print(("",line, "line",i," not long enough", data_file))
            continue
        if ((line.split())[CONTACT_COL] == '1'):
            #print "int", (line.split())[CONTACT_COL] 
            contact_graph.add_edge(int((line.split())[0]),int((line.split())[1]),weight=1.0)
        if ((line.split())[AWARENESS_COL] == '1'):
            awareness_graph.add_edge(int((line.split())[0]),int((line.split())[1]),weight=1.0)
    print(("Adding graphs ", data_file.name, "with ", len(contact_graph.nodes())," nodes and ", len(contact_graph.edges()), " edges"))
    name = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', data_file.name)[3]
    print(("Downloaded ",name))
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


def plot_comp_results(plt_name,all_data,metric_name,plt_type='avg',saveName=None,noNewFigure=False):


    print("\n\b Plotting ",all_data)
    if (plt_type.startswith('avg')):
 
        if (noNewFigure == False):
            plt.figure()

       
        for alg_name in list(all_data.keys()):
            
            #Iterate through each kind of data
            for data in all_data[alg_name][metric_name]:
                data_avg = []

                for record in data[metric_name]:
                    #Store records in data_avg
                    for i,entry in enumerate(record):
                        if (i+1 > len(data_avg)):
                            data_avg.append([entry])
                            assert(len(data_avg) == i+1)
                        else:
                            data_avg[i].append(entry)

                to_plot = []
                #Average entries
                for entry in data_avg:
                    to_plot.append(float(sum(entry))/len(entry))

                #Plot data average
                print(("Plottting data",metric_name," for alg",alg_name,":",to_plot))
                plt.plot(list(range(len(to_plot))),to_plot,label=alg_name)
            
        plt.title("Graph: %s" % graph_name)
        plt.legend()
        plt.xlabel("Iteration Count")
        plt.ylabel("Average Value of %s" % plt_name)
        
        if (saveName != None):
            plt.savefig("plots/%s.png"%saveName,bbox_inches='tight')
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



G_list = single_contact_networks_list

alg_list= ['greedy','random']

metric_list = ['diameter','num_edges','coverage',  'max_coverage','max_second_coverage','clustering','local_bridges','committee']

trials = 2 #metrics
iterations = 20 #degree dist
frequency = 5
COMMITTEE_SZ = 6
COVERAGE_MIN = 0.9999
CLOSURE_PARAM = 0.08
max_tries = 3


print("Starting simulation")
print(("Coverage minimum fraction %.3f, committee size %i and closure prob %.2f" % (COVERAGE_MIN,COMMITTEE_SZ,CLOSURE_PARAM)))

plt.rcParams.update({'font.size': 14})

#Store result
results = {}

for z,graph_desc in enumerate(G_list):

    if (z not in [4]):
        continue

    G,name = graph_desc
    print(("G=%s has" %name, len(G.nodes()), " nodes"))
    G = nx.convert_node_labels_to_integers(G)
    data_set = []
    
    #repeat process several times, adding all times to one list
    for alg in alg_list:
        print(("Using %s method"%alg))
        t,graphs,committee,covg,max_covg = get_distribution_coverage_time(G,COMMITTEE_SZ,COVERAGE_MIN,CLOSURE_PARAM,trials,max_tries,alg=alg,draw_freq=0,show_ecc=False)
        data_set.append((t,graphs,covg,max_covg,committee,alg))

    print(data_set)
    #Post process graphs
    for data in data_set:
        #DEBUG 
        #copy time distribution of convergence
        time_dist = []
        alg = data[-1] 

        print("\nFull data for ",alg,data)
        results[name] = {}
        results[name][alg] = {}
        

        results[name][alg]['diameter'] = [] 
        results[name][alg]['num_edges'] = [] 
        results[name][alg]['coverage'] = [] 
        results[name][alg]['max_coverage'] = [] 
        results[name][alg]['max_second_coverage'] = [] 
        results[name][alg]['clustering'] = [] 
        results[name][alg]['local_bridges'] = [] 
        results[name][alg]['committee'] = [] 

        print(("Calculating stats (%s)"%alg))   

        #for x,record in enumerate(data):
        #Save times
        times = data[0]
        time_dist = time_dist + times
    
        #Record diameter trends
        graph_trials = data[1]

        #Record committees
        committee_lists = data[4]

        for y,trial in enumerate(graph_trials):
            diameter = []
            edges = []
            clustering = []
            local_bridges = []
            degrees = []
            max_second_covg = []
            print(("Checking graphs in trial", trial))
            for i,graph in enumerate(trial):
                #print "Graph edges",len(graph.edges())
                diameter.append(nx.diameter(graph))
                edges.append(len(graph.edges()))
                clustering.append(nx.average_clustering(graph))
                local_bridges.append(len(list_local_bridges(graph))/2)
                
                #Remove committee from graph and retry
                committee = committee_lists[y][i]
                #print "Coverage at time ",i,"of first set is ",covg, " and should be ",record[2][y][i]
                G= graph.copy() 
                for c in committee:
                    G.remove_node(c)
                G = nx.convert_node_labels_to_integers(G)
                _,covg = greedy_expected_max_coverage_set(construct_awareness_from_contact_graph(G),COMMITTEE_SZ,1)
                #print(("Adding coverage value",covg))
                max_second_covg.append(covg/len(G))
                
    
                

                results[name][alg]['diameter'].append(diameter)
                results[name][alg]['num_edges'].append(edges)
                results[name][alg]['clustering'].append(clustering)
                results[name][alg]['local_bridges'].append(local_bridges)
                results[name][alg]['max_second_coverage'].append(max_second_covg)
            

            #Record coverage
            results[name][alg]['coverage'].append(data[2])
            results[name][alg]['max_coverage'].append(data[3])

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
                    
                results[name][alg]['committee'].append(committee_remain)
                
         
        #Plot data
        for metric in metric_list:
            plot_comp_results("%s - k=%i - closure=%.2f" %(name,COMMITTEE_SZ,CLOSURE_PARAM),results[name],metric,saveName="%s-%s"%(metric,name),noNewFigure=True)
   

"""

    plt.ylim([-0.1,1.1])
        for alg in alg_list:
            plot_comp_results("%s - k=%i - closure=%.2f" %(name,COMMITTEE_SZ,CLOSURE_PARAM),results[name][alg]['max_second_coverage'],'(greedy) second',greedy_results[name]['alg']['max_coverage'][0],'(greedy) first','(Greedy) Max vs Alt Coverage',saveName="first-second-greedy-%s"%name)

"""
print("Done")
#plt.show()

"""

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

                committee,_ = greedy_expected_max_coverage_set(G,COMMITTEE_SZ,1)

        
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
                    axarr[x,y].set_title("Greedy k=%i,p=%.2f %s - %i"%(COMMITTEE_SZ,CLOSURE_PARAM,name,i))
                else:
                    axarr[x,y].set_title("%i"%i)

                #random
                degree_sequence_r=sorted(degrees_r[i], reverse=True) # degree sequence
                degreeCount_r=collections.Counter(degree_sequence_r)
                deg_r, cnt_r = list(zip(*list(degreeCount_r.items())))        
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
        break    
    print("Done")

    #Show plot from one trial
    plt.show()
"""
