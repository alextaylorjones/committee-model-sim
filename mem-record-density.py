        #!/usr/bin/python
import networkx as nx
import collections
import math
import numpy as np
import random as rand
import copy
import matplotlib.pyplot as plt
from itertools import combinations as comb
from scipy.interpolate import interp1d

from os import listdir as ld
#Load data

def get_mutual_contact_density(G,x,y):
    #JS Metric
    xNbr = set(G[x].keys())
    yNbr = set(G[y].keys())
    if (len(xNbr.union(yNbr)) == 0):
        print "No contacts for  pair",x,y
        return 1.0

    #return float(len(xNbr.intersection(yNbr)))/float(len(xNbr.union(yNbr)))
    return float(len(xNbr.intersection(yNbr)))

if __name__=="__main__":
    CONTACT_COL = 4 #indexed from 0
    AWARENESS_COL = 6 #indexed from 0

    contact_graphs = []
    awareness_graphs = []
    data_files = []



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
                print((line, "line",i," not long enough", data_file))
                continue
            if ((line.split())[CONTACT_COL] == '1'):
               #print "int", (line.split())[CONTACT_COL] 
               contact_graph.add_node(int((line.split())[0]))
               contact_graph.add_node(int((line.split())[1]))
               contact_graph.add_edge(int((line.split())[0]),int((line.split())[1]))

            if ((line.split())[AWARENESS_COL] == '1'):
               awareness_graph.add_node(int((line.split())[0]))
               awareness_graph.add_node(int((line.split())[1]))
               awareness_graph.add_edge(int((line.split())[0]),int((line.split())[1]))

        print(("Adding graphs ", data_file.name, "with ", len(contact_graph.nodes())," nodes and ", len(contact_graph.edges()), " edges"))
        contact_graphs.append((contact_graph,data_file.name))
        awareness_graphs.append((awareness_graph,data_file.name))

        #DEBUG
    #close files
    for data_file in data_files:
        data_file.close()

    #DEBUG

    for i,graph_desc in enumerate(contact_graphs):
        graph,name = graph_desc
        awareness_graph = awareness_graphs[i][0]
        prob_mr_lt2 = []
        prob_mr_ge3 = []

        for x,y in comb(graph.nodes(),2):
            try:
                if (nx.shortest_path_length(graph,source=x,target=y) <= 2):
                    if (x in awareness_graph.nodes() and y in awareness_graph[x].keys()):
                        prob_mr_lt2.append(1)
                    else:
                        prob_mr_lt2.append(0)
                else:
                    if (x in awareness_graph.nodes() and y in awareness_graph[x].keys()):
                        prob_mr_ge3.append(1)
                    else:
                        prob_mr_ge3.append(0)
                if (nx.shortest_path_length(graph,source=y,target=x) <= 2):
                    if (y in awareness_graph.nodes() and x in awareness_graph[y].keys()):
                        prob_mr_lt2.append(1)
                    else:
                        prob_mr_lt2.append(0)
                else:
                    if (y in awareness_graph.nodes() and x in awareness_graph[y].keys()):
                        prob_mr_ge3.append(1)
                    else:
                        prob_mr_ge3.append(0)
            except:
                print "Error occured for pair",x,y


        print "Graph name: ",name
        print "\t Pr(MR|dis <= 2)",float(sum(prob_mr_lt2))/len(prob_mr_lt2), " (num= ",sum(prob_mr_lt2), "with ", len(prob_mr_lt2), "total pairs"
        print "\t Pr(MR|dis >= 3)",float(sum(prob_mr_ge3))/len(prob_mr_ge3), " (num= ",sum(prob_mr_ge3), "with ", len(prob_mr_ge3), "total pairs"


    #end load data
    bucket_width = 1
    X = np.arange(0.0,50,bucket_width)

    Y1 = [[] for _ in X]
    Y2 = [[] for _ in X]



    for i,graph_desc in enumerate(contact_graphs):
        contact_graph,name = graph_desc

        print "Analyzing graph file", name
        
        #same name, no need to pull that
        awareness_graph = awareness_graphs[i][0]
        
        
        dis_1 = {}   
        dis_2 = {}
        
        for x in contact_graph.nodes():
            if (x not in awareness_graph.nodes()): 
                print "No awareness neighors for node",x
                contact_graph.remove_node(x)

        #dis 1 pairs
        for x,y in comb(contact_graph.nodes(),2):

            #mutual contact relationships
            if (x in contact_graph.neighbors(y)):
                aware = 0

                if (x in awareness_graph[y].keys()):
                    aware = 1
               
                if ((y,x) not in dis_1.keys()):
                    dis_1[(y,x)] = (get_mutual_contact_density(contact_graph,y,x),aware)
            #mutual contact relationships
            if (y in contact_graph.neighbors(x)):
                aware = 0
                if (y in awareness_graph[x].keys()):
                    aware = 1
               
                if ((x,y) not in dis_1.keys()):
                    dis_1[(x,y)] = (get_mutual_contact_density(contact_graph,y,x),aware)
        

            #mutual_nbrs = len((set(contact_graph[x].keys()).intersection(set(contact_graph[y].keys())).difference(set([x,y]))))

            if (nx.shortest_path_length(contact_graph,source=x,target=y) == 2):
                aware = 0
                if (y in awareness_graph[x].keys()):
                    aware = 1
                dis_2[(x,y)] = (get_mutual_contact_density(contact_graph,x,y),aware)

            if (nx.shortest_path_length(contact_graph,source=y,target=x) == 2):
                aware = 0
                if (x in awareness_graph[y].keys()):
                    aware = 1
                dis_2[(y,x)] = (get_mutual_contact_density(contact_graph,y,x),aware)


        """
        #dis 2 pairs
        for x in contact_graph.nodes():
            for y in contact_graph[x].keys():
                for z in contact_graph[y].keys():
                    if (z == x):
                        continue

                    aware = 0
                    if (x in awareness_graph[z].keys() or z in awareness_graph[x].keys()):
                        aware = 1

                    #Ensure we havent counted this before in distance 1 contacts or as the reverse order in distance 2 contgacts

                    if ((x,z) not in dis_2.keys() and (x,z) not in dis_1.keys()):
                        dis_2[(x,y)] = (get_mutual_contact_density(contact_graph,x,z),aware)
        """

        print "\nGraph ",name
        print "Distance 1 contacts:"
        print " Total Number", len(dis_1)
        print " Total Awareness", sum([d[1] for d in dis_1.values()])
        print "\n\nDensity:", sum([d[1] for d in dis_1.values()])/float(len(dis_1))
        print "Distance 2 contacts:"
        print " Total Number", len(dis_2)
        print " Total Awareness", sum([d[1] for d in dis_2.values()])
        print "\n\nDensity:", sum([d[1] for d in dis_2.values()])/float(len(dis_2))



        
        
        for k in dis_1:
            #print "Entry:",dis_1[k]
            i = 0
            if (dis_1[k][0] < X[0]):
                i = 1
            else:
                while (X[i] <= dis_1[k][0]):
                    i += 1
                    if (i == len(X)):
                        break
            #print "added for ",X[i-1],"and value is",dis_1[k][0]
            Y1[i-1].append(dis_1[k][1])

        for k in dis_2:
            #print "Entry:",dis_2[k]
            i = 0
            if (dis_2[k][0] < X[0]):
                i = 1
            else:
                while (X[i] <= dis_2[k][0]):
                    i += 1
                    if (i == len(X)):
                        break
            #print "added for ",X[i-1],"and value is",dis_2[k][0]
            Y2[i-1].append(dis_2[k][1])

 
        dis_3_total = 0

        dis_3_aware = 0
        for x,y in comb(contact_graph.nodes(),2):
            if (nx.shortest_path_length(contact_graph,source=x,target=y) == 3):
                dis_3_total += 1
                if (y in awareness_graph[x].keys()):
                    dis_3_aware += 1

            if (nx.shortest_path_length(contact_graph,source=y,target=x) == 3):
                dis_3_total += 1
                if (x in awareness_graph[y].keys()):
                    dis_3_aware += 1
        print "Distance 3 total %i : awareness also %i" % (dis_3_total,dis_3_aware)

    #Average resutls
    for i in range(len(X)):
        if (len(Y1[i]) > 1):
            Y1[i] = float(sum(Y1[i]))/float(len(Y1[i]))
        else:
            Y1[i] = None
        if (len(Y2[i]) > 1):
            Y2[i] = float(sum(Y2[i]))/float(len(Y2[i]))
        else:
            Y2[i] = None

    #plot 
    for x in X:
        print x,",",
    print "\n"
    for y in Y1:
        print y, ",",
    print "\n"
    for x in X:
        print x,",",
    print "\n"
    for y in Y2:
        print y, ",",

    print "\n\n\n"
    plt.title("Density of Memory Records")
    plt.xlabel("Fraction of i or j Contacts That Are Mutual")
    plt.ylabel("Likelihood of (i,j) Awareness Edge")
    plt.plot(X,Y1,'bo',label="dis(i,j) = 1")
    #plt.plot(X,[(0.5 + 0.5*x) for x in X],'--',color='b',label="JS ideal, dis(i,j) = 1")

    plt.plot(X,Y2,'r^',label="dis(i,j) = 2")
    #plt.plot(X,[(0.5*x) for x in X],'--',color='r',label="JS ideal, dis(i,j) = 2")
    plt.legend()


    plt.show()

    Y1 = [[]  for _ in X]
    Y2 = [[]  for _ in X]
      
