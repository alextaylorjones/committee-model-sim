#!/usr/bin/python
import networkx as nx
import collections
import math
import numpy as np
import random as rand
import copy
import matplotlib.pyplot as plt
from itertools import combinations as comb

from os import listdir as ld
#Load data

def get_mutual_contact_density(G,x,y):
    #JS Metric
    xNbr = set(G[x].keys())
    yNbr = set(G[y].keys())
    
    if (len(xNbr.union(yNbr)) == 0):
        print "No contacts for  pair",x,y
        return 1.0

    return float(len(xNbr.intersection(yNbr)))/float(len(xNbr.union(yNbr)))
    #return float(len(xNbr.intersection(yNbr)))
     

if __name__=="__main__":
    CONTACT_COL = 4 #indexed from 0
    AWARENESS_COL = 8 #indexed from 0

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
                contact_graph.add_edge(int((line.split())[0]),int((line.split())[1]))

            if ((line.split())[AWARENESS_COL] == '1'):
                awareness_graph.add_edge(int((line.split())[0]),int((line.split())[1]))
        print(("Adding graphs ", data_file.name, "with ", len(contact_graph.nodes())," nodes and ", len(contact_graph.edges()), " edges"))
        contact_graphs.append((contact_graph,data_file.name))
        awareness_graphs.append((awareness_graph,data_file.name))

        #DEBUG
        
        
                
                
    #close files
    for data_file in data_files:
        data_file.close()


    #end load data

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
        

            mutual_nbrs = len((set(contact_graph[x].keys()).intersection(set(contact_graph[y].keys())).difference(set([x,y]))))
            if (mutual_nbrs > 0):

                aware = 0
                if (x in awareness_graph[y].keys() and y in awareness_graph[x].keys()):
                    aware = 1

                dis_2[(x,y)] = (get_mutual_contact_density(contact_graph,x,y),aware)

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



        
        
        X1 = np.arange(0.0,1.1,0.1)
        X2 = np.arange(0.1,1.2,0.1)

        X_range = [(X1[i],X2[i]) for _ in X1]
        Y = [[] for _ in X_range]

        for k in dis_1.keys():
            mutual_nbr_ratio = dis_1[k][0]
            for i,xr in enumerate(X_range):
                x_lo = xr[0]
                x_hi = xr[1]            
                if (x_lo <= mutual_nbr_ratio and x_hi >= mutual_nbr_ratio):
                    #add one to list of awareness relationship status
                    print "Adding awareness of", dis_1[k][1]," for pair",k," with mutual ratio",mutual_nbr_ratio, "in range",xr
                    Y[i].append(dis_1[k][1]) 


        #average the awareness counts

        plt.plot(X1,Y)
        plt.show()

