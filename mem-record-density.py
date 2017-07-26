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
        contact_graph = nx.DiGraph()
        awareness_graph = nx.DiGraph()
        
        #read lines
        for i,line in enumerate(data_file):
            if (len(line.split()) != 10):
                print((line, "line",i," not long enough", data_file))
                continue
            if ((line.split())[CONTACT_COL] == '1'):
                #print "int", (line.split())[CONTACT_COL] 
                contact_graph.add_edge(int((line.split())[0]),int((line.split())[1]),weight=1.0)
            if ((line.split())[AWARENESS_COL] == '1'):
                awareness_graph.add_edge(int((line.split())[0]),int((line.split())[1]),weight=1.0)
        print(("Adding graphs ", data_file.name, "with ", len(contact_graph.nodes())," nodes and ", len(contact_graph.edges()), " edges"))
        contact_graphs.append((contact_graph,data_file.name))
        awareness_graphs.append((awareness_graph,data_file.name))
        
        
                
                
    #close files
    for data_file in data_files:
        data_file.close()


    #end load data

    for i,graph_desc in enumerate(contact_graphs):
        contact_graph,name = graph_desc

        print "Analyzing graph file", name

        #same name, no need to pull that
        awareness_graph = awareness_graphs[i][0]
        
           
        dis_1_density = []
        dis_2_density = []

        #dis 1 pairs
        for x,y in comb(contact_graph.nodes(),2):

            #mutual contact relationships
            if (x in contact_graph.neighbors(y) and y in contact_graph.neighbors(x)):
                aware = 0
                if (x in awareness_graph[y] or y in awareness_graph[x]):
                    aware = 1

                dis_1_density.append((x,y,get_mutual_contact_density(contact_graph,x,y),aware))


        print "Graph ",name
        print "Distance 1 contacts"
        print " Total Number", len(dis_1_density)
        print " Total Awareness", sum([d[3] for d in dis_1_density])
        print "\n\nDensity:", sum([d[3] for d in dis_1_density])/float(len(dis_1_density))


