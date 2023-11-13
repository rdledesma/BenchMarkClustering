#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:28:35 2023

@author: dario
"""
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv('sa_15_vimod.csv')

d = d[['date', 'TOA', 'Clear sky GHI', 'GHI', 'ghi', 'Mak', 'alpha', 'CTZ',
       'delta', 'kc', 'kcmod', 'kt', 'ktmod', 'ktmod:kcmod', 'ktmod:alpha',
       'Mak:alpha', 'kcmod:alpha', 'ktmod:Mak', 'Mak:kcmod', 'f0', 'Train',
       'vi', 'cc', 'vin']]


for n in range(2,11):
        
    X = d[['vi','kcmod']]
    
    kmeans = KMeans(n_clusters=n).fit(X)
    
    centroids = kmeans.cluster_centers_
    
    
    # Predicting the clusters
    labels = kmeans.predict(X)
    # Getting the cluster centers
    C = kmeans.cluster_centers_
    # colores=['red','green','blue','gray','yellow']
    asignar=[]
    pre_clase=[]
    for row in labels:
        # asignar.append(colores[row])
        pre_clase.append(row)
    
    d[f'clus_{n}'] = pre_clase
 


d.to_csv('sa_15_aleatory.csv', index=False)
 
    