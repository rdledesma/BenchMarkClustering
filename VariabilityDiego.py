#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 22:45:22 2023

@author: solar
"""

import pandas as pd
import numpy as np

d = pd.read_csv('sa_15.csv')
d['date'] = pd.to_datetime(d.date)

d = d.dropna()
d = d[(d.CTZ>0)]


vi = []


import numpy as np
for Y in d.date.dt.year.unique():
    dfil = d[d.date.dt.year==Y]
    for N in dfil.date.dt.dayofyear.unique():
        dff = dfil[dfil.date.dt.dayofyear== N]
        dff['GHIK-1'] = dff.ghi.shift(+1)
        dff['med1'] = ((dff.ghi - dff['GHIK-1']) ** 2 + 1)**0.5
        dff['GHIargpK-1'] = dff['Clear sky GHI'].shift(+1)
        dff['med2'] = ((dff['Clear sky GHI'] - dff['GHIargpK-1']) ** 2 + 1)**0.5
        
        dff['vi'] = np.where(dff.med1.isna() | dff.med1.isna() ,np.nan,dff.med1/dff.med2  )
        
        # if len(dff.vi.dropna()>0):
        #     dff['vi'] = dff.vi/max(dff.vi.dropna()) 
        
        for x in dff.vi:
            vi.append(x)


d['vi'] = vi
d['vi'] = d.vi / d.vi.max()

s = d[d.kc<1.3]
s = s.dropna()



from sklearn.neighbors import KernelDensity

# Supongamos que 'df' es tu DataFrame y tiene columnas 'vi' y 'kc'
# X será una matriz con las columWnas 'vi' y 'kc' como características
X = s[['vi', 'kc']]


kde = KernelDensity(bandwidth=0.001)  # Puedes ajustar el ancho de banda según sea necesario
kde.fit(X)

log_densidades =  kde.score_samples(X[:])



# Normaliza los logaritmos de las densidades a valores entre 0 y 1
densidades = np.exp(log_densidades)
densidades = (densidades - densidades.min()) / (densidades.max() - densidades.min())


s['densidad'] = densidades


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(s.kc,s.vi,  s.densidad, c='black', s=0.1)
#ax.scatter(d[d.cdiego == 1].vi, d[d.cdiego == 1].kc, d[d.cdiego == 1].densidad, c='red', s=10)
ax.scatter( 0.94, 0.025, 0.69, c='cyan', s=100)
ax.scatter(0.912, 0.0156,  0.025, c='red', s=100)
ax.scatter(0.15, 0.01,  0.07, c='green', s=100)
ax.scatter(0.766, 0.937,  0.007, c='blue', s=100)
ax.scatter(0.735, 0.539,  0.005, c='gray', s=100)







from sklearn.cluster import KMeans 
centroides = [[0.94,0.025, 0.69 ],
              [0.912, 0.0156,  0.025],[0.15, 0.01,  0.07],
              [0.766, 0.937,  0.007], [0.735, 0.539,  0.005]]

kmeans = KMeans(n_clusters=5, init=centroides)
kmeans.fit(s[['vi','kc', 'densidad']])


clusters = kmeans.predict(s[['vi','kc', 'densidad']])


s['cls'] = clusters


d['cls'] = s.cls



from sklearn.model_selection import train_test_split

X = s[['TOA', 'Clear sky GHI', 'GHI', 'ghi', 'Mak', 'alpha', 'CTZ',
       'delta', 'kc', 'kcmod', 'kt', 'ktmod']]
y = s.cls
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=400)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)


y_pred = rf.predict(X_test)

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rf.classes_)
disp.plot()
plt.show()



d['clsDiego'] = rf.predict(d[['TOA', 'Clear sky GHI', 'GHI', 'ghi', 'Mak', 'alpha', 'CTZ',
       'delta', 'kc', 'kcmod', 'kt', 'ktmod']])




    

s = d[['date', 'TOA', 'Clear sky GHI', 'GHI', 'ghi', 'Mak', 'alpha', 'CTZ',
       'delta', 'kc', 'kcmod', 'kt', 'ktmod', 'clsDiego']].to_csv('sa_15_Diego.csv', index=False)
