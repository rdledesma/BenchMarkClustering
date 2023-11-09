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

# d = d.dropna()
d = d[(d.CTZ>0)]


vi = []


import numpy as np
for Y in d.date.dt.year.unique():
    dfil = d[d.date.dt.year==Y]
    for N in dfil.date.dt.dayofyear.unique():
        dff = dfil[dfil.date.dt.dayofyear== N]
        dff['GHIK-1'] = dff.GHI.shift(+1)
        dff['med1'] = ((dff.GHI - dff['GHIK-1']) ** 2 + 1)**0.5
        dff['GHIargpK-1'] = dff['Clear sky GHI'].shift(+1)
        dff['med2'] = ((dff['Clear sky GHI'] - dff['GHIargpK-1']) ** 2 + 1)**0.5
        
        dff['vi'] = np.where(dff.med1.isna() | dff.med1.isna() ,np.nan,dff.med1/dff.med2  )
        
        # if len(dff.vi.dropna()>0):
        #     dff['vi'] = dff.vi/max(dff.vi.dropna()) 
        
        for x in dff.vi:
            vi.append(x)


d['vi'] = vi


daily = d.groupby(d['date'].dt.date).mean().reset_index()


daily['date'] = pd.to_datetime(daily.date)
daily['vi'] = daily.vi / daily.vi.max()




# daily['GHIK-1'] = daily.ghi.shift(+1)
# daily['GHIccK-1'] = daily['Clear sky GHI'].shift(+1)
# daily['med1'] = ((daily.ghi - daily['GHIK-1']) ** 2 + 1)**0.5
# daily['med2'] = ((daily['Clear sky GHI'] - daily['GHIccK-1']) ** 2 + 1)**0.5
# daily['vi'] = np.where(daily.med1.isna() | daily.med1.isna() ,np.nan,daily.med1/daily.med2  )



# import matplotlib.pyplot as plt
# plt.plot(daily.vi, daily.kc, 'o', alpha=0.2)
# plt.plot(1, 0.9, 'o', color="red")
# plt.plot(0.7, 0.15, 'o', color="green")
# plt.plot(2, 0.6, 'o', color="black")
# plt.plot(15, 0.7, 'o', color="pink")



# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans 

# daily = daily.dropna()

# centroides = [[1,0.9],[0.7,0.15],[2,0.6],[15,0.7]]

# kmeans = KMeans(n_clusters=4, init=centroides)
# kmeans.fit(daily[['vi','kc']])

# plt.scatter(daily.vi,daily.kc, c=kmeans.labels_)
# # plt.plot(kmeans.cluster_centers_, '.r')
# plt.plot(11, 0.58, 'o', color="pink")
# plt.plot(7, 0.57, 'o', color="pink")
# plt.plot(2, 0.6, 'o', color="black")
# plt.plot(1, 0.9, 'o', color="red")
# plt.plot(0.7, 0.15, 'o', color="green")
# plt.show() 



import numpy as np
import matplotlib.pyplot as plt

# Supongamos que tienes un DataFrame llamado "df" con columnas "vi" y "kc"

# Puntos dados en el array
puntos = np.array([[0.1, 0.1], [0.15, 1], [0.3, 0.6], [0.8, 0.7]])

# Asignar cada punto del dataframe al punto más cercano en el array
daily['punto_cercano'] = [np.argmin(np.linalg.norm(puntos - np.array([vi, kc]), axis=1)) for vi, kc in zip(daily['vi'], daily['kc'])]

# Plotear los grupos
for i in range(len(puntos)):
    grupo = daily[daily['punto_cercano'] == i]
    plt.scatter(grupo['vi'], grupo['kc'], label=f'Punto {i+1}')

# Plotear los puntos dados
plt.scatter(puntos[:, 0], puntos[:, 1], color='red', marker='x', label='Puntos dados')

# Añadir etiquetas y leyenda
plt.xlabel('vi')
plt.ylabel('kc')
plt.legend()
plt.title('Grupos de pares vi y kc')

# Mostrar el gráfico
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Supongamos que tienes un DataFrame llamado "df" con columnas "vi" y "kc"

# Puntos dados en el array
puntos = np.array([[0.1, 0.1], [0.15, 1], [0.3, 0.6], [0.8, 0.7]])

# Etiquetas para los puntos
etiquetas = {
    (0.1, 0.1): 'dia nublado',
    (0.15, 1): 'dia claro',
    (0.3, 0.6): 'dia variable',
    (0.8, 0.7): 'día altamente variable'
}

# Colores para los puntos en el array
colores_puntos = ['blue', 'green', 'purple', 'orange']

# Asignar cada punto del dataframe al punto más cercano en el array
daily['punto_cercano'] = [np.argmin(np.linalg.norm(puntos - np.array([vi, kc]), axis=1)) for vi, kc in zip(daily['vi'], daily['kc'])]

# Plotear los grupos y guardar el color asignado
colores_asignados = []

for i in range(len(puntos)):
    grupo = daily[daily['punto_cercano'] == i]
    color = colores_puntos[i]
    colores_asignados.append(color)
    plt.scatter(grupo['vi'], grupo['kc'],  color=color)

# Plotear los puntos dados con etiquetas y el color de su grupo
for punto, etiqueta in etiquetas.items():
    idx = np.argmin(np.linalg.norm(puntos - np.array(punto), axis=1))
    color_asignado = colores_asignados[idx]
    plt.scatter(punto[0], punto[1], color=color_asignado, marker='x', label=f'{etiqueta}')

# Añadir etiquetas y leyenda
plt.xlabel('vi')
plt.ylabel('kc')
plt.legend()
plt.title('Grupos de pares vi y kc')

# Mostrar el gráfico
plt.show()





d['cluster'] = pd.merge_asof(d, daily, left_on='date', right_on='date', direction='backward').punto_cercano.values


s  = d[d.date.dt.year == 2015]
s['n'] = s.date.dt.dayofyear
plt.plot(s[s.date.dt.dayofyear == 30].GHI, '-r', label="clase 0")
plt.plot(s[s.date.dt.dayofyear == 1].GHI, '-r')
plt.plot(s[s.date.dt.dayofyear == 1].index, s[s.date.dt.dayofyear == 2].GHI, '-r')
plt.plot(s[s.date.dt.dayofyear == 1].index, s[s.date.dt.dayofyear == 2].GHI, '-r')
plt.plot(s[s.date.dt.dayofyear == 1].index, s[s.date.dt.dayofyear == 24].GHI, '-r')




dia30 = s[s.date.dt.dayofyear == 30].GHI.values
dia1 = s[s.date.dt.dayofyear == 1].GHI.values
dia2 = s[s.date.dt.dayofyear == 2].GHI.values
dia14 = s[s.date.dt.dayofyear == 14].GHI.values


plt.plot(dia30, label="dia nublado")
plt.plot(dia1, label="dia claro")
plt.plot(dia2, label="dia variabble")
plt.plot(dia14, label="altamente variable")
plt.legend()

d[['date', 'TOA', 'Clear sky GHI', 'GHI', 'ghi', 'Mak', 'alpha', 'CTZ',
       'delta', 'kc', 'kcmod', 'kt', 'ktmod', 'cluster']].to_csv('sa_15_stain.csv', index=False)

