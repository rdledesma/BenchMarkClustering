#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:33:09 2023

@author: dario
"""

import pandas as pd
d = pd.read_csv('sa_15_stain.csv')


d['date'] = pd.to_datetime(d.date)
d = d[d.date.dt.year == 2015]

c0 = d[d.cluster == 0]
c1 = d[d.cluster == 1]
c2 = d[d.cluster == 2]
c3 = d[d.cluster == 3]




days = [c0.date.dt.dayofyear.unique()[5],
c1.date.dt.dayofyear.unique()[0],
c2.date.dt.dayofyear.unique()[5],
c3.date.dt.dayofyear.unique()[0]]


d0 =  c0[c0.date.dt.dayofyear == days[0]].ghi.values
d1 =  c1[c1.date.dt.dayofyear == days[1]].ghi.values
d2 =  c2[c2.date.dt.dayofyear == days[2]].ghi.values
d3 =  c3[c3.date.dt.dayofyear == days[3]].ghi.values





import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(1)
sns.relplot( data=d0, kind="line")
sns.relplot( data=d1, kind="line")
sns.relplot( data=d2, kind="line")
sns.relplot( data=d3, kind="line")


plt.figure(2)
plt.plot(d0, label="Overcast")
plt.plot(d1, label="Clear")
plt.plot(d3, label="Variable Part of the Day")
plt.plot(d2, label="Highly Variability All Day")
plt.xlabel("Time")
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.ylabel("GHI w/m²")
plt.legend()




import seaborn as sns
import matplotlib.pyplot as plt

# Configurar el estilo de seaborn
sns.set(style="ticks")

# Crear una figura y ejes
fig, ax = plt.subplots(figsize=(10, 6))

# Graficar los datos con colores predefinidos de seaborn
sns.lineplot(data=d0, label="Overcast", ax=ax, color=sns.color_palette()[0])
sns.lineplot(data=d1, label="Clear", ax=ax, color=sns.color_palette()[1])
sns.lineplot(data=d3, label="Variable Part of the Day", ax=ax, color=sns.color_palette()[2])
sns.lineplot(data=d2, label="Highly Variability All Day", ax=ax, color=sns.color_palette()[3])

# Configurar etiquetas y leyenda
ax.set(xlabel="Time", ylabel="GHI w/m²")
ax.legend()

# Eliminar los ticks del eje x
ax.xaxis.set_major_locator(plt.NullLocator())

# Mostrar el gráfico
plt.show()

