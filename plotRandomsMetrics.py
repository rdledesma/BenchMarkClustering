#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:48:17 2023

@author: dario
"""

import pandas as pd
import matplotlib.pyplot as plt

drmsd = pd.read_csv('randomMetrics/rmsd.csv')
dmae = pd.read_csv('randomMetrics/mae.csv')
dmbe = pd.read_csv('randomMetrics/mbe.csv')


# Calcular la media y el desvío estándar para cada columna
means = drmsd.mean()
stds = drmsd.std()

# Crear el gráfico
plt.figure(figsize=(10, 6))

# Para cada columna, trazar la media y sombrear el rango de un desvío estándar
for column in means.index:
    plt.errorbar(x=column, y=means[column], yerr=stds[column], fmt='o', label=column)

# Añadir etiquetas y leyenda
plt.xlabel('Columnas')
plt.ylabel('Valor Medio')
plt.title('Valores Medios y Rango de Desvío Estándar')
plt.legend()
plt.grid(True)
plt.show()

    
import pandas as pd
import matplotlib.pyplot as plt

# Supongamos que tienes tres DataFrames llamados df1, df2 y df3 con las columnas met_2, met_3, met_4, ...

# Crear un DataFrame combinado con las columnas de interés


# Calcular la media y el desvío estándar para cada columna
means = dmbe.mean()
stds = dmbe.std()

# Crear el gráfico
plt.figure(figsize=(10, 6))

# Para cada columna, trazar la media y sombrear el rango de un desvío estándar
for column in means.index:
    plt.errorbar(x=column, y=means[column], yerr=stds[column], fmt='o', label=column)

# Unir los valores medios con una línea punteada
# Rellenar el área entre los máximos y mínimos con color azul
plt.fill_between(means.index, means - stds, means + stds, color='blue', alpha=0.2)

plt.plot(means.index, means.values, linestyle='--', marker='o', color='black', label='Media')

# Añadir etiquetas y leyenda
plt.xlabel('Columnas')
plt.ylabel('Valor Medio')
plt.title('Valores Medios y Rango de Desvío Estándar')
plt.legend()
# plt.grid(True)
plt.show()





