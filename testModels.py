#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 22:14:02 2023

@author: solar
"""

import itertools
import statsmodels.formula.api as smf
import pandas as pd

df = pd.read_csv('sa_15_cony.csv')

# Supongamos que 'df' es tu DataFrame con las variables kt, m, kc, alpha, kt:kc
variables = ['Mak',
       'alpha', 'kcmod', 'ktmod', 'ktmod:kcmod',
       'ktmod:alpha', 'Mak:alpha', 'kcmod:alpha', 'ktmod:Mak', 'Mak:kcmod']

# Generar todas las combinaciones de variables
combinations = []
for r in range(1, len(variables) + 1):
    combinations.extend(itertools.combinations(variables, r))

# Crear y ajustar modelos para cada combinación
models = {}
for combo in combinations:
    formula = 'kt ~ ' + ' + '.join(combo)
    model = smf.glm(formula, data=df).fit()
    models[combo] = model


# Crear un diccionario para almacenar los valores de AIC
aic_values = {}

# Calcular el AIC para cada modelo
for combo, model in models.items():
    aic_values[combo] = model.aic

# Encontrar el modelo con el menor AIC
best_model_combo = min(aic_values, key=aic_values.get)
best_model = models[best_model_combo]
    


# Obtener el nombre de las variables del mejor modelo
variables_mejor_modelo = best_model.model.exog_names[1:]  # Excluyendo el intercepto

# Obtener los coeficientes del mejor modelo
coeficientes = best_model.params[1:]  # Excluyendo el intercepto
intercept = best_model.params[0]

print(f"intercept {intercept}")
# Mostrar las variables y sus coeficientes
for variable, coeficiente in zip(variables_mejor_modelo, coeficientes):
    print(f"Variable: {variable}, Coeficiente: {coeficiente}")
    
    
    

    
