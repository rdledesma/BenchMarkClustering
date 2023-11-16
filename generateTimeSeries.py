#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 22:35:25 2023

@author: solar
"""

import pandas as pd

d1 = pd.read_csv('measured/GHIcorregido_2013.csv', sep=";", decimal=',')
d2 = pd.read_csv('measured/GHIcorregido_2014.csv',sep=";", decimal=',')
d3 = pd.read_csv('measured/GHIcorregido_2015.csv', sep=";", decimal=',')

d = pd.concat([d1,d2,d3])

d['']
