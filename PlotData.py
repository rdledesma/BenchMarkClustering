#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:42:16 2023

@author: solar
"""

import pandas as pd


d = pd.read_csv('sa_15_stain.csv')
d['date'] = pd.to_datetime(d.date)



d = d[d.alpha>7]