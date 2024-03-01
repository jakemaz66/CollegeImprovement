import pandas as pd
import numpy as np


years_list = [
    '1997_97', '1998_98', '1999_99', '2000_00', '2001_01', '2002_02', '2003_03', '2004_04', '2005_05',
    '2006_06', '2007_07', '2008_08', '2009_09', '2010_10', '2011_11', '2012_12', '2013_13', '2014_14',
    '2015_15', '2016_16', '2017_17', '2018_18', '2019_19', '2020_20', '2021_21', '2022_22'
]

dataframes = {}  

for year in years_list:
    df = f'MERGED{year}_PP.csv'
    #Filtering out dataframes for Duquesne
    df = df.loc[df['INSTNM'] == 'Duquesne University']

    dataframes[year] = pd.read_csv(df)

#Concatenating dataframes
df = pd.concat(dataframes.values())

    