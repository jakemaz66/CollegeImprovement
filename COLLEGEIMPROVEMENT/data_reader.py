import pandas as pd
import numpy as np

def data_reader():
    """
    This function takes in the college data, loops over the years, and returns the concatenated
    dataframes with just Duquesne University
    """

    years_list = [
        '1997_98', '1998_99', '1999_00', '2000_01', '2001_02', '2002_03', '2003_04', '2004_05', '2005_06',
        '2006_07', '2007_08', '2008_09', '2009_10', '2010_11', '2011_12', '2012_13', '2013_14', '2014_15',
        '2015_16', '2016_17', '2017_18', '2018_19', '2019_20', '2020_21', '2021_22'
    ]

    columns_of_interest = ['HIGHDEG', 'CCSIZSET', 'ADM_RATE', 'SATVR75', 'SATMT75', 'PPTUG_EF',
                           'NPT44_PRIV', 'TUITIONFEE_IN', 'AVGFACSAL', 'PFTFAC', 'INEXPFTE',
                           'ENRL_ORIG_YR2_RT', 'FIRSTGEN_WDRAW_ORIG_YR3_RT', 'COMPL_RPY_7YR_RT',
                           'GRAD_DEBT_MDN', 'MDEARN_ALL', 'CIPTFBS1', 'MTHCMP1', 'BBRR2_FED_UG_NOPROG',
                           'BOOKSUPPLY', 'ROOMBOARD_ON', 'OTHEREXPENSE_ON', 'ROOMBOARD_OFF', 'OTHEREXPENSE_OFF',
                           'OTHEREXPENSE_FAM', 'STUFACR', 'IRPS_NRA', 'COUNT_NWNE_4YR'
    ]   

    df = pd.read_csv(r'C:\Users\jakem\CollegeImprovement\COLLEGEIMPROVEMENT\data\data\MERGED1997_98_PP.csv')

    dataframes = {}  

    for year in years_list:
        df = pd.read_csv(fr'C:\Users\jakem\CollegeImprovement\COLLEGEIMPROVEMENT\data\data\MERGED{year}_PP.csv')

        df = df[columns_of_interest]

        #Filtering out dataframes for Duquesne
        df = df.loc[df['INSTNM'] == 'Duquesne University']

        dataframes[year] = df

    #Concatenating dataframes
    df = pd.concat(dataframes.values())
    return df

if __name__ == '__main__':
    df = data_reader()

    