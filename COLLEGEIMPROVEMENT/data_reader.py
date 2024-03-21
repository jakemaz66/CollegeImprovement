import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def data_reader():
    """
    This function takes in the college data, loops over the years, and returns the concatenated
    dataframes
    """

    #Defining Years and Columns of Interest
    years_list = [
        '2006_07', '2007_08', '2008_09', '2009_10', '2010_11', '2011_12', '2012_13', '2013_14', '2014_15',
        '2015_16', '2016_17', '2017_18', '2018_19', '2019_20', '2020_21', '2021_22'
    ]

    columns_of_interest = [
                            #Features
                           'ADM_RATE', 'TUITIONFEE_IN', 'IRPS_NRA', 'ADMCON7',
                           'AVGFACSAL', 'PFTFAC',
                           'UGDS', 'TRANS_4', 'INEXPFTE',
                           'OPENADMP', 
                           'BOOKSUPPLY', 'ROOMBOARD_OFF', 'OTHEREXPENSE_OFF',
                           'OTHEREXPENSE_FAM', 'STUFACR', 'IRPS_NRA', 'INSTNM', 'PRGMOFR', 'WDRAW_ORIG_YR2_RT', 
                           
                           #Target Variables
                           'PCT75_EARN_WNE_P10','COUNT_WNE_P10','MD_EARN_WNE_P10','GRAD_DEBT_MDN_SUPP',
                           'Year'
    ]    

    #Creating Empty Dictionary to store data frames in
    dataframes = {}  

    #Iterating through years and concatenating dataframes
    for year in years_list:
        df = pd.read_csv(fr'C:\Users\jakem\CollegeImprovement\COLLEGEIMPROVEMENT\data\data\MERGED{year}_PP.csv')
        df = df[df['HIGHDEG'] >= 3]

        df['Year'] = year

        df = df[columns_of_interest]
        dataframes[year] = df

    df = pd.concat(dataframes.values())

    return df


def data_reader_study():
    """
    This function takes in the field of study data, loops over the years, and returns the concatenated
    dataframes
    """
    years_list = ['1415_1516', '1516_1617', '1617_1718', '1718_1819', '1819_1920']

    columns_of_interest = ['CIPDESC', 'CREDLEV', 'CREDDESC', 
                        'IPEDSCOUNT1', 'IPEDSCOUNT2', 'INSTNM']

    dataframes = {}

    for year in years_list:
        data = fr'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\FieldOfStudyData{year}_PP.csv'
    
        df = pd.read_csv(data, usecols=columns_of_interest)
        dataframes[year] = df

    df2 = pd.concat(dataframes.values())

    return df2

if __name__ == '__main__':
    df = data_reader()

    universities_to_select = ['Philadelphia College of Osteopathic Medicine',
                          'New York Law School',
                          'Rosalind Franklin University of Medicine and Science',
                          'Meharry Medical College',
                          'Medical University of South Carolina',
                          'Thomas Edison State University',
                          'United States Merchant Marine Academy',
                          'University of Texas Southwestern Medical Center',
                          'University of California-San Francisco',
                          'Harvard University',
                          'Massachusetts Institute of Technology',
                          'Princeton University']

    # Selecting the universities
    selected_universities = df[df['INSTNM'].isin(universities_to_select)]
    avg_endowment = selected_universities.groupby('INSTNM')['ENDOWBEGIN'].mean().reset_index()

    # Plotting the boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=avg_endowment, y='ENDOWBEGIN')
    plt.title('Spread of Endowments of Top Universities')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    df.to_csv(r'C:\Users\jakem\CollegeImprovement-1\COLLEGEIMPROVEMENT\data\CollegeImprovementFinalFile2.csv')
    
   

    