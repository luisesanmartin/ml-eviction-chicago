'''
FEATURE GENERATION
Functions to generate new features from raw data attributes
'''

#### Stil a lot of to-dos / not done

import os
import sys
from sys import version_info
import pandas as pd
import numpy as np


#############################
# READ AND PRE-PROCESS DATA #
#############################

def initial_read(csv_file, prompts=False):
    '''
    Read in first 10 rows of file to extract auto-datatypes,
    Allow user to specify column data types if desired.

    INPUT: csv_file (str) -- path to the csv file to open
        prompts (bool) -- if user wants to see prompts to specify datatypes

    OUTPUT: Prints column names with their automated datatypes
            Asks if user wants to specify datatypes of columns
            Returns None or a dataframe
    '''

    if not os.path.exists(csv_file):
        print("Cannot find file or file does not exist")
        return None

    col_types = {}
    if prompts:
        response = user_input("Would you like to see column names and auto-datatypes from this data (y/n)?")
        if response == 'y':
            df = pd.read_csv(csv_file, nrows=10)
            print(df.dtypes)

        response = user_input("Would you like to change some or all of the column datatypes (y/n)?")
        # dtype will not error if dict contains colnames that  do not exist in data; no need to check
        if response == 'y':
            col_types = specify_coltypes()

    df = read_process_data(csv_file, col_types)
    return df


def user_input(prompt):
    '''
    Code for prompting user input at command line. 
    **EXPECTS y OR n FOR USER RESPONSE.**

    INPUT: prompt/question (str)
    OUTPUT: user response (str)
    '''
    # Code for version check by Chris Simpkins from:
    # http://sweetme.at/2014/01/22/how-to-get-user-input-from-the-command-line-in-a-python-script/
    py3 = version_info[0] > 2 #creates boolean value for test that Python major version > 2
    if py3:
        response = input(prompt)
    else:
        response = raw_input(prompt)

    if response not in ['y', 'n']:
        print("Not a valid response. Please enter y or n.")
        response = user_input(prompt)
    return response


def specify_coltypes():
    '''
    Have user specify the datatypes of passed atrributes/columns
    Function call prints prompts for user to enter colnames and coltypes (str)

    OUTPUT: mapping of attribute names to datatypes (dict)
            e.g. col_types = {'age': 'int', 'name': 'str'}
    '''
    print("Please specify which attributes you would like to change using the following format:")
    print("<colname1> <coltype1>\n<colname2> <coltype2>")
    print("Enter Ctrl+D when you are finished")

    # inspired by https://stackoverflow.com/questions/14147369/make-a-dictionary-in-python-from-input-values
    col_types = dict(col_map.split() for col_map in sys.stdin.read().splitlines())
    return col_types


def read_process_data(csv_file, col_types=None):
    '''
    Purpose: Read in and process csv

    Inputs: csv_file (string) -- path to the csv file to open
            col_types (dict) -- mapping of attribute names to datatypes

    Returns: (dataframe) -- a pre-processed dataframe
    '''
    if not col_types:
        df = pd.read_csv(csv_file)
    else:
        df = pd.read_csv(csv_file, dtype=col_types)

    fill_dict = {}
    # Select columns whose datatypes are numerical (e.g. int, float, etc.)
    numeric_cols_df = df.select_dtypes(include=['number'])
    for attribute in numeric_cols_df.columns:
        fill_dict[attribute] = numeric_cols_df[attribute].median()

    df.fillna(fill_dict, inplace=True)
    return df


##################################
# GENERATE FEATURES & PREDICTORS #
##################################

def create_dummies(df, col_list):
    '''
    Transform cols to binary to prepare for modeling

    INPUT: df, columns to transform (list of str)

    OUTPUTS: returns dataframe with dummy variables
    '''
    assert isinstance(col_list, list), "2nd argument must be a list of column names (as str) to transform"
    rv = pd.get_dummies(df, dummy_na=True, columns=col_list) # drops original columns
    return rv


########################
# QUANTILES GENERATION #
########################
'''
This section creates new features for desired quantiles of existing data features.

Create a list of data features that you want to cut, and loop over each to generate features.
First adds a feature that cuts columns by quantiles (4 for quartiles, 5 for quantiles, etc.).
Then further adds dummies/binary variables for each quantile.
'''
q = 5 # quartiles: q=4, quintiles: q=5

def generate_quantile_year_var(df, features_list, q, year_vars=None):
    '''
    year_vars: (list) year values by which to vary quantile categories
    '''

    if year_vars is None:
        generate_quantile_dummies(df, features_list, q)
    else:
        assert isinstance(year_vars, list), "year_var must be None or a list of year values"
        extract_years(df, year_vars, features_list)


def extract_years(df, years_list, features_list):
    '''
    Extract list of years from dataframe for time-varying feature generation
    '''
    #df[df['year'].isin([2013])]
    for yr in years_list:
        # create a df with only the given yr and only with the columns in col_list
        only_yr_df = df[df['year'] == yr][features_list]
        only_yr_df_w_quants = generate_quantile_dummies(only_yr_df, features_list, q)
        # have to drop so that won't get duplicates when merged again with original datafarme
        only_yr_df_w_quants.drop(columns=features_list, inplace=True)

        # to do: still need to generate quantiles and dummies on these time vars
        # to do: then need to somehow append columns to overall df
        # to do: note that before appending, columns in features_list will have duplicates unles dropped


def generate_quantile_dummies(df, features_list, q):
    '''
    Loop over features list, and create new quantile feature and quantile dummies from each.
    
    INPUT: dataframe (pandas df), features_list: features to loop over (list), q: (int)
    OUTPUT: dataframe with generated features (pandas df)
    '''
    quantile_features = []
    for feature in features_list:
        quant_feat = create_quantiles_feature(df, q, feature)
        quantile_features.append(quant_feat)

    # When creating dummies, original column gets dropped
    # So instead copy feature and use original for creating dummies
    for quant_feat in quantile_features:
        feature_copy = quant_feat + '_categorical'
        df[feature_copy] = df[quant_feat]

    # Use original feature for dummies (since dummies will be named based on original)
    df = create_dummies(df, quantile_features)
    return df


def create_quantiles_feature(df, q, column):
    '''
    Create a new categorical feature that labels rows based on membership in a given quantile.
    
    INPUT:  dataframe (df), q (int): number of quantiles desired (e.g. 5 for quintiles),
            column (str): name of df column to discretize by quantiles
    OUTPUT: modifies dataframe in place, and returns the new column name (str)
    '''
    new_colname = "{}_{}quantiles".format(column, q)
    labels = [x for x in range(1, q + 1)]
    try:
        df[new_colname] = pd.qcut(df[column], q, labels=labels)
    except:
        # dealing with non-unique bin edges, but not dropping the duplicate bins
        # from: https://stackoverflow.com/questions/20158597/how-to-qcut-with-non-unique-bin-edges
        df[new_colname] = pd.qcut(df[column].rank(method='first'), q, labels=labels)
    return new_colname


######################
# Z-SCORE GENERATION #
######################
'''
This section creates new features for desired quantiles of existing data features.

Create a list of data features that you want to cut, and loop over each to generate features.
First adds a feature that cuts columns by quantiles (4 for quartiles, 5 for quantiles, etc.).
Then further adds dummies/binary variables for each quantile.
'''







#########################
# RUN FROM COMMAND LINE #
#########################

def go():
    '''
    Call script from command line
    '''
    usage = "Usage: python feature_gen.py <data filename> <prompts>\n" \
             "<data filename> is csv filename (str) of data to be processed.\n" \
             "<prompts>: Type true if you want to see column names and/or specify datatypes.\n" \
             "Type false if you want to automatically download and process data without prompts."

    num_args = len(sys.argv)
    args = sys.argv

    if num_args < 3:
        print(usage)
        sys.exit(1)
    elif args[2].lower() not in ['true', 'false']:
        print(usage)
        sys.exit(1)
    elif args[2].lower() == 'true':
        df = initial_read(args[1], prompts=True)
    else:
        df = initial_read(args[1], prompts=False)

    if df is not None:
        # UNIQUE SET OF COMMANDS SPECIFIC TO PROJECT
        features = ['poverty-rate', 'median-gross-rent', 'median-household-income', 'median-property-value]
        years = [2012, 2013, 2014, 2015, 2016]
        
        










###############
# LUISE'S CODE#
###############
def create_time_label(df, date_posted, date_funded):
    '''
    '''

    days60 = pd.DateOffset(days=60)

    df['funded'] = np.where(df[date_funded] <= df[date_posted] + days60, 1, 0)


def time_based_split(df, time_col, date_threshold, months_range):
    '''
    '''

    date_lower_threshold = pd.to_datetime(date_threshold)
    date_upper_threshold = date_lower_threshold + \
                           pd.DateOffset(months=months_range)
    df_train = df[df[time_col]<=date_lower_threshold]
    df_test = df[(df[time_col]>date_lower_threshold) \
              & (df[time_col]<=date_upper_threshold)]

    print('train/test threshold:', date_lower_threshold)
    print('test upper threshold:', date_upper_threshold)

    return df_train, df_test


def to_date(df, column):
    '''
    '''

    df[column] = pd.to_datetime(df[column], infer_datetime_format=True)


def discrete_0_1(df, column, value0, value1):
    '''
    '''

    df[column] = df[column].replace(value0, 0)
    df[column] = df[column].replace(value1, 1)
    df[column] = pd.to_numeric(df[column])


def fill_nas_other(df, column, label):
    '''
    '''

    df[column] = df[column].fillna(value=label)


def fill_nas_mode(df, column):
    '''
    '''

    mode = df[column].mode().iloc[0]
    df[column] = df[column].fillna(value=mode)


def fill_nas_median(df, column):
    '''
    Replaces the NaN values of a column (column) in a dataframe (df) with
    the value of the column median

    Inputs:
        - column (column of a pandas dataframe): the column whose NaN
        values we want to fill in with the median. It should be a variable
        included in df.
        - df: the pandas dataframe where column is and where we'll replace
        the NaN values.

    Output: nothing. Modifies the df directly.
    '''

    median = df[column].quantile()
    df[column] = df[column].fillna(value=median)


def discretize(df, column):
    '''
    Creates in the dataframe provided dummy variables indicating that an
    observation belongs to a certain quartile of the column provided.
    Each dummy has the name of the column + a number indicating the quartile.

    Inputs:
        - column (column of a pandas dataframe): the column we want to
        discretize. It should be a continuous variable included in df.
        - df: the pandas dataframe where column is and where we'll add
        the new dummy variables.

    Output: nothing. Modifies the df directly.
    '''
    N_SUBSETS = 4
    WIDE = 1 / N_SUBSETS
    
    xtile = 0
    col = df[column]

    for i in range(1, N_SUBSETS + 1):

        mini = col.quantile(xtile)
        maxi = col.quantile(xtile + WIDE)
        df.loc[(df[column] >= mini) & (df[column] <= maxi), \
               column + '_quartile'] = i
        xtile += WIDE


def create_dummies(df, column):
    '''
    Takes a dataframe (df) and a categorical variable in it (column) and
    creates a dummy for each distinct value of the input categorical
    variable.

    Inputs:
        - column (column of a pandas dataframe): the column we want to
        discretize. It should be a categorical variable included in df.
        - df: the pandas dataframe where column is and where we'll add
        the new dummy variables
    Output: nothing. Modifies the df directly.       
    '''

    for value in df[column].unique():

        df.loc[df[column] == value, column + '_' + str(value)] = 1
        df.loc[df[column] != value, column + '_' + str(value)] = 0


def replace_over_one(df, column):
    '''
    Takes a dataframe (df) and a variable in it (column) and replaces
    the values over one with ones.

    Inputs:
        - column (column of a pandas dataframe): the column whose values
        over one we will replace with ones.
        - df: the pandas dataframe where column is.
    Output: nothing. Modifies the df directly.
    '''

    df.loc[df[column] > 1, column] = 1


def discretize_over_zero(df, column):
    '''
    Takes a dataframe (df) and a variable in it (column) and creates a
    dummy indicating the observations that have a value higher than
    zero.

    Inputs:
        - column (column of a pandas dataframe): the column whose values
        we'll take to create the dummy.
        - df: the pandas dataframe where column is.
    Output: nothing. Modifies the df directly.
    '''

    df.loc[df[column] == 0, column + '_over_zero'] = 0
    df.loc[df[column] > 0, column + '_over_zero'] = 1