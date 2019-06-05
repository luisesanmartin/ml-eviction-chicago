'''
FEATURE GENERATION
Functions to generate new features from raw data attributes
'''


import os
import sys
from sys import version_info
import pandas as pd
import numpy as np


FEATURES = ['poverty-rate', 'median-gross-rent', 'median-household-income', \
            'median-property-value', 'rent-burden']
YEARS_LIST = [2012, 2013, 2014, 2015, 2016, (2012, 2015)]
Q = 5

'''Dataframe new feature names format
attribute_year_#quantiles_categorical
attribute_year_#quantiles_#q
attribute_#quantiles_categorical
attribute_#quantiles_#q
'''


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
    **EXPECTS y OR n FOR USER RESPONSE.** (TO DO: expand allowable responses?)

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

    if response.lower() not in ['y', 'n']:
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
    print("Linux: Enter Ctrl+D when you are finished")
    print("Windows: Hit Enter, Ctrl+Z, and Enter again when you are finished")

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
Create a list of raw attributes to cut, and this section will loop over each to generate features.
First adds a feature that cuts columns by quantiles (4 for quartiles, 5 for quantiles, etc.).
Then further adds dummies/binary variables for each quantile category generated.
'''
def generate_quantile_features(df, attributes_list, q, years_list=None):
    '''
    If want to create quantiles varied by the specified years in years_list, will first extract those 
    years and determine value quantiles only within those years/year ranges.
    Then (or otherwise), creates categorical and dummy variables from quantiles over whole dataset
    
    INPUT:
    - dataframe to cut (df), attributes to cut (list), number of quantiles (q=int)
    - years_list is a list of values or tuples indicating range of years (inclusive) by which to
    calculate quantiles (e.g. years_list = [2013, (2014, 2016), 2017])
    
    OUTPUT:
    - dataframe with a feature for quantiles (as categories) and dummy features for those categories;
    quantiles are calculated for each raw attribute in the attributes_list
    '''
    years_list_usage = 'years_list must be a list of values or tuples indicating range of years \
        (inclusive) by which to calculate quantiles (e.g. years_list = [2013, (2014, 2016), 2017])'
    if years_list:
        assert isinstance(years_list, list), years_list_usage
        df = extract_quantiles_by_year(df, attributes_list, q, years_list)
    df = generate_quantiles_and_dummies(df, attributes_list, q)
    return df


def extract_quantiles_by_year(df, attributes_list, q, years_list):
    '''
    Extract quantiles determined from each year/year-range listed in years_list
    This function allows for time-varying feature generation

    INPUT:
    - dataframe to cut (df), attributes to cut (list), number of quantiles (q=int)
    - years_list is a list of values (int) or tuples indicating range of years (inclusive) by which to
    calculate quantiles (e.g. years_list = [2013, (2014, 2016), 2017])
    
    OUTPUT:
    - dataframe with a feature for quantiles (as categories) and dummy features for those categories;
    quantiles are calculated from each year/year-range and for each raw attribute in the attributes_list
    '''
    for yr in years_list:
        # create a df with only the given yr or range of years and only with the columns in attributes_list
        # (do this for memory, i.e. so don't have to grab all/irrelevant columns each time)
        if isinstance(yr, tuple):
            only_yr_df = df[df['year'].isin(yr)][attributes_list]
            # pass on a suffix based on the year/year-range variable to append to new feature col names
            colname_suffix = '{}-{}'.format(yr[0], yr[1])
        elif isinstance(yr, int):
            only_yr_df = df[df['year'] == yr][attributes_list]
            colname_suffix = yr
        else:
            raise TypeError('values for years_list must be int type or tuples')
        
        only_yr_df_w_quants = generate_quantiles_and_dummies(only_yr_df, attributes_list, q, colname_suffix)
        # drop cols in attributes_list to avoid duplicates when merged again with original dataframe
        only_yr_df_w_quants.drop(columns=attributes_list, inplace=True)
        # merge with main df; code below will match on index and fill NaNs for non-yr rows
        df = df.merge(only_yr_df_w_quants, how='outer', left_index=True, right_index=True)
    return df


def generate_quantiles_and_dummies(df, attributes_list, q, year_var=None):
    '''
    Loop over attributes list, and create new quantile feature and quantile dummies from each
    Paramater year_var is only used if this function is called by extract_quantiles_by_year; it is used to
        add a suffix to new features calculated on time-specific quantiles (time-varying feature generation)
    
    INPUT: dataframe (pandas df) | attributes_list: features to loop over (list of str) | q: (int)
    year_var (str) should be the new feature suffix to indicate that df passed was for yr-specific quantiles
    OUTPUT: dataframe with generated features (pandas df)
    '''
    quantile_features = []
    for attribute in attributes_list:
        # create_categorical_quantiles modifies df, but returns the colname of new feature
        quant_feat_colname = create_categorical_quantiles(df, q, attribute, year_var)
        quantile_features.append(quant_feat_colname)

    # When creating dummies, original column gets dropped
    # So instead copy feature and use original for creating dummies
    for quant_feat in quantile_features:
        feature_copy = quant_feat + '_categorical'
        df[feature_copy] = df[quant_feat]
    # Use original feature for dummies (since dummies will be named based on original)
    df = create_dummies(df, quantile_features)
    return df


def create_categorical_quantiles(df, q, colname, suffix=None):
    '''
    Create a new categorical feature that labels rows based on membership in a given quantile.
    
    INPUT:  dataframe (df) | q (int): number of quantiles desired (e.g. 5 for quintiles),
            column (str): name of df column to discretize by quantiles
            suffix (str): suffix to append to col name, indicating some variation of the feature
    OUTPUT: modifies dataframe in place, and returns the new column name (str)
    '''
    if suffix:
        new_colname = "{}_{}_{}quantiles".format(colname, suffix, q)
    else:
        new_colname = "{}_{}quantiles".format(colname, q)

    labels = [x for x in range(1, q + 1)]
    try:
        df[new_colname] = pd.qcut(df[colname], q, labels=labels)
    except:
        # dealing with non-unique bin edges, but not dropping the duplicate bins
        # from: https://stackoverflow.com/questions/20158597/how-to-qcut-with-non-unique-bin-edges
        df[new_colname] = pd.qcut(df[colname].rank(method='first'), q, labels=labels)
    return new_colname


######################
# Z-SCORE GENERATION #
######################
'''
This section creates new features that assign z-scores to each row, given an attribute.
Create a list of data features that you want to scale, and loop over each to generate features.

Z-score = row value - mean / standard deviation
'''


#######################
# NORMALIZED FEATURES #
#######################
'''
This section creates new features that assign normalized scores to each row, given an attribute.
Create a list of data features that you want to scale, and loop over each to generate features.

Normalization = row value - min / (max-min)
'''



#################################
# PREVIOUS YEAR'S EVICTION RATE #
#################################
'''
This section creates a new feature that adds the previous year's eviction rate for that block group.
'''

def create_prev_yr_feature(df, feature):
    '''

    OUTPUT: df with added feature
    '''
    years = df.year.value_counts().index.tolist()
    years.sort()

    col = 'prev-yr_{}'.format(feature)
    mid_df = pd.DataFrame(columns=('GEOID', 'year', col))

    for yr in years[1:]:
        yr_df = df[df['year'] == yr][['year', 'GEOID']]
        prev_yr_df = df[df['year'] == yr - 1][['GEOID', feature]]
        yr_df = yr_df.merge(prev_yr_df) # Auto-merges on common col, GEOID
        yr_df.rename(columns={feature: col}, inplace=True)
        mid_df = mid_df.append(yr_df, ignore_index=True)

    mid_df = mid_df.astype(dtype={'GEOID':'int64', 'year':'int64'}) # b/c dataframe autotyped these as objects
    df = df.merge(mid_df, how='left')
    return df


####################
# RUN FROM IPYTHON #
####################
def run(csv_file, prompts=False):
    '''
    Call this function from within iPython to generate new df.
    INPUT: path to csv file (str), whether to show prompts (bool)
    '''
    # TO DO: specify coltypes function (sys) does not work yet in ipython
    # TO DO: so passing prompts=True will end early
    df = initial_read(csv_file, prompts)
    if df is not None:
        # UNIQUE SET OF COMMANDS SPECIFIC TO PROJECT
        df = generate_quantile_features(df, FEATURES, Q, YEARS_LIST)
    return df


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
             "Type false if want to automatically download and process data without prompts.\n" \
             "**For now, edit paramaters inside script (e.g. year variation, attributes to cut."

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
        df = generate_quantile_features(df, FEATURES, Q, YEARS_LIST)
        #return df?
        #write out to CSV?
        print(df.dtypes)

if __name__ == "__main__":
    go()