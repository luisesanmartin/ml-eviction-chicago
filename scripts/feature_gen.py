import pandas as pd
import numpy as np

features = ['poverty-rate', 'median-gross-rent', 'median-household-income', 'median-property-value]
years = [2012, 2013, 2014, 2015, 2016]

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