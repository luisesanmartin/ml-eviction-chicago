#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '../scripts')
import pipeline
import pandas as pd


# # ML Analysis
# 
# We start by exporting the data:

# In[2]:


file = '../data/work/block-groups_2012-2016_with-acs_with-q-features.csv'
df = pipeline.read(file)


# Defining the train/test sets:

# In[3]:


years = [2013, 2014, 2015]
dic_sets = {}
for year in years:
    next_year = year + 1
    dic_sets['train_in_' + str(year)] = (df[df['year']==year], df[df['year']==next_year])


# In[4]:


pipeline.columns_list(df)


# Defining the feature columns:

# In[5]:


features = ['population_4quantiles_1.0', 'population_4quantiles_2.0', 'population_4quantiles_3.0',             'population_4quantiles_4.0', 'poverty-rate_4quantiles_1.0', 'poverty-rate_4quantiles_2.0',             'poverty-rate_4quantiles_3.0', 'poverty-rate_4quantiles_4.0',             'renter-occupied-households_4quantiles_1.0', 'renter-occupied-households_4quantiles_2.0',             'renter-occupied-households_4quantiles_3.0', 'renter-occupied-households_4quantiles_4.0',             'pct-renter-occupied_4quantiles_1.0', 'pct-renter-occupied_4quantiles_2.0',             'pct-renter-occupied_4quantiles_3.0', 'pct-renter-occupied_4quantiles_4.0',             'median-gross-rent_4quantiles_1.0', 'median-gross-rent_4quantiles_2.0',             'median-gross-rent_4quantiles_3.0', 'median-gross-rent_4quantiles_4.0',             'median-gross-rent_4quantiles_1.0', 'median-gross-rent_4quantiles_2.0',             'median-gross-rent_4quantiles_3.0', 'median-gross-rent_4quantiles_4.0',             'median-gross-rent_4quantiles_nan', 'median-household-income_4quantiles_1.0',             'median-household-income_4quantiles_2.0', 'median-household-income_4quantiles_3.0',             'median-household-income_4quantiles_4.0', 'median-household-income_4quantiles_nan',             'median-property-value_4quantiles_1.0', 'median-property-value_4quantiles_2.0',             'median-property-value_4quantiles_3.0', 'median-property-value_4quantiles_4.0',             'median-property-value_4quantiles_nan', 'rent-burden_4quantiles_1.0',             'rent-burden_4quantiles_2.0', 'rent-burden_4quantiles_3.0', 'rent-burden_4quantiles_4.0',             'rent-burden_4quantiles_nan', 'pct-white_4quantiles_1.0', 'pct-white_4quantiles_2.0',             'pct-white_4quantiles_3.0', 'pct-white_4quantiles_4.0', 'pct-af-am_4quantiles_1.0',             'pct-af-am_4quantiles_2.0', 'pct-af-am_4quantiles_3.0', 'pct-af-am_4quantiles_4.0',             'pct-hispanic_4quantiles_1.0', 'pct-hispanic_4quantiles_2.0', 'pct-hispanic_4quantiles_3.0',             'pct-hispanic_4quantiles_4.0', 'pct-am-ind_4quantiles_1.0', 'pct-am-ind_4quantiles_2.0',             'pct-am-ind_4quantiles_3.0', 'pct-am-ind_4quantiles_4.0', 'pct-asian_4quantiles_1.0',             'pct-asian_4quantiles_2.0', 'pct-asian_4quantiles_3.0', 'pct-asian_4quantiles_4.0',             'pct-nh-pi_4quantiles_1.0', 'pct-nh-pi_4quantiles_2.0', 'pct-nh-pi_4quantiles_3.0',             'pct-nh-pi_4quantiles_4.0', 'pct-multiple_4quantiles_1.0', 'pct-multiple_4quantiles_2.0',             'pct-multiple_4quantiles_3.0', 'pct-multiple_4quantiles_4.0', 'pct-other_4quantiles_1.0',             'pct-other_4quantiles_2.0', 'pct-other_4quantiles_3.0', 'pct-other_4quantiles_4.0',             'total_for_public_assistance_income_4quantiles_1.0',             'total_for_public_assistance_income_4quantiles_2.0',             'total_for_public_assistance_income_4quantiles_3.0',             'total_for_public_assistance_income_4quantiles_4.0',             'with_public_assistance_income_4quantiles_1.0', 'with_public_assistance_income_4quantiles_2.0',             'with_public_assistance_income_4quantiles_3.0', 'with_public_assistance_income_4quantiles_4.0',             'estimate_total_in_labor_force_4quantiles_1.0', 'estimate_total_in_labor_force_4quantiles_2.0',             'estimate_total_in_labor_force_4quantiles_3.0', 'estimate_total_in_labor_force_4quantiles_4.0',             'estimate_civilian_unemployed_4quantiles_1.0', 'estimate_civilian_unemployed_4quantiles_2.0',             'estimate_civilian_unemployed_4quantiles_3.0', 'estimate_civilian_unemployed_4quantiles_4.0',             'total_for_householder_tenure_4quantiles_1.0', 'total_for_householder_tenure_4quantiles_2.0',             'total_for_householder_tenure_4quantiles_3.0', 'total_for_householder_tenure_4quantiles_4.0',             'renter_occupied_4quantiles_1.0', 'renter_occupied_4quantiles_2.0',             'renter_occupied_4quantiles_3.0', 'renter_occupied_4quantiles_4.0',             'renter_moved_2015/2010_later_4quantiles_1.0', 'renter_moved_2015/2010_later_4quantiles_2.0',             'renter_moved_2015/2010_later_4quantiles_3.0', 'renter_moved_2015/2010_later_4quantiles_4.0',             'renter_moved_2010-2014/2000-2009_4quantiles_1.0', 'renter_moved_2010-2014/2000-2009_4quantiles_2.0',             'renter_moved_2010-2014/2000-2009_4quantiles_3.0', 'renter_moved_2010-2014/2000-2009_4quantiles_4.0',             'renter_moved_2000-2009/1990-1999_4quantiles_1.0', 'renter_moved_2000-2009/1990-1999_4quantiles_2.0',             'renter_moved_2000-2009/1990-1999_4quantiles_3.0', 'renter_moved_2000-2009/1990-1999_4quantiles_4.0',             'renter_moved_1990-1999/1980-1989_4quantiles_1.0', 'renter_moved_1990-1999/1980-1989_4quantiles_2.0',             'renter_moved_1990-1999/1980-1989_4quantiles_3.0', 'renter_moved_1990-1999/1980-1989_4quantiles_4.0',             'renter_moved_1980-1989/1970-1979_4quantiles_1.0', 'renter_moved_1980-1989/1970-1979_4quantiles_2.0',             'renter_moved_1980-1989/1970-1979_4quantiles_3.0', 'renter_moved_1980-1989/1970-1979_4quantiles_4.0',             'renter_moved_1979/1969_earlier_4quantiles_1.0', 'renter_moved_1979/1969_earlier_4quantiles_2.0',             'renter_moved_1979/1969_earlier_4quantiles_3.0', 'renter_moved_1979/1969_earlier_4quantiles_4.0']


# Defining the label:

# In[10]:


label = 'upper10th_by_year'


# Setting the parameters for the evaluation table:

# In[7]:


fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
classifiers = pipeline.CLASSIFIERS
parameters = pipeline.PARAMETERS_MID


# Creating the evaluation table:

# In[11]:


evaluation_table = pipeline.evaluation_table(classifiers, parameters, dic_sets, fractions, features, label)


# Exporting the evaluation table:

# In[ ]:


evaluation_table.to_csv('../outputs/evaluation_table.csv')

