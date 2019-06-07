#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.insert(0, '../scripts')
import pipeline
import feature_gen as fg
import pandas as pd


# # ML Analysis
# 
# We start by exporting the data:

# In[2]:


file = '../data/work/block-groups_2012-2016_with-acs_with-gen-features.csv'
df = pipeline.read(file)


# Defining the train/test sets:

# In[3]:


years = [2013, 2014, 2015]
dic_sets = {}
for year in years:
    next_year = year + 1
    dic_sets['train ' + str(year) + ' / test ' + str(next_year)] = (df[df['year']==year], df[df['year']==next_year])


# Defining the feature columns:

# In[4]:


features = ['prev-yr_population_4quantiles_1.0', 'prev-yr_population_4quantiles_2.0', 'prev-yr_population_4quantiles_3.0',             'prev-yr_population_4quantiles_4.0', 'prev-yr_poverty-rate_4quantiles_1.0', 'prev-yr_poverty-rate_4quantiles_2.0',             'prev-yr_poverty-rate_4quantiles_3.0', 'prev-yr_poverty-rate_4quantiles_4.0',             'prev-yr_renter-occupied-households_4quantiles_1.0', 'prev-yr_renter-occupied-households_4quantiles_2.0',             'prev-yr_renter-occupied-households_4quantiles_3.0', 'prev-yr_renter-occupied-households_4quantiles_4.0',             'prev-yr_pct-renter-occupied_4quantiles_1.0', 'prev-yr_pct-renter-occupied_4quantiles_2.0',             'prev-yr_pct-renter-occupied_4quantiles_3.0', 'prev-yr_pct-renter-occupied_4quantiles_4.0',             'prev-yr_median-gross-rent_4quantiles_1.0', 'prev-yr_median-gross-rent_4quantiles_2.0',             'prev-yr_median-gross-rent_4quantiles_3.0', 'prev-yr_median-gross-rent_4quantiles_4.0',             'prev-yr_median-gross-rent_4quantiles_nan', 'prev-yr_median-household-income_4quantiles_1.0',             'prev-yr_median-household-income_4quantiles_2.0', 'prev-yr_median-household-income_4quantiles_3.0',             'prev-yr_median-household-income_4quantiles_4.0', 'prev-yr_median-household-income_4quantiles_nan',             'prev-yr_median-property-value_4quantiles_1.0', 'prev-yr_median-property-value_4quantiles_2.0',             'prev-yr_median-property-value_4quantiles_3.0', 'prev-yr_median-property-value_4quantiles_4.0',             'prev-yr_median-property-value_4quantiles_nan', 'prev-yr_rent-burden_4quantiles_1.0',             'prev-yr_rent-burden_4quantiles_2.0', 'prev-yr_rent-burden_4quantiles_3.0', 'prev-yr_rent-burden_4quantiles_4.0',             'prev-yr_rent-burden_4quantiles_nan', 'prev-yr_pct-white_4quantiles_1.0', 'prev-yr_pct-white_4quantiles_2.0',             'prev-yr_pct-white_4quantiles_3.0', 'prev-yr_pct-white_4quantiles_4.0', 'prev-yr_pct-af-am_4quantiles_1.0',             'prev-yr_pct-af-am_4quantiles_2.0', 'prev-yr_pct-af-am_4quantiles_3.0', 'prev-yr_pct-af-am_4quantiles_4.0',             'prev-yr_pct-hispanic_4quantiles_1.0', 'prev-yr_pct-hispanic_4quantiles_2.0', 'prev-yr_pct-hispanic_4quantiles_3.0',             'prev-yr_pct-hispanic_4quantiles_4.0', 'prev-yr_pct-am-ind_4quantiles_1.0', 'prev-yr_pct-am-ind_4quantiles_2.0',             'prev-yr_pct-am-ind_4quantiles_3.0', 'prev-yr_pct-am-ind_4quantiles_4.0', 'prev-yr_pct-asian_4quantiles_1.0',             'prev-yr_pct-asian_4quantiles_2.0', 'prev-yr_pct-asian_4quantiles_3.0', 'prev-yr_pct-asian_4quantiles_4.0',             'prev-yr_pct-nh-pi_4quantiles_1.0', 'prev-yr_pct-nh-pi_4quantiles_2.0', 'prev-yr_pct-nh-pi_4quantiles_3.0',             'prev-yr_pct-nh-pi_4quantiles_4.0', 'prev-yr_pct-multiple_4quantiles_1.0', 'prev-yr_pct-multiple_4quantiles_2.0',             'prev-yr_pct-multiple_4quantiles_3.0', 'prev-yr_pct-multiple_4quantiles_4.0', 'prev-yr_pct-other_4quantiles_1.0',             'prev-yr_pct-other_4quantiles_2.0', 'prev-yr_pct-other_4quantiles_3.0', 'prev-yr_pct-other_4quantiles_4.0',             'prev-yr_eviction-filings_4quantiles_1.0', 'prev-yr_eviction-filings_4quantiles_2.0',             'prev-yr_eviction-filings_4quantiles_3.0', 'prev-yr_eviction-filings_4quantiles_4.0',             'prev-yr_eviction-filing-rate_4quantiles_1.0', 'prev-yr_eviction-filing-rate_4quantiles_2.0',             'prev-yr_eviction-filing-rate_4quantiles_3.0', 'prev-yr_eviction-filing-rate_4quantiles_4.0',             'prev-yr_evictions_4quantiles_1.0', 'prev-yr_evictions_4quantiles_2.0',             'prev-yr_evictions_4quantiles_3.0', 'prev-yr_evictions_4quantiles_4.0',             'prev-yr_eviction-rate_4quantiles_1.0', 'prev-yr_eviction-rate_4quantiles_2.0',             'prev-yr_eviction-rate_4quantiles_3.0', 'prev-yr_eviction-rate_4quantiles_4.0',             'prev-yr_evictions-effectiveness_4quantiles_1.0', 'prev-yr_evictions-effectiveness_4quantiles_2.0',             'prev-yr_evictions-effectiveness_4quantiles_3.0', 'prev-yr_evictions-effectiveness_4quantiles_4.0',             'total_for_public_assistance_income_4quantiles_1.0',             'total_for_public_assistance_income_4quantiles_2.0',             'total_for_public_assistance_income_4quantiles_3.0',             'total_for_public_assistance_income_4quantiles_4.0',             'with_public_assistance_income_4quantiles_1.0', 'with_public_assistance_income_4quantiles_2.0',             'with_public_assistance_income_4quantiles_3.0', 'with_public_assistance_income_4quantiles_4.0',             'estimate_total_in_labor_force_4quantiles_1.0', 'estimate_total_in_labor_force_4quantiles_2.0',             'estimate_total_in_labor_force_4quantiles_3.0', 'estimate_total_in_labor_force_4quantiles_4.0',             'estimate_civilian_unemployed_4quantiles_1.0', 'estimate_civilian_unemployed_4quantiles_2.0',             'estimate_civilian_unemployed_4quantiles_3.0', 'estimate_civilian_unemployed_4quantiles_4.0',             'total_for_householder_tenure_4quantiles_1.0', 'total_for_householder_tenure_4quantiles_2.0',             'total_for_householder_tenure_4quantiles_3.0', 'total_for_householder_tenure_4quantiles_4.0',             'renter_occupied_4quantiles_1.0', 'renter_occupied_4quantiles_2.0',             'renter_occupied_4quantiles_3.0', 'renter_occupied_4quantiles_4.0',             'renter_moved_2015/2010_later_4quantiles_1.0', 'renter_moved_2015/2010_later_4quantiles_2.0',             'renter_moved_2015/2010_later_4quantiles_3.0', 'renter_moved_2015/2010_later_4quantiles_4.0',             'renter_moved_2010-2014/2000-2009_4quantiles_1.0', 'renter_moved_2010-2014/2000-2009_4quantiles_2.0',             'renter_moved_2010-2014/2000-2009_4quantiles_3.0', 'renter_moved_2010-2014/2000-2009_4quantiles_4.0',             'renter_moved_2000-2009/1990-1999_4quantiles_1.0', 'renter_moved_2000-2009/1990-1999_4quantiles_2.0',             'renter_moved_2000-2009/1990-1999_4quantiles_3.0', 'renter_moved_2000-2009/1990-1999_4quantiles_4.0',             'renter_moved_1990-1999/1980-1989_4quantiles_1.0', 'renter_moved_1990-1999/1980-1989_4quantiles_2.0',             'renter_moved_1990-1999/1980-1989_4quantiles_3.0', 'renter_moved_1990-1999/1980-1989_4quantiles_4.0',             'renter_moved_1980-1989/1970-1979_4quantiles_1.0', 'renter_moved_1980-1989/1970-1979_4quantiles_2.0',             'renter_moved_1980-1989/1970-1979_4quantiles_3.0', 'renter_moved_1980-1989/1970-1979_4quantiles_4.0',             'renter_moved_1979/1969_earlier_4quantiles_1.0', 'renter_moved_1979/1969_earlier_4quantiles_2.0',             'renter_moved_1979/1969_earlier_4quantiles_3.0', 'renter_moved_1979/1969_earlier_4quantiles_4.0']


# Defining the label:

# In[5]:


label = 'upper10th_by_year'


# Setting the parameters for the evaluation table:

# In[6]:


fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
classifiers = pipeline.CLASSIFIERS
parameters = pipeline.PARAMETERS
metric = 'precision_at_0.1'


# Creating the evaluation table:

# In[7]:


evaluation_table = pipeline.evaluation_table(classifiers, parameters, dic_sets, fractions, features, label, metric)


# # Results analysis

# Creating an object with the best model and best metric:

# In[8]:


model, metric = pipeline.model_best_average(evaluation_table, metric)


# Generating parameters for the final outcome of our analysis:

# In[9]:


year_train = 2016
year_predict = 2017
train_X_final = df[df['year']==year_train][features]
train_y_final = df[df['year']==year_train][label]


# Generating the final prediction dataset:

# In[10]:


el_features = ['population_4quantiles_1.0', 'population_4quantiles_2.0', 'population_4quantiles_3.0',             'population_4quantiles_4.0', 'poverty-rate_4quantiles_1.0', 'poverty-rate_4quantiles_2.0',             'poverty-rate_4quantiles_3.0', 'poverty-rate_4quantiles_4.0',             'renter-occupied-households_4quantiles_1.0', 'renter-occupied-households_4quantiles_2.0',             'renter-occupied-households_4quantiles_3.0', 'renter-occupied-households_4quantiles_4.0',             'pct-renter-occupied_4quantiles_1.0', 'pct-renter-occupied_4quantiles_2.0',             'pct-renter-occupied_4quantiles_3.0', 'pct-renter-occupied_4quantiles_4.0',             'median-gross-rent_4quantiles_1.0', 'median-gross-rent_4quantiles_2.0',             'median-gross-rent_4quantiles_3.0', 'median-gross-rent_4quantiles_4.0',             'median-gross-rent_4quantiles_nan', 'median-household-income_4quantiles_1.0',             'median-household-income_4quantiles_2.0', 'median-household-income_4quantiles_3.0',             'median-household-income_4quantiles_4.0', 'median-household-income_4quantiles_nan',             'median-property-value_4quantiles_1.0', 'median-property-value_4quantiles_2.0',             'median-property-value_4quantiles_3.0', 'median-property-value_4quantiles_4.0',             'median-property-value_4quantiles_nan', 'rent-burden_4quantiles_1.0',             'rent-burden_4quantiles_2.0', 'rent-burden_4quantiles_3.0', 'rent-burden_4quantiles_4.0',             'rent-burden_4quantiles_nan', 'pct-white_4quantiles_1.0', 'pct-white_4quantiles_2.0',             'pct-white_4quantiles_3.0', 'pct-white_4quantiles_4.0', 'pct-af-am_4quantiles_1.0',             'pct-af-am_4quantiles_2.0', 'pct-af-am_4quantiles_3.0', 'pct-af-am_4quantiles_4.0',             'pct-hispanic_4quantiles_1.0', 'pct-hispanic_4quantiles_2.0', 'pct-hispanic_4quantiles_3.0',             'pct-hispanic_4quantiles_4.0', 'pct-am-ind_4quantiles_1.0', 'pct-am-ind_4quantiles_2.0',             'pct-am-ind_4quantiles_3.0', 'pct-am-ind_4quantiles_4.0', 'pct-asian_4quantiles_1.0',             'pct-asian_4quantiles_2.0', 'pct-asian_4quantiles_3.0', 'pct-asian_4quantiles_4.0',             'pct-nh-pi_4quantiles_1.0', 'pct-nh-pi_4quantiles_2.0', 'pct-nh-pi_4quantiles_3.0',             'pct-nh-pi_4quantiles_4.0', 'pct-multiple_4quantiles_1.0', 'pct-multiple_4quantiles_2.0',             'pct-multiple_4quantiles_3.0', 'pct-multiple_4quantiles_4.0', 'pct-other_4quantiles_1.0',             'pct-other_4quantiles_2.0', 'pct-other_4quantiles_3.0', 'pct-other_4quantiles_4.0',             'eviction-filings_4quantiles_1.0', 'eviction-filings_4quantiles_2.0',             'eviction-filings_4quantiles_3.0', 'eviction-filings_4quantiles_4.0',             'eviction-filing-rate_4quantiles_1.0', 'eviction-filing-rate_4quantiles_2.0',             'eviction-filing-rate_4quantiles_3.0', 'eviction-filing-rate_4quantiles_4.0',             'evictions_4quantiles_1.0', 'evictions_4quantiles_2.0',             'evictions_4quantiles_3.0', 'evictions_4quantiles_4.0',             'eviction-rate_4quantiles_1.0', 'eviction-rate_4quantiles_2.0',             'eviction-rate_4quantiles_3.0', 'eviction-rate_4quantiles_4.0',             'evictions-effectiveness_4quantiles_1.0', 'evictions-effectiveness_4quantiles_2.0',             'evictions-effectiveness_4quantiles_3.0', 'evictions-effectiveness_4quantiles_4.0']
acs_features = ['total_for_public_assistance_income_4quantiles_1.0',             'total_for_public_assistance_income_4quantiles_2.0',             'total_for_public_assistance_income_4quantiles_3.0',             'total_for_public_assistance_income_4quantiles_4.0',             'with_public_assistance_income_4quantiles_1.0', 'with_public_assistance_income_4quantiles_2.0',             'with_public_assistance_income_4quantiles_3.0', 'with_public_assistance_income_4quantiles_4.0',             'estimate_total_in_labor_force_4quantiles_1.0', 'estimate_total_in_labor_force_4quantiles_2.0',             'estimate_total_in_labor_force_4quantiles_3.0', 'estimate_total_in_labor_force_4quantiles_4.0',             'estimate_civilian_unemployed_4quantiles_1.0', 'estimate_civilian_unemployed_4quantiles_2.0',             'estimate_civilian_unemployed_4quantiles_3.0', 'estimate_civilian_unemployed_4quantiles_4.0',             'total_for_householder_tenure_4quantiles_1.0', 'total_for_householder_tenure_4quantiles_2.0',             'total_for_householder_tenure_4quantiles_3.0', 'total_for_householder_tenure_4quantiles_4.0',             'renter_occupied_4quantiles_1.0', 'renter_occupied_4quantiles_2.0',             'renter_occupied_4quantiles_3.0', 'renter_occupied_4quantiles_4.0',             'renter_moved_2015/2010_later_4quantiles_1.0', 'renter_moved_2015/2010_later_4quantiles_2.0',             'renter_moved_2015/2010_later_4quantiles_3.0', 'renter_moved_2015/2010_later_4quantiles_4.0',             'renter_moved_2010-2014/2000-2009_4quantiles_1.0', 'renter_moved_2010-2014/2000-2009_4quantiles_2.0',             'renter_moved_2010-2014/2000-2009_4quantiles_3.0', 'renter_moved_2010-2014/2000-2009_4quantiles_4.0',             'renter_moved_2000-2009/1990-1999_4quantiles_1.0', 'renter_moved_2000-2009/1990-1999_4quantiles_2.0',             'renter_moved_2000-2009/1990-1999_4quantiles_3.0', 'renter_moved_2000-2009/1990-1999_4quantiles_4.0',             'renter_moved_1990-1999/1980-1989_4quantiles_1.0', 'renter_moved_1990-1999/1980-1989_4quantiles_2.0',             'renter_moved_1990-1999/1980-1989_4quantiles_3.0', 'renter_moved_1990-1999/1980-1989_4quantiles_4.0',             'renter_moved_1980-1989/1970-1979_4quantiles_1.0', 'renter_moved_1980-1989/1970-1979_4quantiles_2.0',             'renter_moved_1980-1989/1970-1979_4quantiles_3.0', 'renter_moved_1980-1989/1970-1979_4quantiles_4.0',             'renter_moved_1979/1969_earlier_4quantiles_1.0', 'renter_moved_1979/1969_earlier_4quantiles_2.0',             'renter_moved_1979/1969_earlier_4quantiles_3.0', 'renter_moved_1979/1969_earlier_4quantiles_4.0']


# In[11]:


acs_2017 = pipeline.read('../data/raw/block-groups_2017_acs-only.csv')
df_2017 = df[df['year']==2016][features + ['GEOID', 'year']]
df_2017['year'] = 2017
for col in el_features:
    df_2017 = df_2017.rename({col: 'prev-yr_' + col})
df_2017 = df_2017.merge(acs_2017, on=('GEOID', 'year'), how='left')


# In[12]:


predict_X_final = df_2017[features]
final_model = model.fit(train_X_final, train_y_final)
predictions = pipeline.get_predictions(final_model, predict_X_final)
df_2017['final_predictions'] = predictions
df_2017.to_csv('../outputs/final_predictions.csv')


# In[13]:


with open("../outputs/final_model.txt", "w") as text_file:
    text_file.write(str(model))

