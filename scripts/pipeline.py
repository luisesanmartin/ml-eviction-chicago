import time
import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.tree as tree
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import ParameterGrid


pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=Warning)

# Our list of classifiers
CLASSIFIERS = {'Gradient boosting': GradientBoostingClassifier,
               'Ada boosting': AdaBoostClassifier,
               'Bagging': BaggingClassifier,
               'Random forest': RandomForestClassifier,
               'Linear SVC': LinearSVC,
               'SVC': SVC,
               'Logistic regression': LogisticRegression,
               'Decision tree': DecisionTreeClassifier,
               'Nearest neighbors': KNeighborsClassifier}

# Small list of classifiers -- used to check if the code was running properly
CLASSIFIERS_small = {'Logistic regression': LogisticRegression,
               'Decision tree': DecisionTreeClassifier}

# Small list of parameters -- used to check if the code was running properly
PARAMETERS_small = \
{'Gradient boosting': {'subsample': [0.5, 0.2],
                       'max_depth': [1, 5],
                       'n_estimators': [50, 100],
                       'max_features': [0.2, 1/3],
                       'learning_rate': [0.1, 0.25],
                       'random_state': [0]},
 'Ada boosting': {'base_estimator': [DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=0)],
                  'n_estimators': [30, 50],
                  'learning_rate': [0.5, 0.1],
                  'random_state': [0]},
 'Bagging': {'base_estimator': [SVC(C=1.0, kernel='linear', random_state=0),
                             DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=0),
                             LogisticRegression(C=1.0, penalty='l1', random_state=0)],
             'n_estimators': [10, 100],
             'max_samples': [1/3],
             'max_features': [1/5, 1/3],
             'n_jobs': [10],
             'random_state': [0]},
 'Random forest': {'n_estimators': [100, 1000, 10000],
                   'criterion': ['gini', 'entropy'],
                   'max_features': [0.1, 0.2, 1/3, 1/2],
                   'n_jobs': [10],
                   'random_state': [0]},
 'SVC': {'C': [0.001, 0.01, 0.1, 1, 10],
         'kernel': ['linear', 'rbf'],
         'random_state': [0]},
 'Linear SVC': {'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'dual': [False],
                'random_state': [0]},
 'Logistic regression': {'C': [0.001, 0.01, 0.1, 1, 10],
                         'penalty': ['l1', 'l2'],
                         'random_state': [0]},
 'Decision tree': {'max_depth': [1, 5, 10, 20, 50],
                   'criterion': ['gini', 'entropy'],
                   'min_samples_split': [5, 10, 50, 100],
                   'random_state': [0]},
 'Nearest neighbors': {'n_neighbors': [1],
                       'n_jobs': [10]}}

# Our final list of parameters
PARAMETERS = \
{'Gradient boosting': {'subsample': [1.0, 0.5, 0.2],
                       'max_depth': [1, 5, 10, 20, 50],
                       'n_estimators': [50, 100, 200],
                       'max_features': [0.1, 0.2, 1/3, 1/2],
                       'learning_rate': [0.1, 0.25, 0.5],
                       'random_state': [0]},
 'Ada boosting': {'base_estimator': [DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=0),
                                  DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=0),
                                  LogisticRegression(C=1.0, penalty='l1', random_state=0),
                                  LogisticRegression(C=0.1, penalty='l1', random_state=0),
                                  LogisticRegression(C=1.0, penalty='l2', random_state=0),
                                  LogisticRegression(C=0.1, penalty='l2', random_state=0)],
                  'n_estimators': [30, 50, 100],
                  'learning_rate': [1.0, 0.5, 0.1],
                  'random_state': [0]},
 'Bagging': {'base_estimator': [SVC(C=1.0, kernel='linear', random_state=0),
                             SVC(C=0.1, kernel='linear', random_state=0),
                             SVC(C=1.0, kernel='rbf', random_state=0),
                             SVC(C=0.1, kernel='rbf', random_state=0),
                             DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=0),
                             DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=0),
                             LogisticRegression(C=1.0, penalty='l1', random_state=0),
                             LogisticRegression(C=0.1, penalty='l1', random_state=0),
                             LogisticRegression(C=1.0, penalty='l2', random_state=0),
                             LogisticRegression(C=0.1, penalty='l2', random_state=0), 
                             ],
             'n_estimators': [10, 100],
             'max_samples': [1/3, 1/2],
             'max_features': [1/5, 1/3],
             'n_jobs': [10],
             'random_state': [0]},
 'Random forest': {'n_estimators': [100, 1000, 10000],
                   'max_depth': [1, 5, 10, 20, 50],
                   'criterion': ['gini', 'entropy'],
                   'max_features': [0.1, 0.2, 1/3, 1/2],
                   'n_jobs': [10],
                   'random_state': [0]},
 'SVC': {'C': [0.001, 0.01, 0.1, 1, 10],
         'kernel': ['linear', 'rbf'],
         'random_state': [0]},
 'Linear SVC': {'C': [0.001, 0.01, 0.1, 1, 10],
                            'penalty': ['l1', 'l2'],
                            'dual': [False],
                            'random_state': [0]},
 'Logistic regression': {'C': [0.001, 0.01, 0.1, 1, 10],
                         'penalty': ['l1', 'l2'],
                         'random_state': [0]},
 'Decision tree': {'max_depth': [1, 5, 10, 20, 50],
                   'criterion': ['gini', 'entropy'],
                   'min_samples_split': [5, 10, 50, 100],
                   'random_state': [0]},
 'Nearest neighbors': {'n_neighbors': [1, 3],
                       'n_jobs': [10]}}


def read(csv_file):
    '''
    Reads the data of a csv file and returns a Pandas dataframe of it.

    Inputs:
        - csv_file: a csv file containing the dataframe
    '''

    return pd.read_csv(csv_file)

def get_predictions(classifier, X_test):
    '''
    Returns a Pandas Series with the prediction scores.

    Inputs:
        - classifier object
        - X_test: test dataset (Pandas)
    Output: Array with the prediction scores
    '''

    if hasattr(classifier, 'predict_proba'):
        pred_scores = classifier.predict_proba(X_test)[:,1]
    else:
        pred_scores = classifier.decision_function(X_test)

    return pred_scores

def simple_classifier(y_test):
    '''
    Returns a float number with the accuracy if we just predicted every
    value in the test set to be 1/0, whatever fraction is higher in y_test.

    Inputs:
        y_test: Pandas series with the test label
    Output: accuracy of this simple classifier method
    '''

    mean = y_test.mean()

    print("Predicting every data point's value to be 1, " + \
          "the accuracy is", round(mean*100, 1), "%")

    return mean

def accuracy(classifier, threshold, X_test, y_test):
    '''
    Returns the accuracy (float) of a classifier given a certain threshold,
    and a certain test set (X_test and y_test).

    Inputs:
        - classifier: the model we are using
        - threshold: a fraction that denotes the upper percent of the
                     population that will have positively predicted labels
        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: accuracy (float)
    '''

    df = pd.DataFrame()
    df['pred_scores'] = get_predictions(classifier, X_test)
    df['y_test'] = y_test.reset_index(drop=True)
    df.sort_values(by=['pred_scores'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['pred_label'] = np.where(df.index + 1 <= \
                                threshold * len(y_test), 1, 0)
    acc = accuracy_score(df['pred_label'], df['y_test'])

    return acc

def precision(classifier, threshold, X_test, y_test):
    '''
    Returns the precision (float) of a classifier given a certain
    threshold, and a certain test set (X_test and y_test).

    Inputs:
        - classifier: the model we are using
        - threshold: a fraction that denotes the upper percent of the
                     population that will have positively predicted labels
        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: precision (float)
    '''

    df = pd.DataFrame()
    df['pred_scores'] = get_predictions(classifier, X_test)
    df['y_test'] = y_test.reset_index(drop=True)
    df.sort_values(by=['pred_scores'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['pred_label'] = np.where(df.index + 1 <= \
                                threshold * len(y_test), 1, 0)
    c = confusion_matrix(df['y_test'], df['pred_label'])
    true_negatives, false_positive, false_negatives, true_positives = c.ravel()
    prec = true_positives / (false_positive + true_positives)

    return prec

def recall(classifier, threshold, X_test, y_test):
    '''
    Returns the recall (float) of a classifier given a certain
    threshold, and a certain test set (X_test and y_test).

    Inputs:
        - classifier: the model we are using
        - threshold: a fraction that denotes the upper percent of the
                     population that will have positively predicted labels
        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: recall (float)
    '''

    df = pd.DataFrame()
    df['pred_scores'] = get_predictions(classifier, X_test)
    df['y_test'] = y_test.reset_index(drop=True)
    df.sort_values(by=['pred_scores'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['pred_label'] = np.where(df.index + 1 <= \
                                threshold * len(y_test), 1, 0)
    c = confusion_matrix(df['y_test'], df['pred_label'])
    true_negatives, false_positive, false_negatives, true_positives = c.ravel()
    rec = true_positives / (false_negatives + true_positives)

    return rec

def f1(classifier, threshold, X_test, y_test):
    '''
    Returns the f1 score (float) of a classifier given a certain
    threshold, and a certain test set (X_test and y_test).

    Inputs:
        - classifier: the model we are using
        - threshold: a fraction that denotes the upper percent of the
                     population that will have positively predicted labels
        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: f1 score (float)
    '''

    df = pd.DataFrame()
    df['pred_scores'] = get_predictions(classifier, X_test)
    df['y_test'] = y_test.reset_index(drop=True)
    df.sort_values(by=['pred_scores'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['pred_label'] = np.where(df.index + 1 <= \
                                threshold * len(y_test), 1, 0)
    score = f1_score(df['y_test'], df['pred_label'])

    return score

def area_under_curve(classifier, X_test, y_test):
    '''
    Returns the area under the curve (float) of a classifier
    given a certain test set (X_test and y_test).

    Inputs:
        - classifier: the model we are using
        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: area under the curve (float)
    '''

    pred_scores = get_predictions(classifier, X_test)
    fpr, tpr, _ = roc_curve(y_test, pred_scores, pos_label=1)
    area = auc(fpr, tpr)

    return area

def precision_recall_curves(classifier, X_test, y_test):
    '''
    (This function uses code borrowed from the lab 4)

    Plots the precision and recall curves of a classifier, given a
    certain test set.

    Inputs:
        - classifier: the model we are using
        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: plot object
    '''

    pred_scores = get_predictions(classifier, X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, \
                                    pred_scores,pos_label=1)
    population = [1.*sum(pred_scores>=threshold)/len(pred_scores) \
                 for threshold in thresholds]
    p, = plt.plot(population, precision[:-1], color='b')
    r, = plt.plot(population, recall[:-1], color='r')
    plt.legend([p, r], ['precision', 'recall'])
    
    return plt

def evaluation_table(classifiers, parameters, datasets, fractions, features, label, preferred_metric):
    '''
    (Please notice that this function might take a while to run)

    Returns a dataframe where each row is a classifier from classifiers
    and each column is a model performance indicator. Each classifier
    is evaluated on the same features and the same label.

    It also exports that dataframe as a csv file with the name 
    'evaluation_table.csv' and generates a graph of the models with the
    best average preferred metric for each type of classifier with the 
    name 'selected_models.png'.

    Inputs:
        - classifiers: a dictionary with the (untrained) classifiers 
                       we want to use
        - parameters: a dictionary with the parameters we want to try out.
                      Each key must be associated with a key from the
                      classifiers dictionary
        - datasets: a dictionary of pairs of datasets - training and
                    testing sets
        - fractions: a list of floats where each number denotes the upper
                     percent of the population for which the precision and
                     recall will be evaluated
        - features: the list of features we want to use for all models
        - label: the label we want to use for all models
        - preferred_metric: the metric we will use to determine the best model
                           for each classifier

    Output: a Pandas dataframe - the evaluation table
    '''

    # Time stamp:
    begin = time.time()
    
    # Generating the df
    precision_cols = ['precision_at_' + str(i) for i in fractions]
    recall_cols = ['recall_at_' + str(i) for i in fractions]
    df = pd.DataFrame(columns=['Exact classifier', 'classifier', \
                               'parameters', 'dataset','baseline'] \
                               + precision_cols + recall_cols + ['AUC ROC'])
    
    # Counting the total number of models -- for on-the-run progress reporting
    total = 0
    n_datasets = len(datasets)
    for clf in classifiers:
        clf_n = 1
        for parameter in parameters[clf]:
            clf_n = clf_n * len(parameters[clf][parameter])
        total += clf_n
    total = total * n_datasets
    
    # Starting with the loop
    i = 1
    for dataset in datasets:

        # Generating datasets
        print('Dataset:', dataset)
        train_set, test_set = datasets[dataset]
        train_X = train_set[features]
        train_y = train_set[label]
        test_X = test_set[features]
        test_y = test_set[label]

        baseline = simple_classifier(test_y)

        for classifier in classifiers:

            parameters_list = list(ParameterGrid(parameters[classifier]))

            for parameter in parameters_list:

                # Progress reporting
                print('\nRunning model', i, 'out of', total)
                print('Progress:', round(i/total*100, 1), '%')
                print('Classifier:', classifier)
                i += 1

                # Estimating models and metrics
                clf = classifiers[classifier](**parameter)
                model = clf.fit(train_X, train_y)
                precision_metrics = [precision(model, fraction, 
                                     test_X, test_y) for fraction in fractions]
                recall_metrics = [recall(model, fraction, test_X, test_y) \
                                  for fraction in fractions]

                # Appending results
                size = len(df)
                df.loc[size] = [str(clf), classifier, parameter, dataset,
                                   baseline] + precision_metrics + \
                                   recall_metrics + \
                                   [area_under_curve(model, test_X, test_y)]

    df.to_csv('../outputs/evaluation_table.csv')
    graph = graph_models_best_average(df, preferred_metric)
    graph.figure.savefig('../outputs/selected_models.png')
    
    # Time stamp
    end = time.time()
    print('\nFinished!')
    print('This job took', round((end-begin)/60, 1), 'minutes to run')
    print('Good job!')

    #return df, best_models
    return df

def graph_models_best_average(df, metric):
    '''
    Evaluates the input dataframe (df), which is a table where each row
    is a classifier from and each column is a model performance indicator
    (the output returned in evaluation_table). Then it uses the metric
    provided as input (metric) to determine the model with the best
    average metric for each classifier, and graphs each one of those best
    models by classifier to show their performance in the selected metric.

    Inputs:
        - df: 
        - metric: the metric we will use to determine the best model
                  for each classifier

    Output: graph object
    '''
    plot_df = pd.DataFrame(columns=['classifier', 'dataset', metric])
    sets = df['dataset'].unique()
    for classifier in df['classifier'].unique():
        temp_df = df[df['classifier']==classifier][['Exact classifier', 'classifier', 'dataset', metric]]
        grouped_df = temp_df.groupby(['Exact classifier', 'classifier']).mean().\
                     sort_values(by=[metric], ascending=False)
        model = grouped_df.index[0][0]

        for dataset in sets:
            perf_metric = temp_df[temp_df['Exact classifier']==model][temp_df['dataset']==dataset][metric].values[0]
            plot_df.loc[len(plot_df)] = classifier, dataset, perf_metric

    graph = sns.lineplot(x='dataset', y=metric, data=plot_df, hue='classifier')
    
    return graph

def model_best_average(df, metric):
    '''
    Evaluates the input dataframe (df), which is a table where each row
    is a classifier from and each column is a model performance indicator
    (the output returned in evaluation_table). Then it uses the metric
    provided as input (metric) to determine the model with the best average
    metric, and returns that model and the average value of its best metric.

    Inputs:
        - df: 
        - metric: the metric we will use to determine the best model
                  for each classifier

    Output: graph object
    '''

    best_metric = 0
    best_model = None

    for classifier in df['classifier'].unique():
        
        temp_df = df[df['classifier']==classifier][['Exact classifier', 'classifier', 'dataset', metric]]
        grouped_df = temp_df.groupby(['Exact classifier', 'classifier']).mean().\
                     sort_values(by=[metric], ascending=False)
        model = grouped_df.index[0][0]
        perf_metric = grouped_df[metric].iloc[0]
        
        if perf_metric > best_metric:
            best_metric = perf_metric
            best_model = eval(model)

    return best_model, best_metric