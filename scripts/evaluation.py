import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


def get_predictions(classifier, X_test):
    '''
    Returns a Pandas Series with the prediction scores.

    Inputs:
        - classifier object
        - X_test: test dataset (Pandas)
    Output: Pandas series with the prediction scores
    '''

    if hasattr(classifier, 'predict_proba'):
        pred_scores = pd.Series(classifier.predict_proba(X_test)[:,1])
    else:
        pred_scores = pd.Series(classifier.decision_function(X_test))

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

    if mean >= 0.5:
        print("Predicting every data point's value to be 1, " + \
              "the accuracy is", round(mean*100, 1), "%")

        return mean

    else:
        acc = 1 - mean
        print("Predicting every data point's value to be 0, " + \
              "the accuracy is", round(acc*100, 1), "%")

        return acc


def accuracy(classifier, threshold, X_test, y_test):
    '''
    Returns the accuracy (float) of a classifier given a certain threshold,
    and a certain test set (X_test and y_test).

    Inputs:
        - classifier: the model we are using
        - threshold: the threshold we use to calculate accuracy
        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: accuracy (float)
    '''

    pred_scores = get_predictions(classifier, X_test)
    pred_label = np.where(pred_scores >= threshold, 1, 0)
    acc = accuracy_score(pred_label, y_test)

    return acc


def precision(classifier, threshold, X_test, y_test):
    '''
    Returns the precision (float) of a classifier given a certain
    threshold, and a certain test set (X_test and y_test).

    Inputs:
        - classifier: the model we are using
        - threshold: the threshold we use to calculate precision
        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: precision (float)
    '''

    pred_scores = get_predictions(classifier, X_test)
    pred_label = np.where(pred_scores >= threshold, 1, 0)
    c = confusion_matrix(y_test, pred_label)
    true_negatives, false_positive, false_negatives, true_positives = c.ravel()
    prec = true_positives / (false_positive + true_positives)

    return prec


def recall(classifier, threshold, X_test, y_test):
    '''
    Returns the recall (float) of a classifier given a certain
    threshold, and a certain test set (X_test and y_test).

    Inputs:
        - classifier: the model we are using
        - threshold: the threshold we use to calculate recall
        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: recall (float)
    '''

    pred_scores = get_predictions(classifier, X_test)
    pred_label = np.where(pred_scores >= threshold, 1, 0)
    c = confusion_matrix(y_test, pred_label)
    true_negatives, false_positive, false_negatives, true_positives = c.ravel()
    rec = true_positives / (false_negatives + true_positives)

    return rec


def f1(classifier, threshold, X_test, y_test):
    '''
    Returns the f1 score (float) of a classifier given a certain
    threshold, and a certain test set (X_test and y_test).

    Inputs:
        - classifier: the model we are using
        - threshold: the threshold we use to calculate the f1 score
        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: f1 score (float)
    '''

    pred_scores = get_predictions(classifier, X_test)
    pred_label = np.where(pred_scores >= threshold, 1, 0)
    score = f1_score(y_test, pred_label)

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

def evaluation_table(classifiers, X_test, y_test):
    '''
    (Please notice that this function might take a while to run)
    Returns a dataframe where each row is a classifier from classifiers
    and each column is a model performance indicator. Each classifier
    is evaluated on the same features and the same label.

    Inputs:
        - classifiers: a list of the classifiers we want to evaluate
        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: a Pandas dataframe - the evaluation table
    '''

    df = pd.DataFrame()
    fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    df['classifier'] = classifiers
    df['features'] = X_test.columns
    df['baseline'] = [simple_classifier(y_test)]*len(df)

    sorted_predictions = []
    for classifier in classifiers:
        sorted_predictions.append(get_predictions(classifier, X_test))

    for metric in ['precision', 'recall']:
        
        for fraction in fractions:

            l = []
            i = 0

            for classifier in classifiers:

                pred_scores = sorted_predictions[i]
                threshold = pred_scores.quantile(1 - fraction)

                if metric == 'precision':

                    l.append(precision(classifier, threshold, X_test, y_test))

                else:

                    l.append(recall(classifier, threshold, X_test, y_test))

                i += 1

            col_name = metric + '_' + str(fraction)
            df[col_name] = l

    auc_values = []
    for classifier in classifiers:
        auc_values.append(area_under_curve(classifier, X_test, y_test))
    df['auc_roc'] = auc_values
    
    return df


def evaluation_table2(classifiers, X_datasets, y_test):
    '''
    (Please notice that this function might take a while to run)
    Returns a dataframe where each row is a classifier from classifiers
    and each column is a model performance indicator. The classifiers
    must be one-feature models, and the i-th classifier from classifiers
    is evaluated on the i-th feature from X_datasets. They all use
    the same label (y_test).
    The difference between this function and evaluation_table is that
    in this one each X set can be different, while evaluation_table uses the
    same X set for all the classifiers. Also, each X set is a one-feature
    Pandas series.

    Inputs:
        - classifiers: a list of the classifiers we want to evaluate
        - X_datasets: a list of Pandas series with the features we want to
                      evaluate.
        - y_test: a Pandas series with the label of the test set
    Output: a Pandas dataframe - the evaluation table
    '''

    df = pd.DataFrame()
    fractions = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]

    df['classifier'] = classifiers
    df['features'] = [X.name for X in X_datasets]
    df['baseline'] = [simple_classifier(y_test)]*len(df)

    sorted_predictions = []
    for i, classifier in enumerate(classifiers):
        sorted_predictions.append(get_predictions(classifier, \
            X_datasets[i].values.reshape(-1, 1)))

    for metric in ['precision', 'recall']:
        
        for fraction in fractions:

            l = []
            i = 0

            for num, classifier in enumerate(classifiers):

                pred_scores = sorted_predictions[i]
                threshold = pred_scores.quantile(1 - fraction)

                if metric == 'precision':

                    l.append(precision(classifier, threshold, \
                        X_datasets[num].values.reshape(-1, 1), y_test))

                else:

                    l.append(recall(classifier, threshold, \
                        X_datasets[num].values.reshape(-1, 1), y_test))

                i += 1

            col_name = metric + '_' + str(fraction)
            df[col_name] = l

    auc_values = []
    for num, classifier in enumerate(classifiers):
        auc_values.append(area_under_curve(classifier, \
            X_datasets[num].values.reshape(-1, 1), y_test))
    df['auc_roc'] = auc_values
    
    return df