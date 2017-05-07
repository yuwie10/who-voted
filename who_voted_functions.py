import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score



#Plotting functions

def calc_percent_response(df, col1, col2 = 'voted'):
    '''
    Calculate percent of observations in a category 
    that did and did not vote
    '''
    grouped = df.groupby([col1, col2]).size().unstack(fill_value = 0)
    grouped['%N'] = grouped['N'] / (grouped['N'] + grouped['Y'])
    grouped['%Y'] = grouped['Y'] / (grouped['N'] + grouped['Y'])
    return grouped

def plot_categ_by_voted(df, col, col_response = 'voted', save = False):
    '''
    Plot countplots and proportions of categories grouped by Y and N
    '''
    percents = calc_percent_response(df, col1 = col, col2 = col_response)
    order = percents.index.tolist()
    percents = percents['%N'].tolist() + percents['%Y'].tolist()
    
    _ = sns.set(font_scale = 1);
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10);
    ax = sns.countplot(x = col, hue = col_response, 
                       data = df, 
                       order = order, hue_order = ['N', 'Y'])
    for p, i in zip(ax.patches, percents):
        height = p.get_height()
        _ = ax.text(p.get_x() + p.get_width()/2,
                    height + 10,
                    '{:1.2f}'.format(i),
                    ha="center");
    _ = plt.setp(ax.get_xticklabels(), rotation=45);
    if save:
        ax.get_figure().savefig('figure_{}.png'.format(col))
    return ax

def calc_percent_categ(df, col, voted = 'voted'):
    '''
    What proportion of those who did or did not vote 
    were a given level of a categorical variable?
    '''
    
    response = ['N', 'Y']
    grouped = df.groupby([voted, col]).size().unstack(fill_value = 0)
    levels = grouped.columns.tolist()
    
    percents = []
    for j in levels:
        for i in response:
            percent = grouped.loc[i][j] / grouped.loc[i].sum()
            percents.append(percent)
    
    return grouped, percents

def plot_voted_by_categ(df, col, col_response = 'voted'):
    '''
    Plot countplots and proportions of individuals who did and did not vote
    grouped by category
    '''
    grouped, percents = calc_percent_categ(df, col = col, voted = col_response)
    hue_order = grouped.columns.tolist()
    
    _ = sns.set(font_scale = 1);
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10);
    ax = sns.countplot(x = col_response, hue = col, 
                       data = df, 
                       order = ['N', 'Y'], hue_order = hue_order)
    ax.set_title(col);
    for p, i in zip(ax.patches, percents):
        height = p.get_height()
        _ = ax.text(p.get_x() + p.get_width()/2,
                    height + 10,
                    '{:1.3f}'.format(i),
                    ha="center");
    return ax

def split_df(df, low, high, col = 'state_house'):
    '''
    Divide a df into sections based on a lower and upper bound
    '''
    return df.ix[(df[col] > low) & (df[col] < high)]

def plot_quant(df, col, col_response = 'voted_pred'):
    '''
    Plot distribution of quantitative variables by predicted response.
    '''
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9);

    _ = plt.hist(df.ix[df[col_response] == 'Y'][col],
                    bins = 50, label = 'Y')
    _ = plt.hist(df.ix[df[col_response] == 'N'][col],
                    bins = 50, label = 'N')
    
    _ = plt.legend();
    _ = plt.title('Distribution of ' + col + ' by voters');
    _ = plt.xlabel(col)
    _ = plt.ylabel('Counts')



#Load and save data

def load_data(file, str1 = 'congress_district', str2 = 'state_house', compression = 'infer'):
    '''
    Load voter dataset.
    Code 'congress_district' and 'state_house' as categorical variables.
    Replace any 'nan' strings with NaN.
    '''
    dtypes = {str1:str, str2:str}
    data = pd.read_csv(file, compression = compression, dtype = dtypes)
    data.replace({'nan':np.nan}, inplace = True)
    return data

def for_submission(clf_fit, X_test, series_id, submit_num):
    predictions = clf_fit.predict_proba(X_test)
    to_submit = pd.DataFrame(predictions[:, 1], columns = ['voted'])
    to_submit = pd.concat([series_id, to_submit], axis = 1)
    to_submit.to_csv('submit_{}.csv'.format(str(num)), index = False)
    return to_submit



#Prepping data to a ML model

def get_cols(df, response_col = 'voted'):
    '''
    Get list of quantitative and categorical variables.
    '''
    quant = []
    categ = []
    for col in df:
        if df[col].dtype != 'O':
            quant.append(col)
        elif col != response_col:
            categ.append(col)
    return quant, categ

def get_y_vector(df, col):
    '''
    Returns y vector of yes and no voters.
    '''
    dummy = pd.get_dummies(df[col])
    return dummy['Y'].as_matrix()

def design_Xmatrix(df, quant_cols, categ_cols, standardize = False, matrix = True):
    '''
    Converts a dataframe to an X matrix for ML.
    
    Inputs:
    df: dataframe to convert
    quant_cols: list of quantitative columns to standardize.
    categ_cols: list of categorical columns to one-hot encode
    
    Output: design matrix
    '''
    
    if standardize:
        #standardize continuous variables
        quant = df[quant_cols]
        quant = (quant - quant.mean()) / quant.std()
    else:
        quant = df[quant_cols]
    
    #convert categorical variables to dummy variables
    categ = pd.get_dummies(df[categ_cols], drop_first = True)
    
    #merge back into a single df
    X_df = pd.concat([quant, categ], axis = 1)

    if matrix:
        return X_df.as_matrix()

    else:
        return X_df

def transform_to_continuous(df, card_var, voted_counts, response_col = 'voted'):
    '''
    Transform high cardinal data to a continuous variable to use in predictive model.
    The two transformations are: 
    
    1) supervised ratio = #positive instances / #total instances
    2) weight of evidence = ln((#positive instances/total positives)/(#negative instances/total negatives))
    
    Input: df grouped by cardinal variable and voted, list of unique cardinal variables, and 
    series with counts of those who voted and did not vote
    Output: dictionary of cardinal variable (key) and transformation (value)
    '''

    #creates a dataframe where counts of response is grouped by the cardinal variable
    counts = pd.DataFrame(df.groupby([card_var, response_col]).size(), columns = ['counts'])

    #list of distinct values of cardinal variable
    list_unique = counts.index.levels[0].tolist()
    
    ratios = {}
    woe = {}
    for i in list_unique:
        pos = counts.loc[i, 'Y']
        neg = counts.loc[i, 'N']
        
        #supervised ratio
        value = (pos / (pos + neg)).item()
        ratios[i] = value
        
        #weight of evidence
        v = np.log((pos.item()/voted_counts['Y'])/(neg.item()/voted_counts['N']))
        woe[i] = v
    
    return ratios, woe

def create_card_df(df, list_cols, list_replace):
    '''
    Create dataframe where the cardinal variables have been 
    replaced with the corresponding transformed values.
    '''
    df_copy = df.copy()
    for col, replace in zip(list_cols, list_replace):
        df_copy[col].replace(replace, inplace = True)
    return df_copy



#Evaluate models

def cross_val_LL(clf, X, y, cv):
    scores = cross_val_score(clf, X, y, cv = cv, scoring = 'neg_log_loss')
    return np.mean(scores), np.std(scores)

def recombine_categorical(df, cols_list, col1 = 'features', 
    col2 = 'importance'):
    '''Recombine categorical variables to determine feature importance'''

    df_features = pd.DataFrame(columns=[col1, col2])
    for col in cols_list:
        df1 = df.ix[df[col1].str.contains(col)] #rows with categorical variable
        df_features = df_features.append({col1:col, 
            col2:df1[col2].sum()}, ignore_index = True)
    return df_features

def find_important_features(df, clf, categorical_cols, continuous_cols, 
    col1 = 'features', col2 = 'importance'):
    '''
    Create a dataframe that lists the important features 
    used by the classifier to make decisions.

    Inputs:
    df: the dataframe to get column names from
    clf: the fitted classifier
    categorical_cols: list of categorical column names
    continuous_cols: list of continuous column names
    '''
    list_cols = df.columns.tolist()
    feature_importance = pd.DataFrame(list(zip(list_cols, 
        clf.feature_importances_)), columns = [col1, col2])
    categorical = recombine_categorical(feature_importance, 
        categorical_cols, col1 = col1, col2 = col2)
    features = pd.DataFrame(columns = [col1, col2])
    for col in continuous_cols:
        row = feature_importance.ix[feature_importance[col1] == col]
        features = features.append(row)
    features = features.append(categorical)
    features = features.sort_values(col2, ascending = False)
    return features


