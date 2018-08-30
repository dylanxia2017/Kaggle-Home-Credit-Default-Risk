from sklearn.externals import joblib
import pandas as pd
from tqdm import tqdm
import numpy as np
from lightgbm import LGBMClassifier, plot_importance, plot_metric, plot_tree, LGBMRanker
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import itertools
import gc
import collections
import time
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import PLSRegression as PLS
from sklearn.linear_model import LinearRegression

def aggrate_trend(series):
    if series.size == 1:
        # just 1 record, no trends
        return 3
    trend = None
    prev = None
    for i, v in series.iteritems():
        if prev != None:
            if trend == None:
                if prev <= v:
                    # increasing trend
                    trend = 0
                else:
                    # decreasing trend
                    trend = 1
            elif trend == 'increasing':
                if prev > v:
                    # not a stable trend
                    return 2
            else:
                if prev <= v:
                    # not a stable trend
                    return 2
        prev = v
    return trend 


def rename_value_column_name(prefix):
    def fun(column):
        if 'SK_ID' not in str(column):
            return prefix + '_' + str(column)
        else:
            return column
    return fun

def agg_sum(groupby, prefix, column, rename_col):
    if rename_col:
        n = prefix+'_'+rename_col+'_SUM'
    else:
        n = prefix+'_'+column+'_SUM'
    t = groupby[column]\
    .sum().reset_index().rename(columns={column:n})
    return t

def agg_max(groupby, prefix, column, rename_col):
    if rename_col:
        n = prefix+'_'+rename_col+'_MAX'
    else:
        n = prefix+'_'+column+'_MAX'
    t = groupby[column]\
    .max().reset_index().rename(columns={column:n})
    return t

def agg_min(groupby, prefix, column, rename_col):
    if rename_col:
        n = prefix+'_'+rename_col+'_MIN'
    else:
        n = prefix+'_'+column+'_MIN'
    t = groupby[column]\
    .min().reset_index().rename(columns={column:n})
    return t

def agg_median(groupby, prefix, column, rename_col):
    if rename_col:
        n = prefix+'_'+rename_col+'_MEDIAN'
    else:
        n = prefix+'_'+column+'_MEDIAN'
    t = groupby[column]\
    .median().reset_index().rename(columns={column:n})
    return t

def agg_mean(groupby, prefix, column, rename_col):
    if rename_col:
        n = prefix+'_'+rename_col+'_MEAN'
    else:
        n = prefix+'_'+column+'_MEAN'
    t = groupby[column]\
    .mean().reset_index().rename(columns={column:n})
    return t

def agg_std(groupby, prefix, column, rename_col):
    if rename_col:
        n = prefix+'_'+rename_col+'_STD'
    else:
        n = prefix+'_'+column+'_STD'
    t = groupby[column]\
    .std().reset_index().rename(columns={column:n})
    return t
def agg_nunique(groupby, prefix, column, rename_col):
    if rename_col:
        n = prefix+'_'+rename_col+'_UNIQUE'
    else:
        n = prefix+'_'+column+'_UNIQUE'
    t = groupby[column]\
    .nunique().reset_index().rename(columns={column:n})
    return t

def get_trend(series):
    y = series.values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression(n_jobs=16)
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    return trend

import math
def get_first_k(k_portion, gy_cols):
    def t(df):
        l = df.shape[0] * k_portion
        l = math.ceil(l)
        df = df.drop(columns=gy_cols)
        return df[:l]
    return t

def get_last_k(k_portion, gy_cols):
    def t(df):
        l = df.shape[0] * k_portion
        l = math.ceil(l)
        df = df.drop(columns=gy_cols)
        return df[-1*l:]
    return t


def agg_trend(groupby, prefix, column, rename_col):
    print('agg_trend')
    if rename_col:
        n = prefix+'_'+rename_col+'_TREND'
    else:
        n = prefix+'_'+column+'_TREND'
    t = groupby[column]\
    .agg(get_trend).reset_index().rename(columns={column:n})
    return t

def agg_num(groupby, prefix, column, ops, rename_col = None):
    ts = []
    if 'sum' in ops:
        ts.append(agg_sum(groupby, prefix, column, rename_col))
    if 'max' in ops:
        ts.append(agg_max(groupby, prefix, column, rename_col))
    if 'mean' in ops:
        ts.append(agg_mean(groupby, prefix, column, rename_col))
    if 'min' in ops:
        ts.append(agg_min(groupby, prefix, column, rename_col))
    if 'nunique' in ops:
        ts.append(agg_nunique(groupby, prefix, column, rename_col))
    if 'median' in ops:
        ts.append(agg_median(groupby, prefix, column, rename_col))
    if 'std' in ops:
        ts.append(agg_std(groupby, prefix, column, rename_col))
    if 'trend' in ops:
        ts.append(agg_trend(groupby, prefix, column, rename_col))
    return ts

def aggreate_value_count(groupby, prefix, column, rename_col = None):
    if rename_col:
        c = rename_col
    else:
        c = column
    t = groupby[column]\
        .value_counts().unstack().reset_index().rename(rename_value_column_name('VC_'+prefix + '_' + c), axis=1).fillna(0)
    return t


def train_KNN(n_neighbors, KNN_features, X_train, y_train):
    X_train = X_train[KNN_features]
    # fillna
    for feature in KNN_features:
        X_train[feature] = X_train[feature].fillna(X_train[feature].mean())  
       
            
    # train
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    clf.fit(X_train, y_train) 
    return clf

def get_KNN_features(clf, n_neighbors, KNN_features, X_train, y_train, df, is_train=False):
    df = df[KNN_features]
    for feature in KNN_features:
         df[feature] = df[feature].fillna(X_train[feature].mean())
    
    _, indices_lst = clf.kneighbors(df, n_neighbors=n_neighbors)
    if is_train:
        indices_lst = [indices[1:] for indices in indices_lst]

    return [np.mean([y_train.iloc[index] for index in indices]) for indices in indices_lst]


def cross_feature(result_df, source_df, f1, f2, op):
    if op == '-':
        name = 'DIFF%'+f1+'%'+f2
        result_df[name] = source_df[f1].values - source_df[f2].values
    elif op == '/':
        name = 'RATIO%'+f1+'%'+f2
        result_df[name] = source_df[f1].values / source_df[f2].values
    elif op == '+':
        name = 'ADD%'+f1+'%'+f2
        result_df[name] = source_df[f1].values + source_df[f2].values
    elif op == '*':
        name = 'MUL%'+f1+'%'+f2
        result_df[name] = source_df[f1].values * source_df[f2].values
        

def validate_model_lgb(df, categorical_feature, params, sub_prefix='result/', num_fold=5, random_state=850, n_components=10):
    # print(categorical_feature)
    start = time.time()
    sub_X = df[df['TARGET'].isna()]
    sub_X.pop('TARGET')
    sub_ids = sub_X.pop('SK_ID_CURR')
    X = df[df['TARGET'].notna()]
    Y = X[['SK_ID_CURR', 'TARGET']]
    X.pop('TARGET')
    train_SK_ID_CURR = X.pop('SK_ID_CURR')

    X = X.replace([np.inf, -np.inf], 99999)
    sub_X = sub_X.replace([np.inf, -np.inf], 99999)
    skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=random_state)
    valid_scores = []
    train_scores = []
    sub_prediction = None
    importances = None

    oof_df = {
        'SK_ID_CURR': [],
        'OOF':[]
    }
    categorical_feature = [f for f in categorical_feature if f in X.columns]
    
    for train_index, test_index in skf.split(X, Y['TARGET']):
        gc.collect()
        X_train, X_validation = X.iloc[train_index], X.iloc[test_index]
        y_train, y_validation = Y['TARGET'].iloc[train_index], Y['TARGET'].iloc[test_index]
        
        # encoded_sub_X = sub_X.copy()
        # temp_X_train = X_train.copy()
        # temp_X_validation = X_validation.copy()
        # temp_sub_X = encoded_sub_X.copy()

        # for feature in temp_X_train.columns:
        #     temp_X_train[feature] = temp_X_train[feature].fillna(temp_X_train[feature].mean())
              
        #     temp_X_validation[feature] = temp_X_validation[feature].fillna(temp_X_validation[feature].mean())
            
        #     temp_sub_X[feature] = temp_sub_X[feature].fillna(temp_sub_X[feature].mean())
        
        # temp_X_train = temp_X_train.fillna(0)
        # temp_X_validation = temp_X_validation.fillna(0)  
        # temp_sub_X = temp_sub_X.fillna(0)
        
        # scaler = RobustScaler()
        # temp_X_train[temp_X_train.columns] = pd.DataFrame(scaler.fit_transform(temp_X_train[temp_X_train.columns]))
        # temp_X_validation[temp_X_validation.columns] = pd.DataFrame(scaler.transform(temp_X_validation[temp_X_validation.columns]))
        # temp_sub_X[temp_sub_X.columns] = pd.DataFrame(scaler.transform(temp_sub_X[temp_sub_X.columns]))
        
        # temp_X_train = temp_X_train.fillna(0)
        # temp_X_validation = temp_X_validation.fillna(0)  
        # temp_sub_X = temp_sub_X.fillna(0)

        # ### LDA ###
        # reducer = PLS(n_components=n_components)
        # reducer.fit(temp_X_train[temp_X_train.columns], y_train)
        # train_reduced_samples = pd.DataFrame(reducer.transform(temp_X_train[temp_X_train.columns]))
        # valid_reduced_samples = pd.DataFrame(reducer.transform(temp_X_validation[temp_X_validation.columns]))
        # sub_reduced_samples = pd.DataFrame(reducer.transform(temp_sub_X[temp_sub_X.columns])) 


        # for feature in train_reduced_samples.columns:
        #     X_train['LDA_'+str(feature)] = train_reduced_samples[feature].values

        # for feature in valid_reduced_samples.columns:
        #     X_validation['LDA_'+str(feature)] = valid_reduced_samples[feature].values

        # for feature in sub_reduced_samples.columns:
        #     encoded_sub_X['LDA_'+str(feature)] = sub_reduced_samples[feature].values
        # ### LDA ###


        ### MEAN ENCODING ###
        encoder = ce.TargetEncoder(cols=categorical_feature)
        encoder.fit(X_train, y_train)
        X_train = encoder.transform(X_train, y_train)
        X_validation = encoder.transform(X_validation)
        encoded_sub_X = encoder.transform(sub_X)
        ### MEAN ENCODING ###

        clf = LGBMClassifier(
            boosting_type=params['boosting_type'],
            objective=params['objective'],
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            num_leaves=params['num_leaves'],
            feature_fraction=params['feature_fraction'],
            subsample=params['subsample'],
            max_depth=params['max_depth'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            min_child_weight=params['min_child_weight'],
            random_state=random_state,
            verbose=-1,
            nthread=-1
            )
                   
        clf.fit(X_train, y_train, 
                eval_set=[(X_train, y_train), (X_validation, y_validation)], 
                eval_metric='auc',
                verbose = params['verbose'],
                feature_name=list(X_train.columns),
                early_stopping_rounds=1000
                )
        
        train_prediction = clf.predict_proba(X_train)[:, 1]
        train_score = roc_auc_score(y_train, train_prediction)
        train_scores.append(train_score)
        
        valid_prediction = clf.predict_proba(X_validation)[:, 1]
        oof_df['SK_ID_CURR'] += list(train_SK_ID_CURR.iloc[test_index])
        oof_df['OOF'] += list(valid_prediction)

        valid_score = roc_auc_score(y_validation, valid_prediction)
        valid_scores.append(valid_score)
        
        print('Fold', train_score, valid_score, train_score - valid_score, clf.best_iteration_)
        # return clf
        if sub_prediction is None:
            sub_prediction = clf.predict_proba(encoded_sub_X)[:, 1]
        else:
            sub_prediction += clf.predict_proba(encoded_sub_X)[:, 1]
            
        if importances is None:
            importances = clf.booster_.feature_importance(importance_type='gain')
        else:
            importances += clf.booster_.feature_importance(importance_type='gain')
        features = X_train.columns

        del X_train
        del X_validation

    print('\t','AUC mean:', np.mean(valid_scores), 'std:',
          np.std(valid_scores), 'min:', np.min(valid_scores), 'max:', np.max(valid_scores))
    
    oof_df = pd.DataFrame(oof_df)
    oof_df = pd.merge(oof_df, Y, on='SK_ID_CURR', how='left')
    oof_score = roc_auc_score(oof_df['TARGET'], oof_df['OOF'])
    print('\t','OOF AUC:', oof_score)

    submission_df = pd.DataFrame(sub_ids)
    submission_df['TARGET'] = sub_prediction / num_fold
    
    path = sub_prefix + "{:.5f}".format(np.mean(valid_scores)) +'CV_SUB' '.csv'
    print('Save submission to', path)
    submission_df.to_csv(path, index=False)
    
    path = sub_prefix + "{:.5f}".format(np.mean(valid_scores)) +'CV_OOF' '.csv'
    oof_df.to_csv(path, index=False)
    # joblib.dump(clf, 'lgm.pkl')
    print('it took', time.time() - start, 'seconds.')
    return np.mean(train_scores), np.mean(valid_scores), clf, list(zip(features, importances / num_fold))


def divided_statistic(aggrated_df, division_config, numerical_features_config, categorical_features_config):
    prefix = division_config['prefix']
    dfs = division_config['dfs']
    for idx, df in enumerate(dfs):
        df.sort_values(division_config['column'], ascending=False, inplace=True)
        for group in division_config['groups']:
            cut = group * division_config['step']
            grouped_df = df[df[division_config['column']] >= cut]

            # for feature in categorical_features_config:
            #     grouped_df[feature] = grouped_df[feature].fillna('XNA')
            #     grouped_df[feature], _ = pd.factorize(grouped_df[feature])
            grouped_gy = grouped_df.groupby(division_config['groupby_column'])
            division_config['grouped_dfs'][idx] = division_config['grouped_dfs'].get(idx, []) + [grouped_df]
            division_config['grouped_gys'][idx] = division_config['grouped_gys'].get(idx, []) + [grouped_gy]

    # global
    for idx, df in enumerate(dfs):
        grouped_df = df

        # for feature in categorical_features_config:
        #     grouped_df[feature] = grouped_df[feature].fillna('XNA')
        #     grouped_df[feature], _ = pd.factorize(grouped_df[feature])


        grouped_gy = grouped_df.groupby(division_config['groupby_column'])
        division_config['grouped_dfs'][idx] = division_config['grouped_dfs'].get(idx, []) + [grouped_df]
        division_config['grouped_gys'][idx] = division_config['grouped_gys'].get(idx, []) + [grouped_gy]
    division_config['groups'].append('global')

    if division_config['count_all_dfs']:
        for idx, df in enumerate(dfs):
            for group, gy in zip(division_config['groups'], division_config['grouped_gys'][idx]):
                aggrated_df[prefix+'_COUNT_'+str(group)+'_'+str(idx)] = gy.size().reset_index(name='count')['count']
    else:
        for group, gy in zip(division_config['groups'], division_config['grouped_gys'][0]):
            aggrated_df[prefix+'_COUNT_'+str(group)] = gy.size().reset_index(name='count')['count']

    for df_idx, df in enumerate(dfs):
        for f in numerical_features_config:
            time_gys = division_config['grouped_gys'][df_idx]
            groups = division_config['groups']

            for idx, group in enumerate(groups):
                ts = agg_num(time_gys[idx], prefix, f[0], f[1], f[0]+'_'+str(group)+'M_DF_'+str(df_idx))
                for i in range(0, len(ts)):
                    aggrated_df = pd.merge(aggrated_df, ts[i], how='left', on='SK_ID_CURR')  

    for f in categorical_features_config: 
        time_gys = division_config['grouped_gys'][0]
        groups = division_config['groups']

        for idx, group in enumerate(groups):
            # TODO try not aggreate with value count
            t = aggreate_value_count(time_gys[idx], prefix, f, f+'_'+str(group)+'M')                        
            aggrated_df = pd.merge(aggrated_df, t, how='left', on='SK_ID_CURR')

            # ts = agg_num(time_gys[idx], prefix, f, ['mean', 'sum', 'max'], f+'_'+str(group)+'M_DF_'+str(df_idx))
            # for i in range(0, len(ts)):
            #     aggrated_df = pd.merge(aggrated_df, ts[i], how='left', on='SK_ID_CURR')  


    for df in dfs:
        del df
    for group in groups:
        del group
    return aggrated_df

    