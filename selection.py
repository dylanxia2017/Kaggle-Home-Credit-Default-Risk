import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import time
from lightgbm import LGBMClassifier
import lightgbm as lgb
import category_encoders as ce
import warnings
warnings.simplefilter('ignore', UserWarning)
import gc
gc.enable()

nb_runs = 10
print('reading csv...')
data = pd.read_csv('../data/application_train.csv')


def get_feature_importances(data, shuffle, seed=None):
    data = data[data['TARGET'].notna()]
    # Gather real features
    categorical_features= ['EMERGENCYSTATE_MODE', 'ORGANIZATION_TYPE', 'CODE_GENDER', 'REGION_RATING_CLIENT',
        'WALLSMATERIAL_MODE', 'NAME_CONTRACT_TYPE', 'NAME_FAMILY_STATUS', 'FONDKAPREMONT_MODE', 'FLAG_OWN_CAR',
        'NAME_TYPE_SUITE', 'OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 'FLAG_OWN_REALTY',
        'REGION_RATING_CLIENT_W_CITY', 'NAME_HOUSING_TYPE', 'HOUSETYPE_MODE', 'WEEKDAY_APPR_PROCESS_START']

    train_features = [f for f in data if f not in ['TARGET', 'SK_ID_CURR']]
    categorical_features = [f for f in categorical_features if f in train_features]
    # Go over fold and keep track of CV score (train and valid) and feature importances
    
    # Shuffle target if required
    y = data['TARGET'].copy()
    if shuffle:
        # Here you could as well use a binomial distribution
        y = data['TARGET'].copy().sample(frac=1.0)
    

    ### MEAN ENCODING ###
    encoder = ce.TargetEncoder(cols=categorical_features)
    encoder.fit(data, y)
    data = encoder.transform(data, y)
    ### MEAN ENCODING ###
    params = {      
        'boosting_type':'gbdt',
        'objective':'binary',
        'n_estimators':  500,
        'learning_rate':0.1,
        'num_leaves': 123,
        'feature_fraction': 0.1,
        'subsample':0.8715623,
        'max_depth': 4,
        'reg_alpha': 10,
        'reg_lambda':0.0735294,
        'min_child_weight': 20,
        'verbose': False,
    }

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
            random_state=seed,
            verbose=-1,
            nthread=-1
        )
                   
    clf.fit(data[train_features], y, 
            eval_metric='auc',
            verbose = False,
            feature_name=train_features
            )

    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    
    imp_df["importance_gain"] = clf.booster_.feature_importance(importance_type='gain')
    imp_df["importance_split"] = clf.booster_.feature_importance(importance_type='split')
    train_prediction = clf.predict_proba(data[train_features])[:, 1]
    train_score = roc_auc_score(y, train_prediction)
    imp_df['trn_score'] = train_score
    del data
    return imp_df


# Seed the unexpected randomness of this world
np.random.seed(850)
# Get the actual importance, i.e. without shuffling
print('getting actual importance...')
actual_imp_df = get_feature_importances(data=data, shuffle=False)

print(actual_imp_df.head(30))

null_imp_df = pd.DataFrame()
print('getting null importance...')
import time
start = time.time()
dsp = ''
for i in range(nb_runs):
    # Get current run importances
    imp_df = get_feature_importances(data=data, shuffle=True)
    imp_df['run'] = i + 1 
    # Concat the latest importances with the old ones
    null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
    # Erase previous message
    for l in range(len(dsp)):
        print('\b', end='', flush=True)
    # Display current run and time used
    spent = (time.time() - start) / 60
    dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    print(dsp, end='', flush=True)

print(null_imp_df.head(30))

feature_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps_gain = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps_gain = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].mean()
    gain_score = np.log(1e-10 + f_act_imps_gain / (1 + np.percentile(f_null_imps_gain, 75)))  # Avoid didvide by zero
    f_null_imps_split = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps_split = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].mean()
    split_score = np.log(1e-10 + f_act_imps_split / (1 + np.percentile(f_null_imps_split, 75)))  # Avoid didvide by zero
    feature_scores.append((_f, split_score, gain_score))

scores_df = pd.DataFrame(feature_scores, columns=['feature', 'split_score', 'gain_score'])

null_imp_df.to_csv('null_importances_distribution_rf.csv')
actual_imp_df.to_csv('actual_importances_ditribution_rf.csv')

correlation_scores = []
for _f in actual_imp_df['feature'].unique():
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_gain'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_gain'].values
    gain_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance_split'].values
    f_act_imps = actual_imp_df.loc[actual_imp_df['feature'] == _f, 'importance_split'].values
    split_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum() / f_null_imps.size
    correlation_scores.append((_f, split_score, gain_score))

print(correlation_scores)

import json
json.dump(correlation_scores, open('correlation_scores.json', 'w'))
