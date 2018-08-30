import pandas as pd
import json

aggrated_df = pd.read_csv('ag.csv')
sorted_features = json.load(open('sorted_features.json'))   

p1= pd.read_csv('version 2.3_external_1_predict_revised.csv')
p2= pd.read_csv('version 2.3_external_1_predict.csv')
p3= pd.read_csv('version 2.3_external_3_predict.csv')

i = 1000
top_features = [f[0] for f in sorted_features][:i]

df = aggrated_df[top_features + ['TARGET', 'SK_ID_CURR']]
df = pd.merge(df, p2, how='left', on='SK_ID_CURR')
df = pd.merge(df, p3, how='left', on='SK_ID_CURR')

df.to_csv('top1000_data.csv', index=False)