from typing import DefaultDict
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

def c_v(df,K=2,random_state=1342, shuffle = True):
    skf = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=shuffle)
    label = df['target_relabeled'] == 1
    print('Whole Training Set Shape = {}'.format(df.shape))
    print('Whole Training Set Unique keyword Count = {}'.format(df['keyword'].nunique()))
    print('Whole Training Set Target Rate (Disaster) {}/{} (Not Disaster)'.format(df[label]['target_relabeled'].count(), df[~label]['target_relabeled'].count()))

    rs = []
    for fold, (t_idx, v_idx) in enumerate(skf.split(df['text_cleaned'], df['target_relabeled']), 1):
        print('\nFold {} Training Set Shape = {} - Validation Set Shape = {}'.format(fold, df.loc[t_idx, 'text_cleaned'].shape, df.loc[v_idx, 'text_cleaned'].shape))
        print('Fold {} Training Set Unique keyword Count = {} - Validation Set Unique keyword Count = {}'.format(fold, df.loc[t_idx, 'keyword'].nunique(), df.loc[v_idx, 'keyword'].nunique()))
        rs.append((t_idx,v_idx))   
    return rs
# test

# rs = c_v(df=df)