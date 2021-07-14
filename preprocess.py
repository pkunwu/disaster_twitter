__author__ = 'Gunes Evitan@Kaggle.com'

from load import load
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import STOPWORDS
import string
import pandas as pd

#count the ratio of the missing values. df for dataframe. cols for selected column names. kw for graph. return a dict, keys are cols, values are ratio.
def missingvalues(*,df,cols, **kw):
    if cols == []:
        raise ValueError('columns cannot be empty')
    d = {}
    for col in cols:
        d[col] = df[col].isnull().sum()/len(df)
    if kw == {}:
        return d
    elif 'graph' in kw:
        fig, axes = plt.subplots(ncols=1, figsize=(4*len(cols), 4), dpi=100)

        plt.bar(d.keys(), d.values())
        plt.show()
        return d

#test
# filepath = r'..\twitter_disaster\data\train.csv'
# df = load(filepath)
# cols = ['keyword', 'location']
# rs = missingvalues(df = df, cols = cols)

#df for dataframe. cols for column names. print the number of unique values of the given column.
def unique(df, cols):
    for col in cols:
        print(f'Unique values in {col}: {df[col].nunique()}')

#test
# unique(df, cols)

#df for dataframe. col of column name. label for label column. return a dataframe, rows are unique values of col, values are total labelled numbers and corresponding ratios.
def distr(df, col, label, **kw):
    rs = df.groupby(col)[label].agg(['count', 'sum'])
    rs = rs.sort_values('count',ascending = False)
    rows = df[col].nunique()
    if 'graph' in kw:
        sns.set_theme(style="whitegrid")
        f, ax = plt.subplots(figsize=(10, rows))
        sns.set_color_codes("pastel")
        sns.barplot(x='count', y = rs.index, data = rs,color='b',label = 'Total')
        sns.set_color_codes("muted")
        sns.barplot(x="sum", y=rs.index, data=rs, label="Targets", color="b")
        ax.legend(ncol=2, loc="lower right", frameon=True)
        sns.despine(left=True, bottom=True)
        plt.savefig(kw['graph'])
    return rs
#test
# col = 'keyword'
# label = 'target'

#count the word, unique word, char, stop word, url, punctuation, #, @ 
def word_counts(df, col, **kw):
    counts = {
        'word_num': lambda x: len(str(x).split()),
        'unique_word_num': lambda x: len(set(str(x).split())),
        'char_num': lambda x: len(list(str(x))),
        'stop_word_num': lambda x: len([y for y in str(x).lower().split() if y in STOPWORDS]),
        'url_num': lambda x: len([y for y in str(x).lower().split() if 'http' in y or 'https' in y or 'ftp' in y]),
        'punc_num': lambda x: len([y for y in str(x) if y in string.punctuation]),
        '#_num': lambda x: len([y for y in str(x) if y == '#']),
        '@_num': lambda x: len([y for y in str(x) if y == '@'])
    }
    rs = df[col].transform(counts)
    if 'graph' in kw:
        label = df['target'] == 1
        fig, axs = plt.subplots(ncols=1, nrows=len(rs.columns), figsize=(10, 10*len(rs.columns)), dpi=100)
        for i, x in enumerate(rs.columns):
            sns.set_color_codes("pastel")
            sns.distplot(rs.loc[~label][x], label='Not Disaster', ax=axs[i], color='b')
            sns.set_color_codes("muted")
            sns.distplot(rs.loc[label][x], label='Disaster', ax=axs[i], color='red')
            axs[i].legend()
            axs[i].set_title(f'{x} Target Distribution in Training Set')
        plt.savefig(kw['graph'])

    return rs
#test
# col = 'text'
# rs = word_counts(df,col)
# fig, axs = plt.subplots(ncols=1, figsize=(10, 10), dpi=100)
# sns.countplot(x=df['target'], hue=df['target'])
# axs.set_xticklabels(['Not Disaster', 'Disaster'])
# plt.savefig('target_count.png')

# count the labels.
def label_count(df,col,**kw):
    if 'graph' in kw:
        fig, axs = plt.subplots(ncols=1, figsize=(10, 10), dpi=100)
        sns.countplot(x=df['target'], hue=df['target'])
        axs.set_xticklabels(['Not Disaster', 'Disaster'])
        plt.savefig(kw['graph'])
    return len(df[df['target']==1])/len(df[df['target']!=1])

# find the most common word in tweets.
#generate phrase by number of words.
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(' ') if token != '' if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [' '.join(ngram) for ngram in ngrams]

#count the top 100 common phrases.
def common_phrase_count(df, n_gram = 1, N= 100, **kw):
    y_grams = {}
    n_grams = {}
    for tw in df.loc[df['target']==1]['text']:
        for word in generate_ngrams(tw, n_gram = n_gram):
            y_grams[word] = y_grams.get(word,0) + 1
    df_y_grams = pd.DataFrame(sorted(y_grams.items(), key=lambda x: x[1], reverse= True))

    
    for tw in df.loc[df['target']==0]['text']:
        for word in generate_ngrams(tw, n_gram = n_gram):
            n_grams[word] = n_grams.get(word,0) + 1
    df_n_grams = pd.DataFrame(sorted(n_grams.items(), key=lambda x: x[1], reverse= True))


    if 'graph' in kw:
        fig, axs = plt.subplots(ncols=2, figsize=(18, 100), dpi=100)
        sns.set_color_codes("muted")
        sns.barplot(y=df_y_grams[0].values[:N], x=df_y_grams[1].values[:N], ax = axs[0], color = 'red')
        sns.barplot(y=df_n_grams[0].values[:N], x=df_n_grams[1].values[:N], ax=axs[1], color='b')

        for i in range(2):
            axs[i].spines['right'].set_visible(False)
            axs[i].set_xlabel('')
            axs[i].set_ylabel('')

        axs[0].set_title(f'Top {N} most common {n_gram}-grams in Disaster Tweets')
        axs[1].set_title(f'Top {N} most common {n_gram}-grams in Non-disaster Tweets')

        plt.savefig(f'Top {N} most common {n_gram}-grams.png')
    return df_y_grams, df_n_grams

# test
# for i in range(1,4):
#     common_phrase_count(df, n_gram = i, N= 100, graph = 1)
# df['keyword'] = df['keyword'].fillna('na')
# df.to_csv('../twitter_disaster/data/train_cleaned.csv')

