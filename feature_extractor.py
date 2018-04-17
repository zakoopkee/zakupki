import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import PCA

from data_preprocessing import log


columns_drop = ['id', 'Title', 'Uri', 'PublicationDateTimeUTC', 'ProcedureDisplayName', 'CurrencyCode',
                'StatusDisplayName', 'StatusCode', 'SuppliersCount', 'IsWinner']


def extract_features(data: pd.DataFrame, use_pca=False, title_features=1000, pca_features=100, verbose=False):
    onehot_currency = get_onehot(data, 'CurrencyCode')
    onehot_proceduredisplayname = get_onehot(data, 'ProcedureDisplayName')

    ngrams_title = get_ngrams(data, 'Title', n_features=title_features, ngram_range=(3, 3), analyzer='char_wb')
    # ngrams_title = get_ngrams(data, 'Title', n_features=1000, ngram_range=(1, 3), analyzer='word')
    ngrams_proceduredisplayname = get_ngrams(data, 'ProcedureDisplayName', n_features=300, ngram_range=(1, 2), analyzer='word')

    dfs = [data, onehot_currency, onehot_proceduredisplayname, ngrams_title, ngrams_proceduredisplayname]
    pca_ = None
    if use_pca:
        df, pca_ = pca(ngrams_title, pca_features, 'Title', verbose)
        dfs.append(df)

    return pd.concat(dfs, axis=1).drop(columns_drop, axis=1), pca_


def pca(data, n_features, title, verbose):
    if verbose:
        log('start pca')
    pca = PCA(n_components=n_features)
    pca.fit(data)
    tr = pca.transform(data)

    if verbose:
        log('end pca')

    s = 0
    for i, e in enumerate(sorted(pca.explained_variance_ratio_, key=lambda x: -x)):
        s += e
        if s >= 0.9:
            print(i + 1)
            break
    print(s)
    return pd.DataFrame(tr, columns=list(map(lambda ind: f'pca_{title}_{ind}', range(n_features)))), pca
    #print(np.corrcoef(list(map(lambda x: x[0], tr)), djia['^DJI']))
    #print(list(data.columns.values)[list(pca.components_[0]).index(sorted(pca.components_[0], key=lambda x: -x)[0])])


def get_ngrams(data, column, n_features, ngram_range, analyzer):
    vectorizer = HashingVectorizer(input='content', encoding='utf-8', decode_error='strict', lowercase=True,
                                   token_pattern=r'\b\w+\b', ngram_range=ngram_range, analyzer=analyzer,
                                   n_features=n_features, binary=True, norm=None)
    analyze = vectorizer.fit_transform(data[column])
    feats = pd.SparseDataFrame(analyze).fillna(0)
    return pd.DataFrame(feats).add_prefix(f'Ngrams_{column}_')


def get_onehot(data, column):
    values = data[column].astype(str).values
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    labels = data[column].sort_values().unique()
    return pd.DataFrame(onehot_encoded, columns=labels).add_prefix(f'OneHot_{column}_')
