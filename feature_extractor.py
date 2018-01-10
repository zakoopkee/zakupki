import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import HashingVectorizer
columns_drop = ['id', 'Title', 'Uri', 'PublicationDateTimeUTC', 'ProcedureDisplayName', 'RubPrice', 'CurrencyCode',
                'StatusDisplayName', 'StatusCode', 'SuppliersCount', 'IsWinner']


def extract_features(data: pd.DataFrame):
    onehot = get_onehot_currency(data)
    ngrams_title = get_ngrams_title(data)
    ngrams_proceduredisplayname = get_ngrams_proceduredisplayname(data)
    return pd.concat([data, onehot, ngrams_title, ngrams_proceduredisplayname], axis=1).drop(columns_drop, axis=1)


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


def get_onehot_currency(data: pd.DataFrame):
    return get_onehot(data, 'CurrencyCode')


def get_onehot_proceduredisplayname(data: pd.DataFrame):
    return get_onehot(data, 'ProcedureDisplayName')


def get_ngrams_title(data: pd.DataFrame):
    return get_ngrams(data, 'Title', n_features=1000, ngram_range=(3, 3), analyzer='char_wb')


def get_ngrams_proceduredisplayname(data: pd.DataFrame):
    return get_ngrams(data, 'ProcedureDisplayName', n_features=300, ngram_range=(1, 2), analyzer='word')
