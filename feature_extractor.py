import pandas as pd
import pymorphy2
import string
import re

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import HashingVectorizer
columns_drop = ['id', 'Title', 'Uri', 'PublicationDateTimeUTC', 'ProcedureDisplayName', 'CurrencyCode',
                'StatusDisplayName', 'StatusCode', 'SuppliersCount', 'IsWinner']


def extract_features(data: pd.DataFrame):
    onehot_currency = get_onehot(data, 'CurrencyCode')
    onehot_proceduredisplayname = get_onehot(data, 'ProcedureDisplayName')

    ngrams_title = get_ngrams(data, 'Title', n_features=1000, ngram_range=(3, 3), analyzer='char_wb')
    ngrams_proceduredisplayname = get_ngrams(data, 'ProcedureDisplayName', n_features=300, ngram_range=(1, 2), analyzer='word')

    return pd.concat([data, onehot_currency, onehot_proceduredisplayname, ngrams_title, ngrams_proceduredisplayname],
                     axis=1).drop(columns_drop, axis=1)


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


morph = pymorphy2.MorphAnalyzer()
word_regex = re.compile(r'\w+')
remove_punct_map = dict.fromkeys(map(ord, string.punctuation))


def normalize_word(word):
    return morph.parse(word)[0].normal_form


def normalize_string(s: str):
    return ' '.join(map(normalize_word, word_regex.findall(s.translate(remove_punct_map))))


def normalize(data, columns):
    for col in columns:
        data[col] = data[col].map(normalize_string)