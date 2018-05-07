import numpy as np
import pandas as pd
import pymorphy2
import string
import re
import datetime

from currency_converter import CurrencyConverter


def log(s):
    print(f'[{datetime.datetime.now()}]: {s}')


def preprocess_data(data, limit=None, normalize_text=False, save_file=None, verbose=False):
    data['SuppliersCount'].fillna(0, inplace=True)
    data['IsWinner'].fillna(0, inplace=True)
    data['IsWinner'] = pd.to_numeric(data['IsWinner'])
    data['StatusCode'] = pd.to_numeric(data['StatusCode'])
    data['ResultClass'] = [get_result_class(x, y) for x, y in zip(data.StatusCode, data.IsWinner)]
    data = data[data.ResultClass != 3]
    if limit is not None:
        data = get_balanced_sample(data, limit)
        if verbose:
            sample_sizes = data.groupby('ResultClass').size()
            log(f'Unsuccess samples count: {sample_sizes[0]}')
            log(f'Success samples count: {sample_sizes[1]}')
            log(f'Cancel samples count: {sample_sizes[2]}')

    data.Title.fillna("", inplace=True)
    data.ProcedureDisplayName.fillna("", inplace=True)
    for e in ['Nds', 'LawCode', 'LawDisplayName']:
        if e in data.columns:
            data = data.drop(e, axis=1)
    #data = data.drop(['Nds', 'LawCode', 'LawDisplayName'], axis=1)
    data['Amount'] = pd.to_numeric(
        data['Amount'].str.replace(',', '.'), errors='coerce'
    )
    data = convert_currencies(data)
    data['RubPrice'] = pd.to_numeric(data['RubPrice'])
    data['SuppliersCount'] = pd.to_numeric(data['SuppliersCount'])
    data['Ogrn1'] = [int(x[3:5]) for x in data['Ogrn']]
    data.Ogrn1.fillna(0, inplace=True)

    if normalize_text:
        normalize(data, 'Title', verbose)
        normalize(data, 'ProcedureDisplayName', verbose)

    data.reset_index(drop=True, inplace=True)

    if save_file is not None:
        if verbose:
            log(f'Saving {save_file}')
        data.to_csv(save_file, encoding='utf-8', sep='\t', index=False)

    return data
    
def get_balanced_sample(data, count):
    cancel_count = min(count // 3, data[data.ResultClass == 2].shape[0])
    success_count = min((count - cancel_count) // 2, data[data.ResultClass == 1].shape[0])
    unsuccess_count = count - cancel_count - success_count
    balanced_sample = pd.concat([data[data.ResultClass == 2][:cancel_count],
                                 data[data.ResultClass == 1][:success_count],
                                 data[data.ResultClass == 0][:unsuccess_count]])
    return balanced_sample

def get_result_class(status_code, is_winner):
    if status_code == 3:
        return 2
    if status_code == 2:
        return is_winner
    return 3


def convert_currencies(data):
    c = CurrencyConverter()
    coeffs = {'%': 1, 'RUB': 1, 'XDR': 81.35, 'MVR': 3.68, 'STD': 0.00279, 'KHR': 0.01416, 'OMR': 148.35, 'SVC': 6.5189,
              'GMD': 1.1968, 'PНP': 1.128, 'BYR': 0.002828, 'KZT': 0.1727, 'WST': 22.75, 'PEN': 17.69, 'MUR': 1.69,
              'SDG': 8.1346, 'KGS': 0.8269, 'SRD': 7.66, 'TOP': 25.866, 'TZS': 0.02542, 'TJS': 6.4612, 'AMD': 0.1176,
              'BWP': 5.7835, 'UGX': 0.0156, 'MNT': 0.02345, np.nan: 1}
    for code in data['CurrencyCode'].unique():
        if code in coeffs:
            data.loc[data['CurrencyCode'] == code, 'RubPrice'] = data.loc[data['CurrencyCode'] == code, 'Amount'] * coeffs[code]
        else:
            data.loc[data['CurrencyCode'] == code, 'RubPrice'] = (
                data.loc[data['CurrencyCode'] == code, 'Amount'].apply(
                    lambda x: c.convert(x, code, 'RUB')
                )
            )
    return data


morph = pymorphy2.MorphAnalyzer()
word_regex = re.compile(r'\w+')
remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
normalization_cache = {}
cache_hits = 0


def normalize_word(word):
    return morph.parse(word)[0].normal_form


def normalize_string(s: str):
    if s in normalization_cache:
        global cache_hits
        cache_hits += 1
        return normalization_cache[s]
    normalized = ' '.join(map(normalize_word, word_regex.findall(s.translate(remove_punct_map))))
    normalization_cache[s] = normalized
    return normalized


def normalize(data, column, verbose):
    if verbose:
        log(f'Start normalizing {column}')
    data[column] = data[column].map(normalize_string)
    if verbose:
        global cache_hits
        log(f'End normalizing {column}, cache hits: {cache_hits}')
        cache_hits = 0
