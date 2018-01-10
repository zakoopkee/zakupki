import pandas as pd


def preprocess_bad_file(path='result.tsv'):
    data = pd.read_csv(path, sep='\t', error_bad_lines=False, dtype=str)
    data = data[data['Uri'].apply(lambda x: 'startswith' in dir(x) and
                x.startswith('https://zakupki.kontur.ru/'))]
    data = data.drop('id.1', axis=1)
    return data


def save_preprocessed_file(data, path='result.cropped.tsv'):
    data.to_csv(path, sep='\t', encoding='utf8', index=False)
