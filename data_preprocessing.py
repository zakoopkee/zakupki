import pandas as pd
from currency_converter import CurrencyConverter


def preprocess_data(data):
    data['SuppliersCount'].fillna(0, inplace=True)
    data['IsWinner'].fillna(0, inplace=True)
    data.Title.fillna("", inplace=True)
    data = data.drop(['Nds', 'LawCode', 'LawDisplayName'], axis=1)
    data['IsWinner'] = pd.to_numeric(data['IsWinner'])
    data['Amount'] = pd.to_numeric(
        data['Amount'].str.replace(',', '.'), errors='coerce'
    )
    data = convert_currencies(data)
    return data


def convert_currencies(data):
    c = CurrencyConverter()
    c.convert(100, 'USD', 'RUB')

    for code in data['CurrencyCode'].unique():
        if code == '%' or 'XDR':
            continue
        data.loc[data['CurrencyCode'] == code, 'Amount'] = (
            data.loc[data['CurrencyCode'] == code, 'Amount'].apply(
                lambda x: c.convert(x, code, 'RUB')
            )
        )
        data.loc[data['CurrencyCode'] == code, 'CurrencyCode'] = 'RUB'
    return data
