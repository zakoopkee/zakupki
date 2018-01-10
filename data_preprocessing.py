import pandas as pd
from currency_converter import CurrencyConverter


def preprocess_data(data):
    data['SuppliersCount'].fillna(0, inplace=True)
    data['IsWinner'].fillna(0, inplace=True)
    data.Title.fillna("", inplace=True)
    data.ProcedureDisplayName.fillna("", inplace=True)
    data = data.drop(['Nds', 'LawCode', 'LawDisplayName'], axis=1)
    data['IsWinner'] = pd.to_numeric(data['IsWinner'])
    data['Amount'] = pd.to_numeric(
        data['Amount'].str.replace(',', '.'), errors='coerce'
    )
    data = convert_currencies(data)
    return data


def convert_currencies(data):
    c = CurrencyConverter()
    coeffs = {'%': 1, 'RUB': 1, 'XDR': 81.35}
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
