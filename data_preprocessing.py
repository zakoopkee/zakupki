import numpy as np
import pandas as pd
from currency_converter import CurrencyConverter


def preprocess_data(data):
    data['SuppliersCount'].fillna(0, inplace=True)
    data['IsWinner'].fillna(0, inplace=True)
    data.Title.fillna("", inplace=True)
    data.ProcedureDisplayName.fillna("", inplace=True)
    data = data.drop(['Nds', 'LawCode', 'LawDisplayName'], axis=1)
    data['IsWinner'] = pd.to_numeric(data['IsWinner'])
    data['StatusCode'] = pd.to_numeric(data['StatusCode'])
    data['Amount'] = pd.to_numeric(
        data['Amount'].str.replace(',', '.'), errors='coerce'
    )
    data = convert_currencies(data)
    data['ResultClass'] = [get_result_class(x, y) for x, y in zip(data.StatusCode, data.IsWinner)]

    return data


def get_result_class(statusCode, isWinner):
    if (statusCode == 3):
        return 0 # отмена
    if (statusCode == 2):
        if (isWinner == 0):
            return 1
        return 2
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
