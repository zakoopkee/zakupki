from sklearn.ensemble import RandomForestClassifier as RF


def train(train_data, result_column):
    rf = RF(n_estimators=100, max_features='auto', n_jobs=-1)
    return rf.fit(train_data, result_column)


def predict(rf, test_data):
    return rf.predict(test_data)
