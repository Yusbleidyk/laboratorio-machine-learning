from imblearn.over_sampling import SMOTE

def separate_variables(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y

def balance_classes(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
