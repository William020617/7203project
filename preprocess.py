import pandas as pd
from sklearn.impute import SimpleImputer

def load_data(train_file, test_file):
    train_data = pd.read_csv('DM_project_24.csv')
    test_data = pd.read_csv('test_data.csv')
    return train_data, test_data

def preprocess_data(data):
    # 使用众数填充类别特征
    imputer = SimpleImputer(strategy='most_frequent')
    data.iloc[:, -3:] = imputer.fit_transform(data.iloc[:, -3:])
    # 其他特征用均值填充
    data.iloc[:, :-3] = data.iloc[:, :-3].fillna(data.mean())
    return data
