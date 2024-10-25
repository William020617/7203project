import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer

def detect_outlier(train_file, test_file):
    train_data = pd.read_csv('DM_project_24.csv')
    test_data = pd.read_csv('test_data.csv')
    # 检查数据中是否有缺失值，并进行填充（前面103列用均值，后三列用众数）
    train_data.iloc[:, :103] = train_data.iloc[:, :103].fillna(train_data.iloc[:, :103].mean())
    imputer = SimpleImputer(strategy='most_frequent')
    train_data.iloc[:, -3:] = imputer.fit_transform(train_data.iloc[:, -3:])
    test_data.iloc[:, :103] = test_data.iloc[:, :103].fillna(test_data.iloc[:, :103].mean())
    test_data.iloc[:, -3:] = imputer.fit_transform(test_data.iloc[:, -3:])

    # 使用K近邻算法计算每个数据点的距离
    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(train_data)

    # 找出每个数据点的最近邻距离
    neighbors_distance, _ = neigh.kneighbors(train_data)

    # 将最近邻距离转换为一维数组，以便计算百分位数
    distances = neighbors_distance[:, -1]

    # 计算每个数据点到第5个最近邻的距离的百分位数
    threshold = np.percentile(distances, 95)

    # 标记那些距离超过阈值的点为异常值
    outliers = distances > threshold

    # 删除异常值
    train_data_cleaned = train_data[~outliers]

    # 输出处理后的数据
    print(f"原始训练数据大小: {train_data.shape}")
    print(f"去除异常值后的数据大小: {train_data_cleaned.shape}")

    return train_data_cleaned, test_data


def preprocess_data(data):
    return data




