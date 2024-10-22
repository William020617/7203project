from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import numpy as np
import pandas as pd
from preprocess import load_data, preprocess_data


def train_model(X_train, y_train):
    models = {}

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_param_grid = {
        'n_estimators': [100, 150, 200],
        'max_depth': [1, 2, 3],
        'min_samples_split': [50, 100, 150],
        'min_samples_leaf': [50, 100, 150],
    }
    rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5)
    rf_grid_search.fit(X_train, y_train)
    models['RandomForest'] = rf_grid_search.best_estimator_

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_param_grid = {
        'max_depth': [1, 2, 3],
        'min_samples_split': [100, 200, 300],
        'min_samples_leaf': [50, 100, 150],
    }
    dt_grid_search = GridSearchCV(dt_model, dt_param_grid, cv=5)
    dt_grid_search.fit(X_train, y_train)
    models['DecisionTree'] = dt_grid_search.best_estimator_

    # K-Nearest Neighbor
    knn_model = KNeighborsClassifier()
    knn_param_grid = {
        'n_neighbors': [10, 11, 12],
        'weights': ['uniform', 'distance'],
    }
    knn_grid_search = GridSearchCV(knn_model, knn_param_grid, cv=5)
    knn_grid_search.fit(X_train, y_train)
    models['KNN'] = knn_grid_search.best_estimator_

    # Naive Bayes
    nb_model = GaussianNB()
    nb_grid_search = GridSearchCV(nb_model, {}, cv=5)
    nb_grid_search.fit(X_train, y_train)
    models['NaiveBayes'] = nb_grid_search.best_estimator_

    # 保存最佳模型
    for name, model in models.items():
        joblib.dump(model, f'best_model_{name}.pkl')
        print(f'{name} model saved as best_model_{name}.pkl')

    # 打印最佳参数
    for name, model in models.items():
        if name == 'RandomForest':
            print(f"Best parameters - RandomForest: n_estimators={rf_grid_search.best_params_['n_estimators']}, max_depth={rf_grid_search.best_params_['max_depth']}, min_samples_split = {rf_grid_search.best_params_['min_samples_split']}, min_samples_leaf = {rf_grid_search.best_params_['min_samples_leaf']}")
        elif name == 'DecisionTree':
            print(f"Best parameters - DecisionTree: max_depth={dt_grid_search.best_params_['max_depth']}, min_samples_split={dt_grid_search.best_params_['min_samples_split']}, min_samples_leaf = {dt_grid_search.best_params_['min_samples_leaf']}")
        elif name == 'KNN':
            print(f"Best parameters - KNN: n_neighbors={knn_grid_search.best_params_['n_neighbors']}, weights={knn_grid_search.best_params_['weights']}")
        elif name == 'NaiveBayes':
            print(f"Best parameters - NaiveBayes: Use default parameters")
    return models  # 返回所有模型

