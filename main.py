import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import joblib
from sklearn.model_selection import cross_val_score
from modeltrain import train_model
from preprocess import load_data, preprocess_data
def main():
    np.random.seed(42)
    train_data, test_data = load_data('DM_project_24.csv', 'test_data.csv')
    X = preprocess_data(train_data.iloc[:, :-1])
    y = train_data.iloc[:, -1]
    train_model(X, y)
    models = {}
    # 加载所有模型
    try:
        models['RandomForest'] = joblib.load('best_model_RandomForest.pkl')
        models['DecisionTree'] = joblib.load('best_model_DecisionTree.pkl')
        models['KNN'] = joblib.load('best_model_KNN.pkl')
        models['NaiveBayes'] = joblib.load('best_model_NaiveBayes.pkl')
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        return

    # 预测测试数据
    X_test = preprocess_data(test_data)

    for name, model in models.items():
        # 交叉验证评估
        accuracy_cv = cross_val_score(model, X, y, cv=5)
        accuracy = np.mean(accuracy_cv)
        f1 = f1_score(y, model.predict(X), average='weighted')

        print(f"{name} - Accuracy (CV): {accuracy:.3f}, F1 Score (CV): {f1:.3f}")

        # 进行预测
        test_predictions = model.predict(X_test)

        # 保存每个模型的预测结果
        result_report = np.zeros((818, 2))
        result_report[:817, 0] = test_predictions  # 预测结果
        result_report[817, 0] = round(accuracy, 3)  # 准确率
        result_report[817, 1] = round(f1, 3)  # F1分数
        np.savetxt(f'result_{name}.infs4203', result_report, delimiter=',', fmt='%.3f')

if __name__ == "__main__":
    main()
