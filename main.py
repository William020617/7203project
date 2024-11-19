import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import joblib
from sklearn.model_selection import cross_validate
from modeltrain import train_model
from preprocess import detect_outlier, preprocess_data
def main():
    np.random.seed(42)
    train_data, test_data = detect_outlier('DM_project_24.csv', 'test_data.csv')
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
        scoring = ['accuracy', 'f1_weighted']
        scores = cross_validate(model, X, y, cv=5, scoring=scoring)
        accuracy = np.mean(scores['test_accuracy'])
        f1 = np.mean(scores['test_f1_weighted'])

        print(
            f"{name} - Cross-validated Accuracy: {np.mean(scores['test_accuracy']):.3f}, F1 Score: {np.mean(scores['test_f1_weighted']):.3f}")

        # 进行预测
        test_predictions = model.predict(X_test)
        test_predictions = np.where(test_predictions >= 0.5, 1, 0).astype(int)
        # 保存每个模型的预测结果
        with open(f'result_{name}.infs4203', 'w') as f:
            for pred in test_predictions:
                f.write(f"{pred}\n")
            f.write(f"acc: {accuracy:.3f}\n")
            f.write(f"F1: {f1:.3f}\n")


if __name__ == "__main__":
    main()
