# 7203project

## Final choice
- **preprocess**: 
  - Data cleaning: Handling missing and outlier values.
  - Feature scaling: using standardized methods.
- **classification models**:
  - RandomForest
  - DecisionTree
  - KNN
  - NaiveBayes
- **Hyperparameters**:
  - RandomForest: `n_estimators=125`, `max_depth=None`
  - DecisionTree: `max_depth=1`, `min_samples_split=5`
  - KNN: `n_neighbors=11`, `weights=uniform`
  - NaiveBayes: Use default parameters

## Environment Description: 
- **Operating System**: Windows 11 
- **Programming Language**: Python 3.12
- **Additional Installed Packages**:
  - `numpy` 2.1.1
  - `pandas` 2.2.3
  - `scikit-learn` 1.5.2
  - `joblib` 1.4.2

## Reproduction Instructions
1. clone this base：
   ```bash
   git clone https://github.com/William020617/7203project.git
   cd 7203project
2. Create and activate a virtual environment
3. Install dependency packages
4. python main.py

## Additional Justifications:
In this project, cross validation method was used to evaluate the performance of the model to ensure its stability and generalization ability.
