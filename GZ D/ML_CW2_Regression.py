import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load  Train dataset
df = pd.read_excel("../Dataset/TrainDataset2024.xls")

# Step 1: missing values
# replace 999 with NaN
df.replace(999, np.nan, inplace=True)

# Imputation of Gene
df['Gene'] = df['Gene'].fillna(-1)

# Drop pCR
df.drop(columns="pCR (outcome)", inplace=True)

# Categorical imputation
categorical_features = ['ER', 'PgR', 'HER2', 'TrippleNegative', 'ChemoGrade', 'Proliferation', 'HistologyType',
                        'LNStatus', 'TumourStage', 'Gene']
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_features] = imputer_cat.fit_transform(df[categorical_features])

# Numerical Imputation
numerical_features = [col for col in df.columns if
                      col not in categorical_features + ['ID', 'RelapseFreeSurvival (outcome)']]
imputer_num = KNNImputer(n_neighbors=5)
df[numerical_features] = imputer_num.fit_transform(df[numerical_features])

# Step 2: 异常值检测和处理
# 使用箱线图法检测异常值（示例以 Tumour Proliferation 为例）
for col in numerical_features:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # 替换异常值为边界值
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# Step 3: Data Standardization
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Step 4: 数据划分
# 提取特征和目标变量
X = df.drop(columns=['ID', 'RelapseFreeSurvival (outcome)'])
y = df['RelapseFreeSurvival (outcome)']

# PCA
pca = PCA(n_components=0.95)  # 保留 95% 的方差
X = pd.DataFrame(pca.fit_transform(X))

# 按 8:2 分层划分训练集和验证集
X_train_RFS, X_val_RFS, y_train_RFS, y_val_RFS = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Plot target variable distribution
plt.hist(y_train_RFS, bins=30)
plt.title("Target Variable Distribution")
plt.show()

# y_train_RFS_log = np.log1p(y_train_RFS)
# y_val_RFS_log = np.log1p(y_val_RFS)
#
# plt.hist(y_train_RFS_log, bins=30)
# plt.title("Target Variable Distribution")
# plt.show()

# Step 6: Model Selection
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.model_selection import GridSearchCV
#
# param_grid = {  # 网格搜索示例
#     'n_estimators': [100, 200, 300],
#     'learning_rate': [0.01, 0.1, 0.2],
#     'max_depth': [3, 6, 9],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0],
#     'gamma': [0, 0.1, 0.5],
# }
#
# xgb_model = XGBRegressor(random_state=42)
# grid_search = GridSearchCV(xgb_model, param_grid, scoring='neg_mean_squared_error', cv=3, verbose=2, n_jobs=-1)
# grid_search.fit(X_train_RFS, y_train_RFS)
# xgb_regressor = grid_search.best_estimator_
#
# # 模型训练和预测
# xgb_regressor.fit(X_train_RFS, y_train_RFS)
# y_pred_RFS = xgb_regressor.predict(X_val_RFS)
#
# # Step 7: Evaluation
# mse = mean_squared_error(y_val_RFS, y_pred_RFS)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_val_RFS, y_pred_RFS)
# r2 = r2_score(y_val_RFS, y_pred_RFS)
#
# print(f"Mean Squared Error (MSE): {mse:.4f}")
# print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# print(f"Mean Absolute Error (MAE): {mae:.4f}")
# print(f"R² Score: {r2:.4f}")
#
# plt.figure(figsize=(8, 6))
# plt.scatter(y_val_RFS, y_pred_RFS, alpha=0.7)
# plt.plot([min(y_val_RFS), max(y_val_RFS)], [min(y_val_RFS), max(y_val_RFS)], color='red', linestyle='--')
# plt.xlabel("Actual Values")
# plt.ylabel("Predicted Values")
# plt.title("Actual vs Predicted Values (XGBoost)")
# plt.show()

# Step 6: Model Selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

# # 参数网格（随机森林）
# param_grid = {
#     'n_estimators': [100, 200, 300],  # 森林中树的数量
#     'max_depth': [None, 10, 20],  # 每棵树的最大深度
#     'min_samples_split': [2, 5, 10],  # 节点分裂所需的最小样本数
#     'min_samples_leaf': [1, 2, 4],  # 叶子节点的最小样本数
#     'bootstrap': [True, False],  # 是否进行有放回采样
# }

# # 初始化 RandomForestRegressor
# rf_model = RandomForestRegressor(random_state=42)
#
# # 使用网格搜索调参
# grid_search = GridSearchCV(
#     rf_model,
#     param_grid,
#     scoring='neg_mean_squared_error',  # 使用负均方误差作为评分标准
#     cv=3,  # 3 折交叉验证
#     verbose=2,
#     n_jobs=-1  # 并行运行
# )
#
# # 训练模型并搜索最佳参数
# grid_search.fit(X_train_RFS, y_train_RFS)
#
# # 获取最佳模型
# rf_regressor = grid_search.best_estimator_
# print(f"Best Parameters: {grid_search.best_params_}")
#
# # 模型训练和预测
# rf_regressor.fit(X_train_RFS, y_train_RFS)

rf_regressor = RandomForestRegressor(
    bootstrap=True,
    ccp_alpha=0.0,
    criterion='squared_error',
    max_depth=1000,
    max_features='sqrt',
    max_leaf_nodes=None,
    max_samples=None,
    min_impurity_decrease=0.0,
    min_samples_leaf=1,
    min_samples_split=10,
    min_weight_fraction_leaf=0.0,
    n_estimators=100,
    n_jobs=1,
    oob_score=False,
    random_state=None,
    verbose=0,
    warm_start=False,
)

# train
rf_regressor.fit(X_train_RFS, y_train_RFS)
# y_pred_RFS = rf_regressor.predict(X_val_RFS)

# 计算特征重要性（基于特征置换）
perm_importance = permutation_importance(
    rf_regressor,
    X_val_RFS,
    y_val_RFS,
    scoring='r2',  # 使用 R² 作为评估指标
    n_repeats=10,
    random_state=42
)

# 打印特征的重要性
feature_importance = perm_importance.importances_mean
feature_names = X_train_RFS.columns if hasattr(X_train_RFS, 'columns') else range(X_train_RFS.shape[1])
# for name, importance in zip(feature_names, feature_importance):
#     print(f"Feature: {name}, Importance: {importance:.4f}")

# 可视化特征重要性
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Feature Importance (Permutation)')
plt.title('Permutation Feature Importance')
plt.show()

low_importance_features = [name for name, importance in zip(feature_names, feature_importance) if importance < 0.01]
X_train_RFS_reduced = X_train_RFS.drop(columns=low_importance_features)
X_val_RFS_reduced = X_val_RFS.drop(columns=low_importance_features)

# re-train
rf_regressor.fit(X_train_RFS_reduced, y_train_RFS)
y_pred_RFS = rf_regressor.predict(X_val_RFS_reduced)

# Step 7: Evaluation
mse = mean_squared_error(y_val_RFS, y_pred_RFS)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_val_RFS, y_pred_RFS)
r2 = r2_score(y_val_RFS, y_pred_RFS)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# 可视化：实际值 vs 预测值
plt.figure(figsize=(8, 6))
plt.scatter(y_val_RFS, y_pred_RFS, alpha=0.7)
plt.plot([min(y_val_RFS), max(y_val_RFS)], [min(y_val_RFS), max(y_val_RFS)], color='red', linestyle='--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values (Random Forest)")
plt.show()

# Step 8: Predict from Final Testset
# Loading Final Test Dataset
test_data = pd.read_excel("../dataset/FinalTestDataset2024.xls")

# Extract the 'ID' column
test_ids = test_data['ID']

# Data Preprocessing
test_data.replace(999, np.nan, inplace=True)
test_data['Gene'] = test_data['Gene'].fillna(-1)
test_data.drop(columns=['ID'], inplace=True)
test_data[categorical_features] = imputer_cat.transform(test_data[categorical_features])
test_data[numerical_features] = imputer_num.transform(test_data[numerical_features])

# Outliers
for col in numerical_features:
    q1 = test_data[col].quantile(0.20)
    q3 = test_data[col].quantile(0.80)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    test_data[col] = np.where(test_data[col] < lower_bound, lower_bound, test_data[col])
    test_data[col] = np.where(test_data[col] > upper_bound, upper_bound, test_data[col])

# Data Standardization
test_data[numerical_features] = scaler.transform(test_data[numerical_features])

# Applying PCA to reduce dimensions
test_data = pca.transform(test_data)
test_data = pd.DataFrame(test_data)

# 删除低重要性特征
test_data_reduced = test_data.drop(columns=low_importance_features, errors='ignore')

# 确保测试数据特征顺序与训练数据一致
test_data_reduced = test_data_reduced[X_train_RFS_reduced.columns]

# Predict
predictions = pd.DataFrame({"ID": test_ids})

RFS_pred = rf_regressor.predict(test_data_reduced)
predictions['RelapseFreeSurvival (outcome)'] = RFS_pred

# Save predictions to a CSV file
predictions.to_csv("RFSPrediction.csv", index=False)
print("\nRFSPrediction.csv saved!\n")
