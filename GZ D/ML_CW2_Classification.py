import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTENC

# Load  Train dataset
data = pd.read_excel("../Dataset/TrainDataset2024.xls")
# Step 1: 缺失值处理
# replace 999 -> NaN
data.replace(999, np.nan, inplace=True)
# 对于分类变量，用众数填充
categorical_features = ['ER', 'PgR', 'HER2', 'TrippleNegative', 'ChemoGrade', 'Proliferation', 'HistologyType',
                        'LNStatus', 'TumourStage', 'Gene']
for col in categorical_features:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    data[col] = imputer_cat.fit_transform(data[[col]])

# 对于数值变量，用中位数填充
# numerical_features = [col for col in data.columns if
#                       col not in categorical_features + ['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)']]
numerical_features = [
    col for col in data.columns
    if data[col].dtype in ['float64', 'int64'] and col not in categorical_features + ['ID', 'pCR (outcome)',
                                                                                      'RelapseFreeSurvival (outcome)']
]

imputer_num = SimpleImputer(strategy='median')
data[numerical_features] = imputer_num.fit_transform(data[numerical_features])
data = data.dropna(subset=['pCR (outcome)'])  # 删除含缺失值的行

# Step 2: 异常值检测和处理
# 使用箱线图法检测异常值（示例以 Tumour Proliferation 为例）
for col in numerical_features:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    # 替换异常值为边界值
    data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
    data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])

# Step 3: 数据归一化
# 选择需要归一化的列（通常是数值型特征）
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Step 4: 数据划分
# 提取特征和目标变量
X = data.drop(columns=['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)'])  # 去除 ID 和目标列
X_pCr = data[data['pCR (outcome)'].isin([0, 1])]
X_pCr = X_pCr.drop(columns=['ID', 'pCR (outcome)', 'RelapseFreeSurvival (outcome)'])

# 分类任务目标
# remove when pCR is absent and pCR is the target
y_pcr = data[data['pCR (outcome)'].isin([0, 1])]
y_pcr = y_pcr['pCR (outcome)']
y_rfs = data['RelapseFreeSurvival (outcome)']  # 回归任务目标

# print(data['pCR (outcome)'].isnull().sum())  # 检查分类任务目标变量的缺失值 = 0
# print(data['RelapseFreeSurvival (outcome)'].isnull().sum())  # 检查回归任务目标变量的缺失值 = 0

# 按 8:2 分层划分训练集和验证集
X_train_pcr, X_val_pcr, y_train_pcr, y_val_pcr = train_test_split(
    X_pCr, y_pcr, test_size=0.2, random_state=42, stratify=y_pcr
)
X_train_rfs, X_val_rfs, y_train_rfs, y_val_rfs = train_test_split(
    X, y_rfs, test_size=0.2, random_state=42
)

# SMOTE
smote = SMOTENC(categorical_features=categorical_features, random_state=42)
# smote = BorderlineSMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pcr, y_train_pcr)
# y_train_resampled = y_train_resampled.astype(int)
# print("Class distribution after SMOTE:", np.bincount(y_train_resampled))  # [249, 249]

# Step 5: 检查处理结果
# print("训练集和验证集的大小：")
# print("pCR 分类训练集:", X_train_resampled.shape, "验证集:", X_val_pcr.shape)  # pCR 分类训练集: (498, 118) 验证集: (79, 118)
# print("RFS 回归训练集:", X_train_rfs.shape, "验证集:", X_val_rfs.shape)  # RFS 回归训练集: (316, 118) 验证集: (79, 118)

# Step 6: Model Selection: CNN
import torch
from torch.utils.data import DataLoader, TensorDataset

# 转换数据为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_pcr.values, dtype=torch.float32).unsqueeze(1)  # 增加通道维度
y_train_tensor = torch.tensor(y_train_pcr.values, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val_pcr.values, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val_pcr.values, dtype=torch.float32).unsqueeze(1)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
import torch.nn as nn


# 定义 CNN 模型
class CNNClassifier(nn.Module):
    def __init__(self, input_dim):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(input_dim * 64, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # 二分类输出

    def forward(self, x):
        x = self.relu(self.conv1(x))  # 第一层卷积
        x = self.relu(self.conv2(x))  # 第二层卷积
        x = x.view(x.size(0), -1)  # 展平
        x = self.relu(self.fc1(x))  # 全连接层
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))  # 二分类输出
        return x


import torch.optim as optim

# 初始化模型
input_dim = X_train_pcr.shape[1]
model = CNNClassifier(input_dim)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)  # 前向传播
        loss = criterion(outputs, y_batch)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()
        train_loss += loss.item()

    # 验证模型
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, "
          f"Train Loss: {train_loss / len(train_loader):.4f}, "
          f"Validation Loss: {val_loss / len(val_loader):.4f}")

from sklearn.metrics import classification_report, roc_auc_score

# 预测并评估
model.eval()
with torch.no_grad():
    y_pred = model(X_val_tensor).numpy().ravel()
    y_pred_class = (y_pred > 0.5).astype(int)  # 阈值为 0.5

print(classification_report(y_val_pcr.values, y_pred_class))

print("ROC-AUC:", roc_auc_score(y_val_pcr.values, y_pred))
