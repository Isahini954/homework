# 导入所需的库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# 读取数据集
file_trained = 'used_car_train_20200313.csv'
train_df = pd.read_csv(file_trained, sep='\\s+')

# 选择特征与目标变量
selected_features = ['regDate', 'model', 'brand', 'bodyType', 'fuelType', 'kilometer', 'power']
features = train_df[selected_features]  # 特征数据
target = train_df['price']  # 目标变量

# 替换'-'为NaN，以便缺失值后续的处理
features.replace('-', np.nan, inplace=True)

# 以均值策略对缺失值填补
imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(features)

# 分割数据集之训练集、测试集，二者九十分成
f_trained, f_tested, t_trained, t_tested = train_test_split(features, target, test_size=0.1, random_state=42)

# 标准化处理特征
scaler = StandardScaler()
f_train_scaled = scaler.fit_transform(f_trained)  # 训练集特征缩放
f_test_scaled = scaler.transform(f_tested)  # 测试集特征缩放

# 定义回归模型三种
models = {
    '线性回归': LinearRegression(),
    '决策树': DecisionTreeRegressor(random_state=42),
    '随机森林': RandomForestRegressor(random_state=42)
}

# 训练并且评估各模型性能
result = []
best_model = None
best_mae = float('inf')
for name, model in models.items():
    model.fit(f_train_scaled, t_trained)  # 训练模型
    t_pred = model.predict(f_test_scaled)  # 预测测试集
    mae = mean_absolute_error(t_tested, t_pred)  # 计算MAE
    r2 = r2_score(t_tested, t_pred)  # 计算R²
    result.append({'模型': name, 'MAE': mae, 'R²': r2})  # 记录结果
    # 更新最佳模型
    if mae < best_mae:
        best_mae = mae
        best_model = model

# 评估结果
result_df = pd.DataFrame(result)
print(result_df)

# 读取测试集
file_trained = 'used_car_testB_20200421.csv'
test_df = pd.read_csv(file_trained, sep='\s+')

# 测试集执行相同特征选择与预处理
f_test_final = test_df[selected_features]
f_test_final.replace('-', np.nan, inplace=True)
f_test_final = imputer.transform(f_test_final)
f_test_final_scaled = scaler.transform(f_test_final)

# 最佳模型进行预测
t_test_pred = best_model.predict(f_test_final_scaled)

# 创建包含预测结果的DataFrame，生成提交文件
sub = pd.DataFrame({'SaleID': test_df['SaleID'], 'price': t_test_pred})
sub_file_path = 'used_car_sample_submit.csv'
sub.to_csv(sub_file_path, index=False)

# 提交文件生成信息
print("提交文件", sub_file_path, "已生成。")
