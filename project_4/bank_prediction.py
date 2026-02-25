import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("步骤1: 数据加载与探索性分析（EDA）")

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print("训练集形状:", train_df.shape)
print("测试集形状:", test_df.shape)
print("\n目标变量分布:")
print(train_df['subscribe'].value_counts())
print("\n目标变量比例:")
print(train_df['subscribe'].value_counts(normalize=True))

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].hist(train_df['age'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('年龄分布')
axes[0, 0].set_xlabel('年龄')

axes[0, 1].hist(train_df['duration'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('通话时长分布')
axes[0, 1].set_xlabel('通话时长(秒)')

axes[1, 0].hist(train_df['campaign'], bins=30, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('营销活动次数分布')
axes[1, 0].set_xlabel('营销活动次数')

subscribe_counts = train_df['subscribe'].value_counts()
axes[1, 1].bar(subscribe_counts.index, subscribe_counts.values, color=['skyblue', 'orange'])
axes[1, 1].set_title('目标变量分布')
axes[1, 1].set_xlabel('是否认购')

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
print("\n特征分布图已保存: feature_distributions.png")
plt.close()

numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in ['id']]
correlation_matrix = train_df[numeric_cols].corr()

plt.figure(figsize=(14, 12))
im = plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im, shrink=0.8)
plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha='right')
plt.yticks(range(len(numeric_cols)), numeric_cols)

for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        text = plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                        ha="center", va="center", color="black" if abs(correlation_matrix.iloc[i, j]) < 0.5 else "white")

plt.title('数值特征相关性热力图', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n相关性热力图已保存: correlation_heatmap.png")
plt.close()

print("\n")
print("步骤2: 特征工程与数据预处理")

train_df['subscribe'] = train_df['subscribe'].map({'yes': 1, 'no': 0})

train_df['duration_log'] = np.log1p(train_df['duration'])
test_df['duration_log'] = np.log1p(test_df['duration'])

categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                    'contact', 'month', 'day_of_week', 'poutcome']

for col in categorical_cols:
    train_df[col] = train_df[col].fillna('unknown')
    test_df[col] = test_df[col].fillna('unknown')

train_df_processed = pd.get_dummies(train_df, columns=categorical_cols, drop_first=True)
test_df_processed = pd.get_dummies(test_df, columns=categorical_cols, drop_first=True)

for col in train_df_processed.columns:
    if col not in test_df_processed.columns and col != 'subscribe':
        test_df_processed[col] = 0

for col in test_df_processed.columns:
    if col not in train_df_processed.columns:
        train_df_processed[col] = 0

test_df_processed = test_df_processed[train_df_processed.columns.drop('subscribe')]

X = train_df_processed.drop(['id', 'subscribe'], axis=1)
y = train_df_processed['subscribe']
test_ids = test_df_processed['id']
X_test = test_df_processed.drop(['id'], axis=1)

numeric_features = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                    'emp_var_rate', 'cons_price_index', 'cons_conf_index', 
                    'lending_rate3m', 'nr_employed', 'duration_log']

scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

print("特征工程完成")
print("训练特征数量:", X.shape[1])
print("测试特征数量:", X_test.shape[1])

print("\n")
print("步骤3: 模型训练与基准评估")

models = {
    '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
    '随机森林': RandomForestClassifier(random_state=42, n_estimators=200, max_depth=20, n_jobs=1),
    'XGBoost': XGBClassifier(random_state=42, n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric='logloss', n_jobs=1)
}

results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=1)
    results[name] = {
        'mean_accuracy': cv_scores.mean(),
        'std_accuracy': cv_scores.std(),
        'cv_scores': cv_scores
    }
    print(f"\n{name}:")
    print(f"  平均Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  各折得分: {cv_scores}")

best_model_name = max(results, key=lambda x: results[x]['mean_accuracy'])
print(f"\n最佳基准模型: {best_model_name}")
print(f"最佳Accuracy: {results[best_model_name]['mean_accuracy']:.4f}")

print("\n")
print("步骤4: 特征选择")

selector = SelectKBest(f_classif, k=30)
X_selected = selector.fit_transform(X, y)
X_test_selected = selector.transform(X_test)

selected_features = X.columns[selector.get_support()].tolist()
print(f"选择的特征数量: {len(selected_features)}")
print(f"选择的特征: {selected_features}")

for name, model in models.items():
    cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring='accuracy', n_jobs=1)
    print(f"\n{name} (特征选择后):")
    print(f"  平均Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

best_model_name_selected = max(results, key=lambda x: results[x]['mean_accuracy'])
best_model = models[best_model_name_selected]
best_score = results[best_model_name_selected]['mean_accuracy']

print(f"\n最终选择模型: {best_model_name_selected}")
print(f"最终交叉验证Accuracy: {best_score:.4f}")

print("\n")
print("步骤5: 预测与提交")

best_model.fit(X_selected, y)

y_pred_proba = best_model.predict_proba(X_test_selected)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

y_pred_labels = np.where(y_pred == 1, 'yes', 'no')

submission = pd.DataFrame({
    'id': test_ids,
    'subscribe': y_pred_labels
})

submission.to_csv('submission_demo.csv', index=False)
print("\n预测完成!")
print(f"预测结果已保存至: submission_demo.csv")

print("\n预测结果统计:")
print(submission['subscribe'].value_counts())
print(f"\n预测为'yes'的比例: {submission['subscribe'].value_counts(normalize=True).get('yes', 0):.2%}")

print("\n" + "="*60)
print("任务完成!")
print("="*60)
print(f"\n最终模型: {best_model_name_selected}")
print(f"交叉验证Accuracy: {best_score:.4f}")
print(f"提交文件: submission_demo.csv")
