import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels import PanelOLS

def remove_outliers(df, columns, n_std=3):
    """異常値の除去"""
    for col in columns:
        mean = df[col].mean()
        std = df[col].std()
        df = df[(abs(df[col] - mean) <= (n_std * std))]
    return df

def check_multicollinearity(X):
    """マルチコリニアリティのチェック"""
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# データの読み込みと前処理
file_path = "../data/processed/combined_data.csv"
data = pd.read_csv(file_path)

# 基本的な特徴量の作成
data["Immigrant Ratio"] = (
    data["Total number of international immigrants"] /
    data["Population - Sex: all - Age: all - Variant: estimates_population-with-un-projections.csv"]
)
data["Fertility Rate"] = data["Fertility rate - Sex: all - Age: all - Variant: estimates"]

# 分析に使用する列の定義
all_columns = [
    "Entity", "Year", "Fertility Rate", "GDP per capita",
    "Share of population living in urban areas",
    "Public spending on education as a share of GDP",
    "Physicians (per 1,000 people)",
    "Total days of leave available for the mother",
    "Average annual working hours per worker",
    "Gender Development Index",
    "Human Development Index",
    "Labor force participation rate, female (% of female population ages 15+) (modeled ILO estimate)",
    "Age-standardized death rate from self-inflicted injuries per 100,000 population - Sex: Both sexes - Age group: all ages",
    "Gender Inequality Index",
    "Immigrant Ratio"
]

# 変数名のマッピング
variable_mapping = {
    "Entity": "Country",
    "Fertility Rate": "Fertility",
    "GDP per capita": "GDP",
    "Share of population living in urban areas": "Urban Pop",
    "Public spending on education as a share of GDP": "Edu Spend",
    "Physicians (per 1,000 people)": "Physicians",
    "Total days of leave available for the mother": "Mat Leave",
    "Average annual working hours per worker": "Work Hours",
    "Gender Development Index": "GDI",
    "Human Development Index": "HDI",
    "Labor force participation rate, female (% of female population ages 15+) (modeled ILO estimate)": "Female Labor",
    "Age-standardized death rate from self-inflicted injuries per 100,000 population - Sex: Both sexes - Age group: all ages": "Suicide Rate",
    "Gender Inequality Index": "GII",
    "Immigrant Ratio": "Immigrants"
}

# データの前処理
filtered_data = data[all_columns].copy()
filtered_data = filtered_data.rename(columns=variable_mapping)

# 国ごとの欠損値を平均値で補完
numeric_columns = filtered_data.select_dtypes(include=[np.number]).columns
filtered_data[numeric_columns] = filtered_data.groupby('Country')[numeric_columns].transform(
    lambda x: x.fillna(x.mean())
)

# 残りの欠損値を含む行を削除
filtered_data = filtered_data.dropna()

# 異常値の除去
numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
filtered_data = remove_outliers(filtered_data, numeric_cols)

# 相関分析
plt.figure(figsize=(12, 10))
correlation_matrix = filtered_data.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    xticklabels=correlation_matrix.columns,
    yticklabels=correlation_matrix.columns,
    annot_kws={'size': 8}
)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title("Correlation Heatmap: Fertility Rate and Related Variables", pad=20)
plt.tight_layout()
plt.savefig("../results/figures/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# マルチコリニアリティのチェック
X_cols = [col for col in numeric_cols if col not in ['Fertility', 'Year']]
vif_df = check_multicollinearity(filtered_data[X_cols])
print("VIF Values:")
print(vif_df)

# 時系列による訓練データとテストデータの分割
train_end_year = 2018
X = filtered_data[X_cols]
y = filtered_data['Fertility']

X_train = X[filtered_data['Year'] <= train_end_year]
X_test = X[filtered_data['Year'] > train_end_year]
y_train = y[filtered_data['Year'] <= train_end_year]
y_test = y[filtered_data['Year'] > train_end_year]

# パネルデータ分析
panel_data = filtered_data.set_index(['Country', 'Year'])
exog_vars = X_cols
panel_model = PanelOLS(panel_data['Fertility'], 
                      panel_data[exog_vars],
                      entity_effects=True)
panel_results = panel_model.fit()
print("\nPanel Regression Results:")
print(panel_results)

# SVMモデル（時系列クロスバリデーション付き）
tscv = TimeSeriesSplit(n_splits=5)
svm_scores = []

for train_idx, val_idx in tscv.split(X_train):
    X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1))
    ])
    
    svm_model.fit(X_train_cv, y_train_cv)
    y_pred_cv = svm_model.predict(X_val_cv)
    score = r2_score(y_val_cv, y_pred_cv)
    svm_scores.append(score)

# 最終モデルの評価
final_svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1))
])
final_svm_model.fit(X_train, y_train)
y_pred_svm = final_svm_model.predict(X_test)

# モデル評価指標の計算
mse_svm = mean_squared_error(y_test, y_pred_svm)
mae_svm = mean_absolute_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)

# 予測結果の可視化（改善版）
plt.figure(figsize=(12, 8))

# メインの散布図
scatter = plt.scatter(y_test, y_pred_svm, 
    alpha=0.7, 
    c=filtered_data.loc[y_test.index, 'Year'],  # 年による色分け
    cmap='viridis', 
    s=100,
    edgecolor='white',
    linewidth=0.5
)

# 完全一致線
plt.plot([y_test.min(), y_test.max()], 
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="理想的な予測線（実測値=予測値）"
)

# 誤差範囲の表示
plt.fill_between(
    [y_test.min(), y_test.max()],
    [y_test.min() - mae_svm, y_test.max() - mae_svm],
    [y_test.min() + mae_svm, y_test.max() + mae_svm],
    alpha=0.2,
    color='gray',
    label=f'平均絶対誤差 (MAE) 範囲: ±{mae_svm:.3f}'
)

# データポイントのラベル付け
for i, (actual, predicted) in enumerate(zip(y_test, y_pred_svm)):
    year = filtered_data.loc[y_test.index[i], 'Year']
    country = filtered_data.loc[y_test.index[i], 'Country']
    
    # 予測誤差が大きいポイントのみラベル表示
    error = abs(actual - predicted)
    if error > mae_svm:
        plt.annotate(f'{country}\n{int(year)}',
                    (actual, predicted),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

# グラフの装飾
plt.colorbar(scatter, label='予測年')
plt.title("出生率の実測値と予測値の比較\n(SVMモデルによる予測)", 
    fontsize=14, 
    pad=20
)
plt.xlabel("実測値 (出生率)", fontsize=12)
plt.ylabel("予測値 (出生率)", fontsize=12)

# 統計情報の追加
stats_text = f'モデル性能指標:\nR² = {r2_svm:.3f}\nMSE = {mse_svm:.3f}\nMAE = {mae_svm:.3f}'
plt.text(0.05, 0.95, 
    stats_text,
    transform=plt.gca().transAxes,
    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8),
    verticalalignment='top',
    fontsize=10
)

plt.legend(fontsize=10, loc='lower right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# グラフの保存
plt.savefig("../results/figures/svm_actual_vs_predicted_improved.png", 
    dpi=300, 
    bbox_inches='tight'
)
plt.close()

# 結果の出力
print("\nSVM Model Results:")
print(f"Mean Cross-validation R² score: {np.mean(svm_scores):.4f}")
print(f"Test set MSE: {mse_svm:.4f}")
print(f"Test set MAE: {mae_svm:.4f}")
print(f"Test set R²: {r2_svm:.4f}")