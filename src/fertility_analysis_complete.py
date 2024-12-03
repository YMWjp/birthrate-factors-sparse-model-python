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
plt.figure(figsize=(15, 12))  # サイズを大きくする
correlation_matrix = filtered_data.select_dtypes(include=[np.number]).drop('Year', axis=1).corr()

# Fertilityと他の変数との相関のみを抽出
fertility_correlations = correlation_matrix['Fertility'].sort_values(ascending=False)
selected_columns = ['Fertility'] + fertility_correlations.index[1:].tolist()
selected_correlation_matrix = correlation_matrix.loc[selected_columns, selected_columns]

# ヒートマップの作成
sns.heatmap(selected_correlation_matrix,
    annot=True,  # 相関係数を表示
    fmt=".3f",   # 小数点3桁まで表示
    cmap="RdBu_r",  # Red-Blue colormap
    center=0,    # 0を中心とした色付け
    square=True, # 正方形のセル
    xticklabels=selected_correlation_matrix.columns,
    yticklabels=selected_correlation_matrix.columns,
    annot_kws={'size': 10}  # 注釈のフォントサイズを大きく
)

# グラフの体裁を整える
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title("Correlation Heatmap: Fertility Rate and Related Variables", pad=20, fontsize=14)

# 余白の調整
plt.tight_layout()

# 高解像度で保存
plt.savefig("../results/figures/fertility_correlation_heatmap.png", 
    dpi=300, 
    bbox_inches='tight'
)
plt.close()

# Create enhanced correlation bar plot
plt.figure(figsize=(12, 8))
fertility_correlations_sorted = fertility_correlations[1:].sort_values()

# Plot horizontal bars with enhanced styling
bars = plt.barh(range(len(fertility_correlations_sorted)), 
                fertility_correlations_sorted,
                color=['#FF6B6B' if x < 0 else '#4ECDC4' for x in fertility_correlations_sorted],
                alpha=0.7)

# Enhance graph decorations
plt.yticks(range(len(fertility_correlations_sorted)), 
           fertility_correlations_sorted.index,
           fontsize=10)
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.title('Correlation Coefficients with Fertility Rate', fontsize=14, pad=20)

# Add value labels with improved positioning
for i, v in enumerate(fertility_correlations_sorted):
    plt.text(v + (0.01 if v >= 0 else -0.01), 
             i,
             f'{v:.3f}',
             va='center',
             ha='left' if v >= 0 else 'right',
             fontsize=10,
             bbox=dict(facecolor='white', 
                      edgecolor='none', 
                      alpha=0.7,
                      pad=1))

# Add enhanced grid and reference line
plt.grid(True, alpha=0.3, axis='x', linestyle='--')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Add explanatory text
plt.text(0.02, -0.15, 
         'Positive correlations indicate variables that increase with fertility rate\n'
         'Negative correlations indicate variables that decrease with fertility rate',
         transform=plt.gca().transAxes,
         fontsize=10,
         style='italic')

plt.tight_layout()

# Save high-resolution bar plot
plt.savefig("../results/figures/fertility_correlation_barplot.png",
    dpi=300,
    bbox_inches='tight'
)
plt.close()

# 相関係数の絶対値でソート（Fertilityとの相関）
abs_correlations = abs(fertility_correlations)
top_correlations = fertility_correlations[abs_correlations.sort_values(ascending=False)[1:].index]

# 上位2つの正の相関と負の相関を取得
positive_corr = top_correlations[top_correlations > 0][:2]
negative_corr = top_correlations[top_correlations < 0][:2]

# 散布図の作成
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle('Strongest Correlations with Fertility Rate', fontsize=16, y=1.02)

# 正の相関をプロット
for i, (var, corr) in enumerate(positive_corr.items()):
    sns.scatterplot(
        data=filtered_data,
        x=var,
        y='Fertility',
        ax=axes[0, i],
        alpha=0.6
    )
    axes[0, i].set_title(f'{var}\nCorrelation: {corr:.3f}')
    axes[0, i].grid(True, alpha=0.3)
    
    # 回帰直線を追加
    sns.regplot(
        data=filtered_data,
        x=var,
        y='Fertility',
        ax=axes[0, i],
        scatter=False,
        color='red',
        line_kws={'linestyle': '--'}
    )

# 負の相関をプロット
for i, (var, corr) in enumerate(negative_corr.items()):
    sns.scatterplot(
        data=filtered_data,
        x=var,
        y='Fertility',
        ax=axes[1, i],
        alpha=0.6
    )
    axes[1, i].set_title(f'{var}\nCorrelation: {corr:.3f}')
    axes[1, i].grid(True, alpha=0.3)
    
    # 回帰直線を追加
    sns.regplot(
        data=filtered_data,
        x=var,
        y='Fertility',
        ax=axes[1, i],
        scatter=False,
        color='red',
        line_kws={'linestyle': '--'}
    )

plt.tight_layout()
plt.savefig("../results/figures/fertility_top_correlations_scatter.png", 
    dpi=300, 
    bbox_inches='tight'
)
plt.close()

# 相関係数を出力
print("\nTop Positive Correlations with Fertility:")
print(positive_corr)
print("\nTop Negative Correlations with Fertility:")
print(negative_corr)

# マルチコリニアリティのチェック
X_cols = [col for col in numeric_cols if col not in ['Fertility', 'Year']]
vif_df = check_multicollinearity(filtered_data[X_cols])
print("VIF Values:")
print(vif_df)

# 時系列による訓練データとテストデータの分割
train_end_year = 2000
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

# Visualization of prediction results (improved version)
plt.figure(figsize=(15, 10))

# Main scatter plot
scatter = plt.scatter(y_test, y_pred_svm, 
    alpha=0.7, 
    c=filtered_data.loc[y_test.index, 'Year'],
    cmap='viridis', 
    s=120,
    edgecolor='white',
    linewidth=0.5
)

# Perfect prediction line
plt.plot([y_test.min(), y_test.max()], 
    [y_test.min(), y_test.max()],
    "r--",
    lw=2,
    label="Perfect Prediction Line (Actual = Predicted)"
)

# Error range display
plt.fill_between(
    [y_test.min(), y_test.max()],
    [y_test.min() - mae_svm, y_test.max() - mae_svm],
    [y_test.min() + mae_svm, y_test.max() + mae_svm],
    alpha=0.2,
    color='gray',
    label=f'Mean Absolute Error (MAE) Range: ±{mae_svm:.3f}'
)

# Data point labeling
for i, (actual, predicted) in enumerate(zip(y_test, y_pred_svm)):
    year = filtered_data.loc[y_test.index[i], 'Year']
    country = filtered_data.loc[y_test.index[i], 'Country']
    
    error = abs(actual - predicted)
    if error > mae_svm:
        plt.annotate(f'{country}\n{int(year)}',
                    (actual, predicted),
                    xytext=(7, 7),
                    textcoords='offset points',
                    fontsize=10,
                    bbox=dict(facecolor='white', 
                             edgecolor='none', 
                             alpha=0.7,
                             pad=2)
                    )

# Graph decorations
cbar = plt.colorbar(scatter, label='Prediction Year')
cbar.ax.tick_params(labelsize=10)

plt.title("Comparison of Actual vs Predicted Fertility Rates\n(SVM Model Predictions)", 
    fontsize=16, 
    pad=20
)
plt.xlabel("Actual Fertility Rate", fontsize=14)
plt.ylabel("Predicted Fertility Rate", fontsize=14)

# Add statistical information
stats_text = (f'Model Performance Metrics:\n'
             f'R² = {r2_svm:.3f}\n'
             f'MSE = {mse_svm:.3f}\n'
             f'MAE = {mae_svm:.3f}')
plt.text(0.05, 0.95, 
    stats_text,
    transform=plt.gca().transAxes,
    bbox=dict(facecolor='white', 
             edgecolor='gray', 
             alpha=0.8,
             pad=5),
    verticalalignment='top',
    fontsize=12
)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12, loc='lower right', bbox_to_anchor=(1, 0.1))
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

plt.savefig("../results/figures/svm_actual_vs_predicted_improved.png", 
    dpi=300, 
    bbox_inches='tight'
)
plt.close()

# Output results
print("\nSVM Model Results:")
print(f"Mean Cross-validation R² score: {np.mean(svm_scores):.4f}")
print(f"Test set MSE: {mse_svm:.4f}")
print(f"Test set MAE: {mae_svm:.4f}")
print(f"Test set R²: {r2_svm:.4f}")