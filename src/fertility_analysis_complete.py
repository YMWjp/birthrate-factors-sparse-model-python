
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "../data/processed/combined_data.csv"
data = pd.read_csv(file_path)

# 2020年のデータのみをフィルタリング
# data = data[data['Year'] == 2020]

# Data preprocessing
data["Immigrant Ratio"] = (
    data["Total number of international immigrants"] /
    data["Population - Sex: all - Age: all - Variant: estimates_population-with-un-projections.csv"]
)
data["Fertility Rate"] = data["Fertility rate - Sex: all - Age: all - Variant: estimates"]

all_columns = [
    "Fertility Rate",
    "GDP per capita",
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

variable_mapping = {
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

filtered_data = data[all_columns].dropna()
filtered_data_renamed = filtered_data.rename(columns=variable_mapping)
# Correlation analysis
plt.figure(figsize=(12, 10))
correlation_matrix = filtered_data_renamed.corr()
sns.heatmap(correlation_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="coolwarm",
    xticklabels=correlation_matrix.columns,
    yticklabels=correlation_matrix.columns,
    annot_kws={'size': 8}  # 相関係数の文字サイズを調整
)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.title("Correlation Heatmap: Fertility Rate and Related Variables", pad=20)
plt.tight_layout()
plt.savefig("../results/figures/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

# Linear regression
X = filtered_data.drop(columns=["Fertility Rate"])
y = filtered_data["Fertility Rate"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_
})
coefficients.to_csv("../results/csv/linear_regression_coefficients.csv", index=False)

# SVM regression
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1))
])
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
mse_svm = mean_squared_error(y_test, y_pred_svm)
r2_svm = r2_score(y_test, y_pred_svm)

# Actual vs Predicted plot for SVM
plt.figure(figsize=(10, 8))
# データポイントに年を追加
for i, (actual, predicted) in enumerate(zip(y_test, y_pred_svm)):
    year = data.loc[y_test.index[i], 'Year']  # テストデータのインデックスから年を取得
    plt.annotate(str(int(year)), 
                (actual, predicted),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8)

plt.scatter(y_test, y_pred_svm, alpha=0.7, edgecolor="k", s=100)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
    "r--", 
    lw=2, 
    label="Perfect Prediction Line"
)
plt.title("Actual vs Predicted Fertility Rates (SVM Model)", fontsize=14, pad=20)
plt.xlabel("Actual Fertility Rate", fontsize=12)
plt.ylabel("Predicted Fertility Rate", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("../results/figures/svm_actual_vs_predicted.png", dpi=300, bbox_inches='tight')
plt.close()

# Print results
print("Linear Regression Results:")
print(f"MSE: {mse_linear}, R^2: {r2_linear}")
print("SVM Regression Results:")
print(f"MSE: {mse_svm}, R^2: {r2_svm}")
