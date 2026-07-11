# ============================================================
# 主題： Linear Regression + 補貨週期 + 安全庫存
# 作者：國立雲林科技大學電機系 林家仁
# 線性回歸預測月需求
# 月需求換算日均需求
# 依補貨週期與安全庫存估算建議存放量
# 再換算合理棧板數

# ============================================================
# 1. 安裝與匯入套件
# ============================================================

!pip install openpyxl -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import re

from google.colab import files
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 2. 圖表基本設定
# ============================================================

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# ============================================================
# 3. 彩色圖表設定
# ============================================================
# 說明：
# 這裡設定一組柔和但清楚的色票，讓每個條狀圖的長條顏色不同，
# 比較適合教學展示與簡報截圖。
# ============================================================

color_palette = [
    "#4E79A7",  # blue
    "#F28E2B",  # orange
    "#E15759",  # red
    "#76B7B2",  # teal
    "#59A14F",  # green
    "#EDC948",  # yellow
    "#B07AA1",  # purple
    "#FF9DA7",  # pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray
    "#86BCB6",
    "#FABFD2",
    "#8CD17D",
    "#B6992D",
    "#499894"
]

def get_colors(n):
    """
    依照資料筆數 n，自動產生足夠數量的顏色。
    """
    return [color_palette[i % len(color_palette)] for i in range(n)]


# ============================================================
# 4. 輔助函數
# ============================================================

def safe_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def safe_ascii(text):
    mapping = {
        "每日出貨箱數": "Daily_Shipment",
        "每日揀貨次數": "Daily_Picking",
        "目前庫存箱數": "Current_Stock",
        "季節係數": "Seasonal_Factor",
        "目前距離主要作業區m": "Distance",
        "體積分數": "Volume_Score",
        "重量分數": "Weight_Score",
        "是否液體_flag": "Liquid",
        "是否易碎_flag": "Fragile",
        "是否季節性_flag": "Seasonal",
        "是否棧板管理_flag": "Pallet_Managed",
        "出貨月化指標": "Monthly_Shipment_Index",
        "揀貨月化指標": "Monthly_Picking_Index",
        "庫存需求壓力": "Stock_Demand_Pressure",
        "季節需求指數": "Seasonal_Demand_Index",
        "高頻大體積指數": "HighFreq_Bulky_Index",
        "重物液體指數": "Heavy_Liquid_Index",
        "商品類別": "Category",
        "KMeans群組": "KMeans_Cluster",
        "目前儲位區": "Current_Area"
    }

    text = str(text)

    for k, v in mapping.items():
        text = text.replace(k, v)

    text = re.sub(r"[^A-Za-z0-9_+\-\.]", "_", text)
    text = re.sub(r"_+", "_", text)

    return text[:45]


# ============================================================
# 5. 上傳 Excel 檔案
# ============================================================

print("請上傳 Warehouse_500SKU_Pallet_Mapping_v3.xlsx")
uploaded = files.upload()

file_name = list(uploaded.keys())[0]

print("已上傳檔案：", file_name)


# ============================================================
# 6. 讀取 Excel：自動偵測真正表頭
# ============================================================

sheet_name = "500件商品"

raw_df = pd.read_excel(file_name, sheet_name=sheet_name, header=None)

header_row = None

for i in range(len(raw_df)):
    row_values = raw_df.iloc[i].astype(str).str.strip().tolist()

    if "SKU編號" in row_values:
        header_row = i
        break

if header_row is None:
    raise ValueError("找不到包含『SKU編號』的表頭列，請確認 Excel 工作表格式。")

df = pd.read_excel(file_name, sheet_name=sheet_name, header=header_row)

df.columns = df.columns.astype(str).str.strip()

df = df.dropna(how="all").reset_index(drop=True)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("Excel 資料讀取完成。")
print("偵測到的表頭列位置：", header_row)


# ============================================================
# 7. 檢查必要欄位
# ============================================================

required_cols = [
    "SKU編號",
    "商品名稱",
    "商品類別",
    "KMeans群組",
    "每日出貨箱數",
    "每日揀貨次數",
    "體積等級",
    "重量等級",
    "是否液體",
    "是否易碎",
    "是否季節性",
    "季節月份",
    "季節係數",
    "目前庫存箱數",
    "每棧板可放箱數",
    "是否棧板管理",
    "目前儲位區",
    "目前距離主要作業區m"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if len(missing_cols) > 0:
    raise ValueError(f"Excel 缺少必要欄位：{missing_cols}")

print("必要欄位檢查完成。")


# ============================================================
# 8. 基礎資料清理
# ============================================================

text_cols = [
    "SKU編號",
    "商品名稱",
    "商品類別",
    "KMeans群組",
    "體積等級",
    "重量等級",
    "是否液體",
    "是否易碎",
    "是否季節性",
    "是否棧板管理",
    "目前儲位區"
]

for col in text_cols:
    df[col] = df[col].astype(str).str.strip()


numeric_cols = [
    "每日出貨箱數",
    "每日揀貨次數",
    "季節係數",
    "目前庫存箱數",
    "每棧板可放箱數",
    "目前距離主要作業區m"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


df["每日出貨箱數"] = df["每日出貨箱數"].fillna(0)
df["每日揀貨次數"] = df["每日揀貨次數"].fillna(0)
df["季節係數"] = df["季節係數"].fillna(1.0)
df["目前庫存箱數"] = df["目前庫存箱數"].fillna(0)
df["每棧板可放箱數"] = df["每棧板可放箱數"].fillna(1)
df["目前距離主要作業區m"] = df["目前距離主要作業區m"].fillna(0)

df["每棧板可放箱數"] = df["每棧板可放箱數"].replace(0, 1)


# ============================================================
# 9. 修正季節性欄位邏輯
# ============================================================

df["季節月份"] = df["季節月份"].fillna("非季節")

df.loc[df["是否季節性"] == "否", "季節月份"] = "非季節"
df.loc[df["是否季節性"] == "否", "季節係數"] = 1.0

df.loc[df["是否季節性"] == "是", "季節係數"] = df.loc[
    df["是否季節性"] == "是",
    "季節係數"
].clip(lower=1.05)


# ============================================================
# 10. 是/否欄位轉 0/1
# ============================================================

yes_no_cols = ["是否液體", "是否易碎", "是否季節性", "是否棧板管理"]

for col in yes_no_cols:
    df[col + "_flag"] = df[col].map({"是": 1, "否": 0}).fillna(0).astype(int)


# ============================================================
# 11. 體積、重量轉換成分數
# ============================================================

volume_map = {
    "小": 1,
    "中": 2,
    "大": 3
}

weight_map = {
    "輕": 1,
    "中": 2,
    "重": 3
}

df["體積分數"] = df["體積等級"].map(volume_map).fillna(2)
df["重量分數"] = df["重量等級"].map(weight_map).fillna(2)

print("資料清理與季節性邏輯修正完成。")


# ============================================================
# 12. 建立工程特徵
# ============================================================

df["出貨月化指標"] = df["每日出貨箱數"] * 30

df["揀貨月化指標"] = df["每日揀貨次數"] * 30

df["庫存需求壓力"] = (
    df["目前庫存箱數"] / (df["每日出貨箱數"] + 1)
).round(2)

df["季節需求指數"] = (
    df["是否季節性_flag"] * (df["季節係數"] - 1.0)
).round(2)

df["高頻大體積指數"] = (
    df["每日揀貨次數"] * df["體積分數"]
).round(2)

df["重物液體指數"] = (
    df["重量分數"] * df["是否液體_flag"]
).round(2)

print("工程特徵建立完成。")


# ============================================================
# 13. 資料預覽
# ============================================================

preview_cols = [
    "SKU編號",
    "商品名稱",
    "商品類別",
    "每日出貨箱數",
    "每日揀貨次數",
    "是否季節性",
    "季節月份",
    "季節係數",
    "目前庫存箱數",
    "是否棧板管理"
]

print("資料預覽：")
print("以下只顯示 10 個關鍵欄位，完整 500 SKU 資料仍會全部投入模型。")
display(df[preview_cols].head(10))


print("是否季節性統計：")
seasonal_count = df["是否季節性"].value_counts().reset_index()
seasonal_count.columns = ["是否季節性", "SKU數量"]
display(seasonal_count)


seasonal_df = df[df["是否季節性"] == "是"].copy()

if len(seasonal_df) > 0:
    print("季節性商品範例：")
    display(seasonal_df[preview_cols].head(10))
else:
    print("目前資料中沒有季節性商品，請確認 Excel 是否有設定『是否季節性 = 是』。")


# ============================================================
# 14. 建立教學用目標值：下期實際需求箱數
# ============================================================

np.random.seed(42)

df["下期實際需求箱數"] = (
    100
    + df["每日出貨箱數"] * 11.5
    + df["每日揀貨次數"] * 4.5
    + df["目前庫存箱數"] * 0.28
    + df["體積分數"] * 25
    + df["重量分數"] * 20
    + df["是否液體_flag"] * 60
    + df["是否易碎_flag"] * 45
    + df["季節需求指數"] * 900
    + df["高頻大體積指數"] * 0.35
    + df["重物液體指數"] * 25
    + np.random.normal(0, 20, len(df))
)

df["下期實際需求箱數"] = df["下期實際需求箱數"].clip(lower=0).round(0)

print("已建立教學用目標欄位：下期實際需求箱數")
print("說明：這是模型訓練用的標準答案 y，不是模型預測值。")


target_preview_cols = [
    "SKU編號",
    "商品名稱",
    "商品類別",
    "每日出貨箱數",
    "每日揀貨次數",
    "是否季節性",
    "季節月份",
    "季節係數",
    "目前庫存箱數",
    "下期實際需求箱數"
]

print("加入下期實際需求箱數後的資料預覽：")
display(df[target_preview_cols].head(10))


# ============================================================
# 15. 篩選棧板管理 SKU
# ============================================================

df_pallet = df[df["是否棧板管理_flag"] == 1].copy()

print("全部 SKU 數量：", len(df))
print("棧板管理 SKU 數量：", len(df_pallet))
print("非棧板管理 SKU 數量：", len(df) - len(df_pallet))


# ============================================================
# 16. 建立線性回歸模型
# ============================================================

numeric_features = [
    "每日出貨箱數",
    "每日揀貨次數",
    "目前庫存箱數",
    "體積分數",
    "重量分數",
    "是否液體_flag",
    "是否易碎_flag",
    "是否季節性_flag",
    "季節係數",
    "目前距離主要作業區m",
    "出貨月化指標",
    "揀貨月化指標",
    "庫存需求壓力",
    "季節需求指數",
    "高頻大體積指數",
    "重物液體指數"
]

categorical_features = [
    "商品類別",
    "KMeans群組",
    "目前儲位區"
]

X = df_pallet[numeric_features + categorical_features]

y = df_pallet["下期實際需求箱數"]


preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("訓練資料筆數：", len(X_train))
print("測試資料筆數：", len(X_test))


model.fit(X_train, y_train)

print("線性回歸模型訓練完成。")


# ============================================================
# 17. 模型預測與評估
# ============================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = safe_mape(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

eval_df = pd.DataFrame({
    "評估指標": ["MAE", "RMSE", "MAPE(%)", "R2"],
    "數值": [
        round(mae, 2),
        round(rmse, 2),
        round(mape, 2),
        round(r2, 4)
    ],
    "說明": [
        "平均每筆預測誤差箱數",
        "較重視大誤差的平均誤差箱數",
        "平均百分比誤差",
        "模型解釋需求變化的能力"
    ]
})

print("模型評估結果：")
display(eval_df)


# ============================================================
# 18. 圖表一：Actual vs Predicted
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.75,
    color="#4E79A7",
    edgecolors="white",
    linewidths=0.6
)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    color="#E15759",
    linewidth=2
)

plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.title("Actual vs Predicted Demand")
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# 19. 圖表二：Residual Plot
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.75,
    color="#59A14F",
    edgecolors="white",
    linewidths=0.6
)

plt.axhline(
    0,
    linestyle="--",
    color="#E15759",
    linewidth=2
)

plt.xlabel("Predicted Demand")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# 20. 回歸係數分析
# ============================================================

regressor = model.named_steps["regressor"]
preprocessor_fitted = model.named_steps["preprocessor"]

cat_encoder = preprocessor_fitted.named_transformers_["cat"]
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)

all_feature_names = numeric_features + list(cat_feature_names)

coef_df = pd.DataFrame({
    "特徵欄位": all_feature_names,
    "回歸係數": regressor.coef_
})

coef_df["回歸係數"] = coef_df["回歸係數"].round(2)

coef_df["影響方向"] = np.where(
    coef_df["回歸係數"] > 0,
    "正向影響",
    np.where(coef_df["回歸係數"] < 0, "負向影響", "無明顯影響")
)

coef_df["影響程度"] = coef_df["回歸係數"].abs().round(2)

coef_df = coef_df.sort_values("影響程度", ascending=False).reset_index(drop=True)

coef_df["圖表用英文特徵"] = coef_df["特徵欄位"].apply(safe_ascii)

print("回歸係數分析：")
display(coef_df.head(20))


# ============================================================
# 21. 圖表三：Top 15 Feature Coefficients
# ============================================================

coef_plot_df = coef_df.head(15).copy()
coef_plot_df = coef_plot_df.sort_values("回歸係數")

plt.figure(figsize=(10, 6))

bar_colors = np.where(
    coef_plot_df["回歸係數"] >= 0,
    "#4E79A7",
    "#E15759"
)

plt.barh(
    coef_plot_df["圖表用英文特徵"],
    coef_plot_df["回歸係數"],
    color=bar_colors
)

plt.axvline(
    0,
    color="#333333",
    linewidth=1
)

plt.xlabel("Coefficient")
plt.ylabel("Feature")
plt.title("Top 15 Regression Coefficients")
plt.grid(axis="x", alpha=0.3)
plt.show()


# ============================================================
# 22. 使用模型預測全部棧板管理 SKU
# ============================================================

df_pallet["下期預測需求箱數"] = model.predict(
    df_pallet[numeric_features + categorical_features]
)

df_pallet["下期預測需求箱數"] = (
    df_pallet["下期預測需求箱數"]
    .clip(lower=0)
    .round(0)
)

df_pallet["預測誤差箱數"] = (
    df_pallet["下期實際需求箱數"]
    - df_pallet["下期預測需求箱數"]
).round(0)

df_pallet["預測日均需求箱數"] = (
    df_pallet["下期預測需求箱數"] / 30
).round(2)


prediction_preview_cols = [
    "SKU編號",
    "商品名稱",
    "每日出貨箱數",
    "每日揀貨次數",
    "是否季節性",
    "季節月份",
    "季節係數",
    "下期實際需求箱數",
    "下期預測需求箱數",
    "預測誤差箱數",
    "預測日均需求箱數"
]

print("下期需求預測結果預覽：")
display(df_pallet[prediction_preview_cols].head(10))


# ============================================================
# 23. 補貨週期與安全庫存天數
# ============================================================

def calc_replenishment_days(row):
    picking = row["每日揀貨次數"]

    if picking >= 100:
        return 3
    elif picking >= 70:
        return 4
    elif picking >= 50:
        return 5
    elif row["是否季節性_flag"] == 1:
        return 10
    elif row["體積等級"] == "大" or row["重量等級"] == "重" or row["是否液體_flag"] == 1:
        return 7
    else:
        return 7


def calc_safety_stock_days(row):
    picking = row["每日揀貨次數"]

    if row["是否季節性_flag"] == 1:
        return 5
    elif picking >= 70:
        return 3
    elif picking >= 50:
        return 2
    else:
        return 2


df_pallet["補貨週期天數"] = df_pallet.apply(calc_replenishment_days, axis=1)
df_pallet["安全庫存天數"] = df_pallet.apply(calc_safety_stock_days, axis=1)


# ============================================================
# 24. 需求轉換：下期預測需求 → 建議存放箱數
# ============================================================

df_pallet["建議存放箱數"] = (
    df_pallet["預測日均需求箱數"]
    * (
        df_pallet["補貨週期天數"]
        + df_pallet["安全庫存天數"]
    )
)

df_pallet["建議存放箱數"] = (
    df_pallet["建議存放箱數"]
    .clip(lower=1)
    .round(0)
)


# ============================================================
# 25. 棧板需求換算
# ============================================================

df_pallet["目前棧板數"] = np.ceil(
    df_pallet["目前庫存箱數"] / df_pallet["每棧板可放箱數"]
)

df_pallet["模型建議棧板數"] = np.ceil(
    df_pallet["建議存放箱數"] / df_pallet["每棧板可放箱數"]
)

df_pallet["目前棧板數"] = (
    df_pallet["目前棧板數"]
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
    .astype(int)
)

df_pallet["模型建議棧板數"] = (
    df_pallet["模型建議棧板數"]
    .replace([np.inf, -np.inf], 0)
    .fillna(0)
    .astype(int)
)

df_pallet["棧板增減"] = (
    df_pallet["模型建議棧板數"]
    - df_pallet["目前棧板數"]
)

print("棧板需求換算完成。")


# ============================================================
# 26. 儲位建議規則
# ============================================================

def recommend_storage(row):
    picking = row["每日揀貨次數"]
    delta = row["棧板增減"]

    if row["是否季節性_flag"] == 1 and delta >= 1:
        return "季節彈性儲位"

    if picking >= 70 and (row["重量等級"] == "重" or row["是否液體_flag"] == 1):
        return "近端低層棧板區"

    if picking >= 70 and row["體積等級"] == "大":
        return "近端大貨位"

    if picking >= 40 and row["體積等級"] == "大":
        return "中段大貨位"

    if delta <= -3:
        return "遠端棧板區"

    return "中段棧板區"


def forward_move(row):
    picking = row["每日揀貨次數"]
    distance = row["目前距離主要作業區m"]
    delta = row["棧板增減"]

    if row["是否季節性_flag"] == 1 and delta >= 2:
        return "建議前移"

    if picking >= 70 and distance >= 40 and delta >= 0:
        return "建議前移"

    if picking >= 50 and delta >= 3:
        return "建議前移"

    return "維持現況"


def forward_reason(row):
    if row["前移判斷"] == "維持現況":
        return "目前儲位可維持"

    if row["是否季節性_flag"] == 1 and row["棧板增減"] >= 2:
        return "季節性商品需求增加，建議旺季前移"

    if row["每日揀貨次數"] >= 70 and row["目前距離主要作業區m"] >= 40:
        return "高頻商品目前距離主要作業區較遠"

    if row["每日揀貨次數"] >= 50 and row["棧板增減"] >= 3:
        return "預測棧板需求明顯增加"

    return "依需求預測與儲位距離建議前移"


df_pallet["建議儲位區"] = df_pallet.apply(recommend_storage, axis=1)
df_pallet["前移判斷"] = df_pallet.apply(forward_move, axis=1)
df_pallet["前移原因"] = df_pallet.apply(forward_reason, axis=1)


# ============================================================
# 27. 結果表
# ============================================================

result_cols = [
    "SKU編號",
    "商品名稱",
    "商品類別",
    "KMeans群組",
    "每日出貨箱數",
    "每日揀貨次數",
    "體積等級",
    "重量等級",
    "是否液體",
    "是否易碎",
    "是否季節性",
    "季節月份",
    "季節係數",
    "目前庫存箱數",
    "每棧板可放箱數",
    "下期實際需求箱數",
    "下期預測需求箱數",
    "預測誤差箱數",
    "預測日均需求箱數",
    "補貨週期天數",
    "安全庫存天數",
    "建議存放箱數",
    "目前棧板數",
    "模型建議棧板數",
    "棧板增減",
    "目前儲位區",
    "目前距離主要作業區m",
    "建議儲位區",
    "前移判斷",
    "前移原因"
]

print("棧板需求預測與儲位建議結果預覽：")
display(df_pallet[result_cols].head(20))


# ============================================================
# 28. 棧板增減區間統計
# ============================================================

bins = [-999, -6, -3, -1, 0, 1, 3, 5, 10, 999]
labels = [
    "<= -6",
    "-5 to -3",
    "-2 to -1",
    "0",
    "+1",
    "+2 to +3",
    "+4 to +5",
    "+6 to +10",
    "> +10"
]

df_pallet["棧板增減區間"] = pd.cut(
    df_pallet["棧板增減"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

pallet_range_count = (
    df_pallet["棧板增減區間"]
    .value_counts()
    .sort_index()
    .reset_index()
)

pallet_range_count.columns = ["棧板增減區間", "SKU數量"]

print("棧板增減區間統計：")
display(pallet_range_count)


plt.figure(figsize=(10, 5))

plt.bar(
    pallet_range_count["棧板增減區間"].astype(str),
    pallet_range_count["SKU數量"],
    color=get_colors(len(pallet_range_count))
)

plt.xlabel("Pallet Change Range")
plt.ylabel("SKU Count")
plt.title("Predicted Pallet Change by Range")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)

for i, v in enumerate(pallet_range_count["SKU數量"]):
    plt.text(i, v + 0.5, str(v), ha="center")

plt.show()


# ============================================================
# 29. 建議儲位區統計
# ============================================================

storage_count = (
    df_pallet["建議儲位區"]
    .value_counts()
    .reset_index()
)

storage_count.columns = ["建議儲位區", "SKU數量"]

print("建議儲位區統計：")
display(storage_count)


storage_english_map = {
    "季節彈性儲位": "Seasonal Flexible Area",
    "近端低層棧板區": "Near Low-level Pallet Area",
    "近端大貨位": "Near Bulky Goods Area",
    "中段大貨位": "Middle Bulky Goods Area",
    "遠端棧板區": "Far Pallet Area",
    "中段棧板區": "Middle Pallet Area"
}

storage_count["Storage_Area_EN"] = storage_count["建議儲位區"].map(storage_english_map)

storage_plot_df = storage_count.sort_values("SKU數量")

plt.figure(figsize=(10, 5))

plt.barh(
    storage_plot_df["Storage_Area_EN"],
    storage_plot_df["SKU數量"],
    color=get_colors(len(storage_plot_df))
)

for i, v in enumerate(storage_plot_df["SKU數量"]):
    plt.text(v + 0.5, i, str(v), va="center")

plt.xlabel("SKU Count")
plt.ylabel("Recommended Storage Area")
plt.title("Recommended Storage Area Distribution")
plt.grid(axis="x", alpha=0.3)
plt.show()


# ============================================================
# 30. 前移判斷統計
# ============================================================

move_count = (
    df_pallet["前移判斷"]
    .value_counts()
    .reset_index()
)

move_count.columns = ["前移判斷", "SKU數量"]

print("前移判斷統計：")
display(move_count)


move_english_map = {
    "建議前移": "Move Forward",
    "維持現況": "Keep Current"
}

move_count["Move_Decision_EN"] = move_count["前移判斷"].map(move_english_map)

plt.figure(figsize=(7, 5))

plt.bar(
    move_count["Move_Decision_EN"],
    move_count["SKU數量"],
    color=get_colors(len(move_count))
)

for i, v in enumerate(move_count["SKU數量"]):
    plt.text(i, v + 0.5, str(v), ha="center")

plt.xlabel("Move Decision")
plt.ylabel("SKU Count")
plt.title("Forward Move Decision")
plt.grid(axis="y", alpha=0.3)
plt.show()


# ============================================================
# 31. Top 10 棧板增加 SKU
# ============================================================

top_increase = (
    df_pallet
    .sort_values("棧板增減", ascending=False)
    .head(10)
    .copy()
)

top_increase["SKU_Label"] = (
    top_increase["SKU編號"].astype(str)
    + " | +"
    + top_increase["棧板增減"].astype(str)
)

top_plot_df = top_increase.sort_values("棧板增減")

plt.figure(figsize=(10, 6))

plt.barh(
    top_plot_df["SKU_Label"],
    top_plot_df["棧板增減"],
    color=get_colors(len(top_plot_df))
)

for i, v in enumerate(top_plot_df["棧板增減"]):
    plt.text(v + 0.2, i, str(v), va="center")

plt.xlabel("Pallet Increase")
plt.ylabel("SKU")
plt.title("Top 10 SKUs by Recommended Pallet Increase")
plt.grid(axis="x", alpha=0.3)
plt.show()


print("Top 10 棧板增加 SKU：")
display(
    top_increase[
        [
            "SKU編號",
            "商品名稱",
            "商品類別",
            "每日揀貨次數",
            "是否季節性",
            "季節月份",
            "下期預測需求箱數",
            "目前棧板數",
            "模型建議棧板數",
            "棧板增減",
            "建議儲位區",
            "前移判斷",
            "前移原因"
        ]
    ]
)


# ============================================================
# 32. 優先前移 SKU
# ============================================================

priority_sku = (
    df_pallet[df_pallet["前移判斷"] == "建議前移"]
    .sort_values(
        ["棧板增減", "每日揀貨次數", "目前距離主要作業區m"],
        ascending=[False, False, False]
    )
)

print("優先前移 SKU 清單：")
display(
    priority_sku[
        [
            "SKU編號",
            "商品名稱",
            "商品類別",
            "每日揀貨次數",
            "是否季節性",
            "季節月份",
            "下期預測需求箱數",
            "目前距離主要作業區m",
            "目前棧板數",
            "模型建議棧板數",
            "棧板增減",
            "目前儲位區",
            "建議儲位區",
            "前移原因"
        ]
    ].head(30)
)


# ============================================================
# 33. 模型摘要
# ============================================================

model_summary = pd.DataFrame({
    "項目": [
        "全部 SKU 數量",
        "棧板管理 SKU 數量",
        "非棧板管理 SKU 數量",
        "訓練資料筆數",
        "測試資料筆數",
        "MAE",
        "RMSE",
        "MAPE(%)",
        "R2",
        "平均下期實際需求箱數",
        "平均下期預測需求箱數",
        "平均目前棧板數",
        "平均模型建議棧板數",
        "建議前移 SKU 數量"
    ],
    "數值": [
        len(df),
        len(df_pallet),
        len(df) - len(df_pallet),
        len(X_train),
        len(X_test),
        round(mae, 2),
        round(rmse, 2),
        round(mape, 2),
        round(r2, 4),
        round(df_pallet["下期實際需求箱數"].mean(), 2),
        round(df_pallet["下期預測需求箱數"].mean(), 2),
        round(df_pallet["目前棧板數"].mean(), 2),
        round(df_pallet["模型建議棧板數"].mean(), 2),
        len(priority_sku)
    ],
    "說明": [
        "原始 SKU 商品數",
        "納入棧板需求預測與儲位分析的商品數",
        "未納入本次棧板分析的商品數",
        "用來訓練線性回歸模型的資料筆數",
        "用來驗證模型預測能力的資料筆數",
        "平均每筆需求預測誤差箱數",
        "較重視大誤差的預測誤差指標",
        "平均百分比誤差",
        "模型解釋需求變化能力",
        "教學資料中的平均標準答案",
        "模型輸出的平均下期預測需求",
        "目前平均棧板使用量",
        "模型建議後的平均棧板需求量",
        "依據需求、距離與棧板增減判斷需要前移的 SKU 數"
    ]
})

print("模型摘要：")
display(model_summary)


# ============================================================
# 34. 匯出 Excel 結果
# ============================================================

output_file = "Warehouse_500SKU_LinearRegression_Teaching_ColorChart_Result.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="500SKU_含實際需求", index=False)
    df_pallet[result_cols + ["棧板增減區間"]].to_excel(writer, sheet_name="棧板預測結果", index=False)
    eval_df.to_excel(writer, sheet_name="模型評估", index=False)
    coef_df.to_excel(writer, sheet_name="回歸係數分析", index=False)
    priority_sku.to_excel(writer, sheet_name="優先前移SKU", index=False)
    storage_count.to_excel(writer, sheet_name="建議儲位統計", index=False)
    move_count.to_excel(writer, sheet_name="前移判斷統計", index=False)
    pallet_range_count.to_excel(writer, sheet_name="棧板增減區間統計", index=False)
    model_summary.to_excel(writer, sheet_name="模型摘要", index=False)

print("Excel 結果檔案已產生：", output_file)

files.download(output_file)

