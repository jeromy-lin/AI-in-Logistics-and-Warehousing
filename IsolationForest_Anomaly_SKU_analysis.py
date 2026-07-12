# ============================================================
# IsolationForest_Anomaly_SKU_analysis 儲位異常偵測範例
# 作者：國立雲林科技大學電機系 林家仁
# 515 SKU Inventory Anomaly Detection
# Isolation Forest Teaching Version
# Goal:
# 1. Detect about 3% anomaly SKUs
# 2. Avoid using teacher answer sheet
# 3. Add Anomaly Score Trend with decision threshold
# 4. Avoid Matplotlib CJK glyph warnings
# ============================================================


# ============================================================
# 1. Install and import packages
# ============================================================

!pip install openpyxl -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import warnings
import re

from google.colab import files
from IPython.display import display

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ============================================================
# 2. Chart settings
# ============================================================
# Important:
# All chart titles, labels, tick labels, legends must use English.
# Tables can still use Chinese.
# ============================================================

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Suppress CJK glyph warning just in case
warnings.filterwarnings(
    "ignore",
    message="Glyph .* missing from font.*"
)


# ============================================================
# 3. Color palette
# ============================================================

color_palette = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
    "#86BCB6",
    "#FABFD2",
    "#8CD17D",
    "#B6992D",
    "#499894"
]

def get_colors(n):
    return [color_palette[i % len(color_palette)] for i in range(n)]


# ============================================================
# 4. Upload Excel file
# ============================================================

print("請上傳 Warehouse_515SKU_Inventory_Anomaly_AI_Clean.xlsx")
uploaded = files.upload()

file_name = list(uploaded.keys())[0]

print("已上傳檔案：", file_name)


# ============================================================
# 5. Read main SKU sheet only
# ============================================================
# Do not read teacher answer sheet.
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

print("主資料表讀取完成。")
print("偵測到的表頭列位置：", header_row)
print("資料筆數：", len(df))
print("欄位數量：", len(df.columns))


# ============================================================
# 6. Check required columns
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
    "目前距離主要作業區m",
    "備註"
]

missing_cols = [col for col in required_cols if col not in df.columns]

if len(missing_cols) > 0:
    raise ValueError(f"Excel 缺少必要欄位：{missing_cols}")

print("必要欄位檢查完成。")


# ============================================================
# 7. Basic data cleaning
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
    "目前儲位區",
    "備註"
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

df["季節月份"] = df["季節月份"].fillna("非季節")

df.loc[df["是否季節性"] == "否", "季節月份"] = "非季節"
df.loc[df["是否季節性"] == "否", "季節係數"] = 1.0

print("資料清理完成。")


# ============================================================
# 8. Convert yes/no columns to 0/1
# ============================================================

yes_no_cols = [
    "是否液體",
    "是否易碎",
    "是否季節性",
    "是否棧板管理"
]

for col in yes_no_cols:
    df[col + "_flag"] = df[col].map({"是": 1, "否": 0}).fillna(0).astype(int)


# ============================================================
# 9. Convert volume and weight levels to scores
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


# ============================================================
# 10. Build demand and pallet indicators
# ============================================================

df["預估下期需求箱數"] = (
    100
    + df["每日出貨箱數"] * 11.5
    + df["每日揀貨次數"] * 4.5
    + df["目前庫存箱數"] * 0.28
    + df["體積分數"] * 25
    + df["重量分數"] * 20
    + df["是否液體_flag"] * 60
    + df["是否易碎_flag"] * 45
    + df["是否季節性_flag"] * (df["季節係數"] - 1.0) * 900
)

df["預估下期需求箱數"] = df["預估下期需求箱數"].clip(lower=1).round(0)

df["預估日均需求箱數"] = (df["預估下期需求箱數"] / 30).round(2)


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


df["補貨週期天數"] = df.apply(calc_replenishment_days, axis=1)
df["安全庫存天數"] = df.apply(calc_safety_stock_days, axis=1)

df["建議存放箱數"] = (
    df["預估日均需求箱數"]
    * (df["補貨週期天數"] + df["安全庫存天數"])
).clip(lower=1).round(0)

df["目前棧板數"] = np.ceil(
    df["目前庫存箱數"] / df["每棧板可放箱數"]
).replace([np.inf, -np.inf], 0).fillna(0).astype(int)

df["模型建議棧板數"] = np.ceil(
    df["建議存放箱數"] / df["每棧板可放箱數"]
).replace([np.inf, -np.inf], 0).fillna(0).astype(int)

df["棧板增減"] = df["模型建議棧板數"] - df["目前棧板數"]

print("需求與棧板基礎指標建立完成。")


# ============================================================
# 11. Build anomaly detection indicators
# ============================================================

df["庫存覆蓋天數"] = (
    df["目前庫存箱數"] / df["預估日均需求箱數"]
).replace([np.inf, -np.inf], np.nan).fillna(0).round(2)

df["庫存差異量"] = (
    df["目前庫存箱數"] - df["建議存放箱數"]
).round(0)

df["庫存差異率"] = (
    df["庫存差異量"] / df["建議存放箱數"]
).replace([np.inf, -np.inf], 0).fillna(0).round(3)

df["棧板差異率"] = (
    (df["目前棧板數"] - df["模型建議棧板數"])
    / df["模型建議棧板數"].replace(0, 1)
).replace([np.inf, -np.inf], 0).fillna(0).round(3)

df["揀貨距離負荷"] = (
    df["每日揀貨次數"] * df["目前距離主要作業區m"]
).round(0)

df["高頻遠距指標"] = (
    np.where(df["每日揀貨次數"] >= 70, 1, 0)
    * df["目前距離主要作業區m"]
).round(2)

df["低頻近端指標"] = (
    np.where(df["每日揀貨次數"] <= 10, 1, 0)
    * np.where(df["目前距離主要作業區m"] <= 20, 1, 0)
).astype(int)

df["季節備貨壓力"] = (
    df["是否季節性_flag"]
    * df["季節係數"]
    * np.maximum(0, -df["庫存差異率"])
).round(3)

df["液體重物風險"] = (
    df["是否液體_flag"]
    * df["重量分數"]
    * df["目前距離主要作業區m"]
).round(2)

print("庫存異常偵測指標建立完成。")


# ============================================================
# 12. Chinese table preview
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
    "預估下期需求箱數",
    "預估日均需求箱數",
    "建議存放箱數",
    "目前棧板數",
    "模型建議棧板數",
    "棧板增減",
    "庫存覆蓋天數",
    "目前距離主要作業區m"
]

print("資料與衍生指標預覽：")
display(df[preview_cols].head(10))


# ============================================================
# 13. Isolation Forest input features
# ============================================================

numeric_features = [
    "每日出貨箱數",
    "每日揀貨次數",
    "目前庫存箱數",
    "每棧板可放箱數",
    "目前距離主要作業區m",
    "體積分數",
    "重量分數",
    "是否液體_flag",
    "是否易碎_flag",
    "是否季節性_flag",
    "季節係數",
    "預估下期需求箱數",
    "預估日均需求箱數",
    "建議存放箱數",
    "目前棧板數",
    "模型建議棧板數",
    "棧板增減",
    "庫存覆蓋天數",
    "庫存差異量",
    "庫存差異率",
    "棧板差異率",
    "揀貨距離負荷",
    "高頻遠距指標",
    "低頻近端指標",
    "季節備貨壓力",
    "液體重物風險"
]

categorical_features = [
    "商品類別",
    "KMeans群組",
    "目前儲位區",
    "體積等級",
    "重量等級"
]

X = df[numeric_features + categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# ============================================================
# 14. Train Isolation Forest
# ============================================================

contamination_rate = 0.03

iso_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", IsolationForest(
            n_estimators=300,
            contamination=contamination_rate,
            random_state=42
        ))
    ]
)

iso_model.fit(X)

X_transformed = iso_model.named_steps["preprocessor"].transform(X)

df["異常判斷值"] = iso_model.predict(X)

df["異常判斷"] = np.where(
    df["異常判斷值"] == -1,
    "異常",
    "正常"
)

# decision_function:
# score < 0 means anomaly
# score >= 0 means normal
df["異常分數"] = iso_model.named_steps["model"].decision_function(X_transformed)
df["異常分數"] = df["異常分數"].round(4)

print("Isolation Forest 異常偵測完成。")
print("Decision threshold: anomaly score < 0")


# ============================================================
# 15. Rule-based anomaly type explanation
# ============================================================

def classify_anomaly(row):
    anomaly_types = []

    if row["庫存覆蓋天數"] <= 2 and row["預估日均需求箱數"] >= 20:
        anomaly_types.append("缺貨風險")

    if row["庫存覆蓋天數"] >= 45 and row["每日揀貨次數"] <= 20:
        anomaly_types.append("庫存積壓")

    if row["棧板增減"] >= 5:
        anomaly_types.append("棧板不足")

    if row["棧板增減"] <= -5:
        anomaly_types.append("棧板過多")

    if row["每日揀貨次數"] >= 70 and row["目前距離主要作業區m"] >= 50:
        anomaly_types.append("高頻遠距")

    if row["每日揀貨次數"] <= 10 and row["目前距離主要作業區m"] <= 20:
        anomaly_types.append("低頻占近端")

    if row["是否季節性_flag"] == 1 and row["季節係數"] >= 1.8 and row["棧板增減"] >= 5:
        anomaly_types.append("季節備貨不足")

    if row["是否液體_flag"] == 1 and row["重量等級"] == "重" and row["目前距離主要作業區m"] >= 45:
        anomaly_types.append("液體重物配置不佳")

    if len(anomaly_types) == 0:
        anomaly_types.append("綜合異常")

    return "、".join(anomaly_types)


def generate_reason(row):
    reasons = []

    if "缺貨風險" in row["異常類型"]:
        reasons.append("目前庫存覆蓋天數偏低，可能無法支撐預估需求")

    if "庫存積壓" in row["異常類型"]:
        reasons.append("目前庫存覆蓋天數偏高，且揀貨頻率偏低")

    if "棧板不足" in row["異常類型"]:
        reasons.append("模型建議棧板數明顯高於目前棧板數")

    if "棧板過多" in row["異常類型"]:
        reasons.append("目前棧板數明顯高於模型建議棧板數")

    if "高頻遠距" in row["異常類型"]:
        reasons.append("高頻商品距離主要作業區過遠")

    if "低頻占近端" in row["異常類型"]:
        reasons.append("低頻商品占用近端儲位")

    if "季節備貨不足" in row["異常類型"]:
        reasons.append("季節性商品在旺季前棧板需求增加")

    if "液體重物配置不佳" in row["異常類型"]:
        reasons.append("液體重物商品距離較遠，搬運風險較高")

    if len(reasons) == 0:
        reasons.append("多項庫存、需求或儲位指標與一般 SKU 差異較大")

    return "；".join(reasons)


def generate_suggestion(row):
    suggestions = []

    if "缺貨風險" in row["異常類型"]:
        suggestions.append("優先補貨")

    if "庫存積壓" in row["異常類型"]:
        suggestions.append("暫緩補貨或移至遠端儲位")

    if "棧板不足" in row["異常類型"]:
        suggestions.append("增加棧板位")

    if "棧板過多" in row["異常類型"]:
        suggestions.append("釋放多餘棧板空間")

    if "高頻遠距" in row["異常類型"]:
        suggestions.append("前移至近端或中段作業區")

    if "低頻占近端" in row["異常類型"]:
        suggestions.append("移至中遠端儲位")

    if "季節備貨不足" in row["異常類型"]:
        suggestions.append("旺季前補貨並移至季節彈性儲位")

    if "液體重物配置不佳" in row["異常類型"]:
        suggestions.append("調整至近端低層棧板區")

    if len(suggestions) == 0:
        suggestions.append("人工複核庫存與儲位配置")

    suggestions = list(dict.fromkeys(suggestions))

    return "；".join(suggestions)


df["異常類型"] = ""

df.loc[df["異常判斷"] == "異常", "異常類型"] = df[df["異常判斷"] == "異常"].apply(
    classify_anomaly,
    axis=1
)

df["異常原因"] = ""

df.loc[df["異常判斷"] == "異常", "異常原因"] = df[df["異常判斷"] == "異常"].apply(
    generate_reason,
    axis=1
)

df["處理建議"] = ""

df.loc[df["異常判斷"] == "異常", "處理建議"] = df[df["異常判斷"] == "異常"].apply(
    generate_suggestion,
    axis=1
)


# ============================================================
# 16. Overall anomaly summary
# ============================================================

summary_count = df["異常判斷"].value_counts().reset_index()
summary_count.columns = ["異常判斷", "SKU數量"]

total_sku = len(df)
anomaly_count = (df["異常判斷"] == "異常").sum()
normal_count = (df["異常判斷"] == "正常").sum()
anomaly_ratio = anomaly_count / total_sku * 100

print("整體異常偵測結果：")
print("全部 SKU 數量：", total_sku)
print("正常 SKU 數量：", normal_count)
print("異常 SKU 數量：", anomaly_count)
print("異常比例：", round(anomaly_ratio, 2), "%")

display(summary_count)


# Chart uses English labels only
summary_plot = summary_count.copy()
summary_plot["Detection_Result_EN"] = summary_plot["異常判斷"].map({
    "正常": "Normal",
    "異常": "Anomaly"
})

plt.figure(figsize=(7, 5))

plt.bar(
    summary_plot["Detection_Result_EN"],
    summary_plot["SKU數量"],
    color=get_colors(len(summary_plot))
)

for i, v in enumerate(summary_plot["SKU數量"]):
    plt.text(i, v + 2, str(v), ha="center")

plt.xlabel("Detection Result")
plt.ylabel("SKU Count")
plt.title("Normal vs Anomaly SKU Count")
plt.grid(axis="y", alpha=0.3)
plt.show()


# ============================================================
# 17. Anomaly score trend line
# ============================================================
# This is the Isolation Forest anomaly score trend.
# The red dashed line is the decision threshold.
# Score < 0 means anomaly.
# ============================================================

score_trend_df = df.sort_values("異常分數", ascending=True).reset_index(drop=True)
score_trend_df["Rank"] = score_trend_df.index + 1

plt.figure(figsize=(11, 5))

plt.plot(
    score_trend_df["Rank"],
    score_trend_df["異常分數"],
    color="#4E79A7",
    linewidth=2,
    marker="o",
    markersize=3,
    alpha=0.8
)

plt.axhline(
    0,
    linestyle="--",
    color="#E15759",
    linewidth=2,
    label="Decision Threshold"
)

plt.fill_between(
    score_trend_df["Rank"],
    score_trend_df["異常分數"],
    0,
    where=(score_trend_df["異常分數"] < 0),
    color="#E15759",
    alpha=0.18,
    label="Anomaly Zone"
)

plt.xlabel("SKU Rank by Anomaly Score")
plt.ylabel("Anomaly Score")
plt.title("Isolation Forest Anomaly Score Trend")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# 18. Anomaly score distribution
# ============================================================

plt.figure(figsize=(10, 5))

plt.hist(
    df["異常分數"],
    bins=30,
    color="#4E79A7",
    edgecolor="white",
    alpha=0.85
)

plt.axvline(
    0,
    linestyle="--",
    color="#E15759",
    linewidth=2,
    label="Decision Threshold"
)

plt.xlabel("Anomaly Score")
plt.ylabel("SKU Count")
plt.title("Anomaly Score Distribution")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.show()


# ============================================================
# 19. Anomaly type statistics
# ============================================================

anomaly_df = df[df["異常判斷"] == "異常"].copy()

type_rows = []

for _, row in anomaly_df.iterrows():
    types = str(row["異常類型"]).split("、")
    for t in types:
        type_rows.append(t)

anomaly_type_count = pd.Series(type_rows).value_counts().reset_index()
anomaly_type_count.columns = ["異常類型", "SKU數量"]

print("異常類型統計：")
display(anomaly_type_count)


# English mapping for chart only
anomaly_type_en_map = {
    "缺貨風險": "Stockout Risk",
    "庫存積壓": "Overstock",
    "棧板不足": "Pallet Shortage",
    "棧板過多": "Excess Pallets",
    "高頻遠距": "High-Frequency Far Storage",
    "低頻占近端": "Low-Frequency Near Storage",
    "季節備貨不足": "Seasonal Stock Risk",
    "液體重物配置不佳": "Liquid/Heavy Misplacement",
    "綜合異常": "Mixed Anomaly"
}

anomaly_type_plot = anomaly_type_count.copy()
anomaly_type_plot["Anomaly_Type_EN"] = anomaly_type_plot["異常類型"].map(anomaly_type_en_map)

plt.figure(figsize=(11, 5))

plt.barh(
    anomaly_type_plot["Anomaly_Type_EN"],
    anomaly_type_plot["SKU數量"],
    color=get_colors(len(anomaly_type_plot))
)

for i, v in enumerate(anomaly_type_plot["SKU數量"]):
    plt.text(v + 0.1, i, str(v), va="center")

plt.xlabel("SKU Count")
plt.ylabel("Anomaly Type")
plt.title("Anomaly Type Distribution")
plt.grid(axis="x", alpha=0.3)
plt.show()


# ============================================================
# 20. Top 20 anomaly SKU table and chart
# ============================================================

top_anomaly = anomaly_df.sort_values("異常分數", ascending=True).head(20)

top_cols = [
    "SKU編號",
    "商品名稱",
    "商品類別",
    "每日出貨箱數",
    "每日揀貨次數",
    "是否季節性",
    "季節月份",
    "季節係數",
    "目前庫存箱數",
    "預估下期需求箱數",
    "預估日均需求箱數",
    "建議存放箱數",
    "庫存覆蓋天數",
    "目前棧板數",
    "模型建議棧板數",
    "棧板增減",
    "目前儲位區",
    "目前距離主要作業區m",
    "異常分數",
    "異常類型",
    "異常原因",
    "處理建議"
]

print("Top 20 異常 SKU 清單：")
display(top_anomaly[top_cols])


top_plot_df = top_anomaly.copy()
top_plot_df["SKU_Label"] = top_plot_df["SKU編號"].astype(str)

top_plot_df = top_plot_df.sort_values("異常分數", ascending=True)

plt.figure(figsize=(10, 7))

plt.barh(
    top_plot_df["SKU_Label"],
    top_plot_df["異常分數"],
    color=get_colors(len(top_plot_df))
)

plt.axvline(
    0,
    linestyle="--",
    color="#E15759",
    linewidth=2,
    label="Decision Threshold"
)

plt.xlabel("Anomaly Score")
plt.ylabel("SKU")
plt.title("Top 20 Most Anomalous SKUs")
plt.legend()
plt.grid(axis="x", alpha=0.3)
plt.show()


# ============================================================
# 21. Inventory coverage distribution
# ============================================================

plt.figure(figsize=(10, 5))

plt.hist(
    df["庫存覆蓋天數"],
    bins=30,
    color="#4E79A7",
    edgecolor="white"
)

plt.xlabel("Inventory Coverage Days")
plt.ylabel("SKU Count")
plt.title("Inventory Coverage Days Distribution")
plt.grid(axis="y", alpha=0.3)
plt.show()


# ============================================================
# 22. Inventory gap ratio vs estimated daily demand
# ============================================================

plot_df = df.copy()

normal_plot = plot_df[plot_df["異常判斷"] == "正常"]
anomaly_plot = plot_df[plot_df["異常判斷"] == "異常"]

plt.figure(figsize=(9, 6))

plt.scatter(
    normal_plot["預估日均需求箱數"],
    normal_plot["庫存差異率"],
    alpha=0.55,
    label="Normal",
    color="#4E79A7"
)

plt.scatter(
    anomaly_plot["預估日均需求箱數"],
    anomaly_plot["庫存差異率"],
    alpha=0.85,
    label="Anomaly",
    color="#E15759",
    edgecolors="white",
    linewidths=0.6
)

plt.axhline(
    0,
    linestyle="--",
    color="#333333",
    linewidth=1
)

plt.xlabel("Estimated Daily Demand")
plt.ylabel("Inventory Gap Ratio")
plt.title("Inventory Gap Ratio vs Estimated Daily Demand")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# 23. Picking frequency vs storage distance
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    normal_plot["每日揀貨次數"],
    normal_plot["目前距離主要作業區m"],
    alpha=0.55,
    label="Normal",
    color="#59A14F"
)

plt.scatter(
    anomaly_plot["每日揀貨次數"],
    anomaly_plot["目前距離主要作業區m"],
    alpha=0.85,
    label="Anomaly",
    color="#E15759",
    edgecolors="white",
    linewidths=0.6
)

plt.xlabel("Daily Picking Count")
plt.ylabel("Distance to Main Operation Area")
plt.title("Picking Frequency vs Storage Distance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ============================================================
# 24. Export Excel result
# ============================================================

output_file = "Warehouse_515SKU_IsolationForest_Anomaly_Result_ENCharts.xlsx"

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
    "是否棧板管理",
    "目前儲位區",
    "目前距離主要作業區m",
    "備註",
    "預估下期需求箱數",
    "預估日均需求箱數",
    "補貨週期天數",
    "安全庫存天數",
    "建議存放箱數",
    "目前棧板數",
    "模型建議棧板數",
    "棧板增減",
    "庫存覆蓋天數",
    "庫存差異量",
    "庫存差異率",
    "棧板差異率",
    "揀貨距離負荷",
    "高頻遠距指標",
    "低頻近端指標",
    "季節備貨壓力",
    "液體重物風險",
    "異常分數",
    "異常判斷",
    "異常類型",
    "異常原因",
    "處理建議"
]

model_summary = pd.DataFrame({
    "項目": [
        "全部 SKU 數量",
        "正常 SKU 數量",
        "異常 SKU 數量",
        "異常比例(%)",
        "Isolation Forest contamination",
        "異常分數門檻",
        "偵測方式",
        "是否使用異常答案欄位"
    ],
    "數值": [
        total_sku,
        normal_count,
        anomaly_count,
        round(anomaly_ratio, 2),
        contamination_rate,
        "異常分數 < 0",
        "無監督式異常偵測",
        "否"
    ],
    "說明": [
        "主資料表中的全部 SKU 數量",
        "模型判定為正常的 SKU 數量",
        "模型判定為異常的 SKU 數量",
        "異常 SKU 佔全部 SKU 的比例",
        "設定模型預期約 3% 樣本為異常",
        "Isolation Forest decision_function 的判斷門檻",
        "不事先告訴模型哪些 SKU 異常",
        "模型未使用教師用異常答案表"
    ]
})

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df[result_cols].to_excel(writer, sheet_name="異常偵測結果", index=False)
    anomaly_df[top_cols].to_excel(writer, sheet_name="異常SKU明細", index=False)
    summary_count.to_excel(writer, sheet_name="正常異常統計", index=False)
    anomaly_type_count.to_excel(writer, sheet_name="異常類型統計", index=False)
    score_trend_df[
        ["Rank", "SKU編號", "商品名稱", "異常分數", "異常判斷", "異常類型"]
    ].to_excel(writer, sheet_name="異常分數排序", index=False)
    model_summary.to_excel(writer, sheet_name="模型摘要", index=False)

print("Excel 結果檔案已產生：", output_file)

files.download(output_file)


# ============================================================
# 25. Teaching summary
# ============================================================

print("本案例教學重點：")
print("1. Isolation Forest 使用無監督式學習，不讀取教師用異常答案表。")
print("2. 異常比例設定為 3%，515 筆資料約抓出 15 筆異常 SKU。")
print("3. 異常分數小於 0 的 SKU 會被判定為異常。")
print("4. Anomaly Score Trend 圖可視覺化顯示異常分數與判斷門檻。")
print("5. 圖表已全部改為英文，表格仍維持中文，避免 Matplotlib 中文字型警告。")
print("6. 異常 SKU 會再用規則轉成缺貨風險、庫存積壓、棧板不足、高頻遠距等管理語言。")
