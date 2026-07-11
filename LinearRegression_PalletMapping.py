# ============================================================
# 主題： Linear Regression + 補貨週期 + 安全庫存
# 作者：國立雲林科技大學電機系 林家仁
# 線性回歸預測月需求
# 月需求換算日均需求
# 依補貨週期與安全庫存估算建議存放量
# 再換算合理棧板數
# ==========================================

# ------------------------------------------------------------
# 0. 安裝與匯入套件
# ------------------------------------------------------------

!pip install openpyxl scikit-learn matplotlib pandas numpy -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import re

from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------
# 0-1. Matplotlib 設定
# ------------------------------------------------------------
# 圖表全部使用英文，避免中文字型問題

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# ------------------------------------------------------------
# 0-2. 工具函數
# ------------------------------------------------------------

def safe_ascii(text):
    """
    將圖表標籤轉成安全英文 / ASCII，避免 matplotlib 中文亂碼。
    """
    text = str(text)

    replace_map = {
        "商品類別_": "Category_",
        "KMeans群組_": "KMeans_Cluster_",
        "目前儲位區_": "Current_Area_",

        "食品類": "Food",
        "飲品類": "Beverage",
        "紙品類": "Paper_Goods",
        "清潔用品": "Cleaning",
        "日用品": "Daily_Goods",
        "季節商品": "Seasonal_Goods",
        "家用品": "Household",
        "寵物用品": "Pet_Goods",
        "耗材類": "Consumables",
        "米糧類": "Rice_Grain",

        "A區": "Area_A",
        "K區": "Area_K",
        "1區": "Area_1",
        "2區": "Area_2",
        "3區": "Area_3",
        "近端區": "Near_Area",
        "中段區": "Middle_Area",
        "遠端區": "Far_Area",
        "彈性區": "Flexible_Area",

        "每日出貨箱數": "Daily_Shipment_Boxes",
        "每日揀貨次數": "Daily_Picking_Frequency",
        "目前庫存箱數": "Current_Inventory_Boxes",
        "體積分數": "Volume_Score",
        "重量分數": "Weight_Score",
        "是否液體_flag": "Liquid_Flag",
        "是否易碎_flag": "Fragile_Flag",
        "是否季節性_flag": "Seasonal_Flag",
        "季節係數": "Seasonal_Coefficient",
        "目前距離主要作業區m": "Distance_to_Main_Area"
    }

    for zh, en in replace_map.items():
        text = text.replace(zh, en)

    text = text.encode("ascii", errors="ignore").decode("ascii")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_")

    if text == "":
        text = "Unknown"

    return text


def safe_mape(y_true, y_pred):
    """
    避免 y_true = 0 造成 MAPE 除以 0。
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = np.where(y_true == 0, 1, y_true)
    return np.mean(np.abs((y_true - y_pred) / denominator)) * 100


# ------------------------------------------------------------
# 1. 上傳 Excel
# ------------------------------------------------------------

print("請上傳檔案：Warehouse_500SKU_Pallet_Mapping_v3.xlsx")
uploaded = files.upload()

file_name = list(uploaded.keys())[0]
print("已上傳檔案：", file_name)


# ------------------------------------------------------------
# 2. 讀取 Excel，並自動偵測真正欄位列
# ------------------------------------------------------------

sheet_name = "500件商品"

raw_df = pd.read_excel(
    file_name,
    sheet_name=sheet_name,
    header=None
)

header_row = None

for i in range(len(raw_df)):
    row_values = raw_df.iloc[i].astype(str).str.strip().tolist()
    if "SKU編號" in row_values:
        header_row = i
        break

if header_row is None:
    raise ValueError("找不到真正欄位列，請確認 Excel 中是否有「SKU編號」。")

print(f"已偵測到真正欄位列：Excel 第 {header_row + 1} 列")

df = pd.read_excel(
    file_name,
    sheet_name=sheet_name,
    header=header_row
)

df.columns = df.columns.astype(str).str.strip()

# 移除空白列與 Unnamed 欄位
df = df.dropna(how="all").reset_index(drop=True)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

print("資料讀取完成")
print("資料筆數：", len(df))
print("欄位數量：", len(df.columns))

display(df.head(10))


# ------------------------------------------------------------
# 3. 檢查必要欄位
# ------------------------------------------------------------

required_columns = [
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

missing_cols = [col for col in required_columns if col not in df.columns]

if missing_cols:
    print("目前讀到的欄位：")
    print(df.columns.tolist())
    raise ValueError(f"Excel 缺少必要欄位：{missing_cols}")
else:
    print("必要欄位檢查完成。")


# ------------------------------------------------------------
# 4. 資料前處理
# ------------------------------------------------------------

yes_no_cols = [
    "是否液體",
    "是否易碎",
    "是否季節性",
    "是否棧板管理"
]

for col in yes_no_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col + "_flag"] = df[col].map({
        "是": 1,
        "否": 0,
        "Y": 1,
        "N": 0,
        "yes": 1,
        "no": 0,
        "True": 1,
        "False": 0,
        "TRUE": 1,
        "FALSE": 0
    }).fillna(0).astype(int)

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

df["體積分數"] = df["體積等級"].astype(str).str.strip().map(volume_map).fillna(2)
df["重量分數"] = df["重量等級"].astype(str).str.strip().map(weight_map).fillna(2)

numeric_cols = [
    "每日出貨箱數",
    "每日揀貨次數",
    "季節係數",
    "目前庫存箱數",
    "每棧板可放箱數",
    "目前距離主要作業區m",
    "體積分數",
    "重量分數"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# 避免除以 0
df["每棧板可放箱數"] = df["每棧板可放箱數"].replace(0, np.nan)

print("資料前處理完成。")


# ------------------------------------------------------------
# 5. 建立教學用目標值：下期實際需求箱數
# ------------------------------------------------------------
# 說明：
# 目前 Excel 沒有真實歷史銷售資料，所以先建立教學用目標。
# 實務上應改成讀取真實「下月實際需求箱數」。
# ------------------------------------------------------------

np.random.seed(42)

df["下期實際需求箱數"] = (
    df["每日出貨箱數"] * 12
    + df["每日揀貨次數"] * 4
    + df["目前庫存箱數"] * 0.35
    + df["體積分數"] * 10
    + df["重量分數"] * 8
)

# 季節性商品需求放大
df["下期實際需求箱數"] = np.where(
    df["是否季節性_flag"] == 1,
    df["下期實際需求箱數"] * df["季節係數"],
    df["下期實際需求箱數"]
)

# 液體商品略微放大
df["下期實際需求箱數"] = np.where(
    df["是否液體_flag"] == 1,
    df["下期實際需求箱數"] * 1.08,
    df["下期實際需求箱數"]
)

# 重物商品略微放大
df["下期實際需求箱數"] = np.where(
    df["重量等級"].astype(str).str.strip() == "重",
    df["下期實際需求箱數"] * 1.05,
    df["下期實際需求箱數"]
)

# 加入隨機波動
noise = np.random.normal(loc=0, scale=40, size=len(df))
df["下期實際需求箱數"] = df["下期實際需求箱數"] + noise

df["下期實際需求箱數"] = df["下期實際需求箱數"].clip(lower=0).round(0)

print("已建立教學用目標欄位：下期實際需求箱數")

display(df[[
    "SKU編號",
    "商品名稱",
    "商品類別",
    "每日出貨箱數",
    "每日揀貨次數",
    "是否季節性",
    "季節係數",
    "目前庫存箱數",
    "下期實際需求箱數"
]].head(10))


# ------------------------------------------------------------
# 6. 篩選棧板管理商品
# ------------------------------------------------------------

df_pallet = df[df["是否棧板管理_flag"] == 1].copy()

print("全部 SKU 數量：", len(df))
print("棧板管理 SKU 數量：", len(df_pallet))
print("非棧板管理 SKU 數量：", len(df) - len(df_pallet))

if len(df_pallet) < 10:
    raise ValueError("棧板管理 SKU 數量太少，無法建立有效模型。")


# ------------------------------------------------------------
# 7. 設定模型輸入 X 與目標 y
# ------------------------------------------------------------

feature_cols_numeric = [
    "每日出貨箱數",
    "每日揀貨次數",
    "目前庫存箱數",
    "體積分數",
    "重量分數",
    "是否液體_flag",
    "是否易碎_flag",
    "是否季節性_flag",
    "季節係數",
    "目前距離主要作業區m"
]

feature_cols_category = [
    "商品類別",
    "KMeans群組",
    "目前儲位區"
]

target_col = "下期實際需求箱數"

X = df_pallet[feature_cols_numeric + feature_cols_category]
y = df_pallet[target_col]


# ------------------------------------------------------------
# 8. 建立線性回歸 Pipeline
# ------------------------------------------------------------

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols_category),
        ("num", "passthrough", feature_cols_numeric)
    ]
)

linear_model = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("regressor", LinearRegression())
    ]
)


# ------------------------------------------------------------
# 9. 訓練集 / 測試集切分
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("訓練資料筆數：", len(X_train))
print("測試資料筆數：", len(X_test))


# ------------------------------------------------------------
# 10. 訓練線性回歸模型
# ------------------------------------------------------------

linear_model.fit(X_train, y_train)

print("線性回歸模型訓練完成。")


# ------------------------------------------------------------
# 11. 模型評估
# ------------------------------------------------------------

y_pred = linear_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = safe_mape(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

eval_df = pd.DataFrame({
    "評估指標": ["MAE", "RMSE", "MAPE", "R2"],
    "數值": [round(mae, 2), round(rmse, 2), round(mape, 2), round(r2, 2)],
    "說明": [
        "平均絕對誤差，代表平均預測誤差箱數",
        "均方根誤差，對大誤差較敏感",
        "平均百分比誤差",
        "模型解釋能力，越接近 1 越好"
    ]
})

print("========== 線性回歸模型評估 ==========")
display(eval_df)


# ------------------------------------------------------------
# 12. 圖 1：Actual vs Predicted
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.75,
    s=60,
    c="#2E86DE",
    edgecolors="black",
    linewidths=0.5
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    color="#E74C3C",
    linewidth=2,
    label="Ideal Prediction Line"
)

plt.xlabel("Actual Demand Boxes")
plt.ylabel("Predicted Demand Boxes")
plt.title("Linear Regression: Actual vs Predicted Demand")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# ------------------------------------------------------------
# 13. 圖 2：Residual Plot
# ------------------------------------------------------------

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.75,
    s=60,
    c=np.where(residuals >= 0, "#27AE60", "#E74C3C"),
    edgecolors="black",
    linewidths=0.5
)

plt.axhline(
    y=0,
    linestyle="--",
    color="black",
    linewidth=2
)

plt.xlabel("Predicted Demand Boxes")
plt.ylabel("Residuals: Actual - Predicted")
plt.title("Residual Analysis Plot")
plt.grid(True, alpha=0.3)
plt.show()


# ------------------------------------------------------------
# 14. 回歸係數分析
# ------------------------------------------------------------

cat_encoder = linear_model.named_steps["preprocess"].named_transformers_["cat"]
cat_feature_names = cat_encoder.get_feature_names_out(feature_cols_category)

all_feature_names = list(cat_feature_names) + feature_cols_numeric
coefficients = linear_model.named_steps["regressor"].coef_

coef_df = pd.DataFrame({
    "特徵欄位": all_feature_names,
    "回歸係數": coefficients
})

coef_df["影響方向"] = np.where(
    coef_df["回歸係數"] >= 0,
    "正向影響",
    "負向影響"
)

coef_df["影響程度"] = coef_df["回歸係數"].abs()
coef_df["圖表用英文特徵"] = coef_df["特徵欄位"].apply(safe_ascii)

# 小數點第二位
coef_df["回歸係數"] = coef_df["回歸係數"].round(2)
coef_df["影響程度"] = coef_df["影響程度"].round(2)

coef_df = coef_df.sort_values("影響程度", ascending=False)

print("========== 回歸係數分析 Top 20 ==========")
display(coef_df.head(20))


# ------------------------------------------------------------
# 15. 圖 3：回歸係數圖
# ------------------------------------------------------------

top_coef = coef_df.head(15).copy()
top_coef = top_coef.sort_values("回歸係數")

colors = np.where(top_coef["回歸係數"] >= 0, "#3498DB", "#E67E22")

plt.figure(figsize=(10, 7))

plt.barh(
    top_coef["圖表用英文特徵"],
    top_coef["回歸係數"],
    color=colors,
    edgecolor="black"
)

plt.xlabel("Regression Coefficient")
plt.ylabel("Feature")
plt.title("Top 15 Features Affecting Monthly Demand Prediction")
plt.grid(axis="x", alpha=0.3)
plt.show()


# ------------------------------------------------------------
# 16. 預測棧板管理 SKU 的月需求
# ------------------------------------------------------------

df_pallet["模型預測月需求箱數"] = linear_model.predict(
    df_pallet[feature_cols_numeric + feature_cols_category]
)

df_pallet["模型預測月需求箱數"] = (
    df_pallet["模型預測月需求箱數"]
    .clip(lower=0)
    .round(0)
)

print("========== 棧板管理 SKU 月需求預測 ==========")

display(df_pallet[[
    "SKU編號",
    "商品名稱",
    "商品類別",
    "KMeans群組",
    "下期實際需求箱數",
    "模型預測月需求箱數"
]].head(10))


# ------------------------------------------------------------
# 17. 月需求換算日均需求
# ------------------------------------------------------------

df_pallet["預測日均需求箱數"] = (
    df_pallet["模型預測月需求箱數"] / 30
)


# ------------------------------------------------------------
# 18. 設定補貨週期與安全庫存天數
# ------------------------------------------------------------

def replenishment_days(row):
    picking = row["每日揀貨次數"]
    seasonal = row["是否季節性_flag"]
    volume = str(row["體積等級"]).strip()
    weight = str(row["重量等級"]).strip()
    liquid = row["是否液體_flag"]

    if picking >= 100:
        return 3

    if picking >= 70:
        return 4

    if picking >= 50:
        return 5

    if seasonal == 1:
        return 10

    if volume == "大" or weight == "重" or liquid == 1:
        return 7

    return 7


def safety_stock_days(row):
    picking = row["每日揀貨次數"]
    seasonal = row["是否季節性_flag"]

    if seasonal == 1:
        return 5

    if picking >= 70:
        return 3

    if picking >= 50:
        return 2

    return 2


df_pallet["補貨週期天數"] = df_pallet.apply(replenishment_days, axis=1)
df_pallet["安全庫存天數"] = df_pallet.apply(safety_stock_days, axis=1)


# ------------------------------------------------------------
# 19. 計算建議存放箱數
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 20. 換算目前棧板數與模型建議棧板數
# ------------------------------------------------------------

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

print("========== 修正版棧板換算結果 ==========")

display(df_pallet[[
    "SKU編號",
    "商品名稱",
    "模型預測月需求箱數",
    "預測日均需求箱數",
    "補貨週期天數",
    "安全庫存天數",
    "建議存放箱數",
    "目前棧板數",
    "模型建議棧板數",
    "棧板增減"
]].head(20))


# ------------------------------------------------------------
# 21. 儲位建議與前移判斷
# ------------------------------------------------------------

def recommend_storage(row):
    picking = row["每日揀貨次數"]
    delta = row["棧板增減"]
    seasonal = row["是否季節性_flag"]
    weight = str(row["重量等級"]).strip()
    volume = str(row["體積等級"]).strip()
    liquid = row["是否液體_flag"]

    if seasonal == 1 and delta >= 1:
        return "季節彈性儲位"

    if picking >= 70 and (weight == "重" or liquid == 1):
        return "近端低層棧板區"

    if picking >= 70 and volume == "大":
        return "近端大貨位"

    if picking >= 40 and volume == "大":
        return "中段大貨位"

    if delta <= -3:
        return "遠端棧板區"

    return "中段棧板區"


def forward_move(row):
    picking = row["每日揀貨次數"]
    delta = row["棧板增減"]
    seasonal = row["是否季節性_flag"]
    distance = row["目前距離主要作業區m"]

    if seasonal == 1 and delta >= 2:
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

print("========== 儲位建議結果 ==========")

display(df_pallet[[
    "SKU編號",
    "商品名稱",
    "每日揀貨次數",
    "目前距離主要作業區m",
    "目前棧板數",
    "模型建議棧板數",
    "棧板增減",
    "建議儲位區",
    "前移判斷",
    "前移原因"
]].head(20))


# ------------------------------------------------------------
# 22. 圖 4：棧板增減區間分布
# ------------------------------------------------------------

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

change_group_summary = (
    df_pallet["棧板增減區間"]
    .value_counts()
    .reindex(labels)
    .fillna(0)
)

plt.figure(figsize=(11, 5))

colors = [
    "#4E79A7", "#76B7B2", "#59A14F",
    "#BAB0AC",
    "#F28E2B", "#E15759", "#B07AA1", "#FF9DA7", "#9C755F"
]

bars = plt.bar(
    change_group_summary.index.astype(str),
    change_group_summary.values,
    color=colors,
    edgecolor="black"
)

plt.xlabel("Pallet Change Range")
plt.ylabel("Number of SKUs")
plt.title("Predicted Pallet Change by Range")
plt.grid(axis="y", alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        int(height),
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.show()


# ------------------------------------------------------------
# 23. 圖 5：建議儲位區統計
# ------------------------------------------------------------

storage_map_for_chart = {
    "季節彈性儲位": "Seasonal Flexible Area",
    "近端低層棧板區": "Near Low-level Pallet Area",
    "近端大貨位": "Near Bulky Goods Area",
    "中段大貨位": "Middle Bulky Goods Area",
    "遠端棧板區": "Far Pallet Area",
    "中段棧板區": "Middle Pallet Area"
}

df_pallet["圖表用建議儲位區"] = (
    df_pallet["建議儲位區"]
    .map(storage_map_for_chart)
    .fillna("Other Area")
)

storage_summary = (
    df_pallet["圖表用建議儲位區"]
    .value_counts()
    .sort_values(ascending=True)
)

plt.figure(figsize=(11, 6))

bars = plt.barh(
    storage_summary.index,
    storage_summary.values,
    color=plt.cm.Set2(np.linspace(0, 1, len(storage_summary))),
    edgecolor="black"
)

plt.xlabel("Number of SKUs")
plt.ylabel("Recommended Storage Area")
plt.title("Recommended Storage Area Summary")
plt.grid(axis="x", alpha=0.3)

for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 1,
        bar.get_y() + bar.get_height() / 2,
        int(width),
        va="center",
        fontsize=10
    )

plt.show()


# ------------------------------------------------------------
# 24. 圖 6：前移判斷
# ------------------------------------------------------------

move_map_for_chart = {
    "建議前移": "Move Forward",
    "維持現況": "Keep Current"
}

df_pallet["圖表用前移判斷"] = (
    df_pallet["前移判斷"]
    .map(move_map_for_chart)
    .fillna("Unknown")
)

move_summary = df_pallet["圖表用前移判斷"].value_counts()

move_order = ["Move Forward", "Keep Current"]
move_summary = move_summary.reindex(move_order).fillna(0)

plt.figure(figsize=(7, 5))

move_colors = ["#E15759", "#76B7B2"]

bars = plt.bar(
    move_summary.index,
    move_summary.values,
    color=move_colors,
    edgecolor="black"
)

plt.xlabel("Forward Move Decision")
plt.ylabel("Number of SKUs")
plt.title("Forward Move Decision Summary")
plt.grid(axis="y", alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        int(height),
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.show()


# ------------------------------------------------------------
# 25. 圖 7：Top 10 棧板增加 SKU
# ------------------------------------------------------------
# 修正重點：
# 原本 Top 20 直立長條圖太擠。
# 改成 Top 10 水平長條圖，閱讀性較好。
# ------------------------------------------------------------

top_increase = (
    df_pallet
    .sort_values("棧板增減", ascending=False)
    .head(10)
    .copy()
)

top_increase["SKU_Label"] = (
    top_increase["SKU編號"].astype(str)
    + " | +"
    + top_increase["棧板增減"].astype(int).astype(str)
)

top_increase = top_increase.sort_values("棧板增減", ascending=True)

plt.figure(figsize=(10, 6))

bars = plt.barh(
    top_increase["SKU_Label"],
    top_increase["棧板增減"],
    color=plt.cm.tab10(np.linspace(0, 1, len(top_increase))),
    edgecolor="black"
)

plt.xlabel("Pallet Increase")
plt.ylabel("SKU ID")
plt.title("Top 10 SKUs by Recommended Pallet Increase")
plt.grid(axis="x", alpha=0.3)

for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.5,
        bar.get_y() + bar.get_height() / 2,
        int(width),
        va="center",
        fontsize=10
    )

plt.show()


# ------------------------------------------------------------
# 26. 優先前移 SKU
# ------------------------------------------------------------

priority_sku = df_pallet[
    df_pallet["前移判斷"] == "建議前移"
].copy()

priority_sku = priority_sku.sort_values(
    ["棧板增減", "每日揀貨次數", "目前距離主要作業區m"],
    ascending=[False, False, False]
)

print("========== 優先前移 SKU Top 20 ==========")

display(priority_sku[[
    "SKU編號",
    "商品名稱",
    "商品類別",
    "KMeans群組",
    "每日揀貨次數",
    "目前距離主要作業區m",
    "目前棧板數",
    "模型建議棧板數",
    "棧板增減",
    "建議儲位區",
    "前移判斷",
    "前移原因"
]].head(20))


# ------------------------------------------------------------
# 27. 模型摘要
# ------------------------------------------------------------

model_summary = pd.DataFrame({
    "項目": [
        "全部SKU數量",
        "棧板管理SKU數量",
        "非棧板管理SKU數量",
        "訓練資料筆數",
        "測試資料筆數",
        "MAE",
        "RMSE",
        "MAPE",
        "R2",
        "建議前移SKU數量",
        "平均目前棧板數",
        "平均模型建議棧板數",
        "平均棧板增減"
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
        round(r2, 2),
        len(priority_sku),
        round(df_pallet["目前棧板數"].mean(), 2),
        round(df_pallet["模型建議棧板數"].mean(), 2),
        round(df_pallet["棧板增減"].mean(), 2)
    ],
    "說明": [
        "Excel中全部商品數量",
        "納入本次棧板需求預測的SKU數量",
        "未納入棧板管理分析的SKU數量",
        "用於訓練線性回歸模型的資料筆數",
        "用於測試模型效果的資料筆數",
        "平均絕對誤差",
        "均方根誤差",
        "平均百分比誤差",
        "模型解釋能力",
        "依模型判斷建議前移的SKU數量",
        "每個SKU目前平均棧板數",
        "每個SKU模型建議平均棧板數",
        "模型建議與目前棧板數的平均差異"
    ]
})

print("========== 模型摘要 ==========")
display(model_summary)


# ------------------------------------------------------------
# 28. 匯出 Excel 結果
# ------------------------------------------------------------

output_file = "Warehouse_500SKU_LinearRegression_Revised_Chinese_Table_Result.xlsx"

pallet_result_cols = [
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
    "模型預測月需求箱數",
    "預測日均需求箱數",
    "補貨週期天數",
    "安全庫存天數",
    "建議存放箱數",
    "目前棧板數",
    "模型建議棧板數",
    "棧板增減",
    "棧板增減區間",
    "目前儲位區",
    "目前距離主要作業區m",
    "建議儲位區",
    "前移判斷",
    "前移原因"
]

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="500SKU_含需求目標", index=False)
    df_pallet[pallet_result_cols].to_excel(writer, sheet_name="棧板預測結果", index=False)
    eval_df.to_excel(writer, sheet_name="模型評估", index=False)
    coef_df.to_excel(writer, sheet_name="回歸係數分析", index=False)
    priority_sku[pallet_result_cols].to_excel(writer, sheet_name="優先前移SKU", index=False)
    model_summary.to_excel(writer, sheet_name="模型摘要", index=False)

print("分析結果已匯出：", output_file)

files.download(output_file)
