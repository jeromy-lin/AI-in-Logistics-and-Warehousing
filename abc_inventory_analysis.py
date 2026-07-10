# ==========================================
# 主題： ABC 分類法：倉儲 SKU 儲位分析
# 作者：國立雲林科技大學電機系 林家仁
# 倉儲 SKU 商品 ABC 分類分析
# 以某倉儲除位設計進行ABC分類 並顯示效益成果
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from IPython.display import display
from google.colab import files

# -----------------------------
# 1. 上傳 Excel 檔案
# -----------------------------
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

print("已上傳檔案：", file_name)

# -----------------------------
# 2. 自動尋找「SKU資料輸入」表頭列
# -----------------------------
raw_df = pd.read_excel(file_name, sheet_name="SKU資料輸入", header=None)

header_row = None

for i in range(len(raw_df)):
    row_values = raw_df.iloc[i].astype(str).tolist()
    if "SKU編號" in row_values:
        header_row = i
        break

if header_row is None:
    raise ValueError("找不到表頭列，請確認工作表中是否有「SKU編號」欄位。")

df = pd.read_excel(file_name, sheet_name="SKU資料輸入", header=header_row)

# 移除空白列與沒有 SKU 的資料列
df = df.dropna(how="all")
df = df.dropna(subset=["SKU編號"])

print("原始 SKU 商品資料")
display(df)

# -----------------------------
# 3. 檢查必要欄位
# -----------------------------
required_columns = [
    "SKU編號",
    "商品名稱",
    "商品類別",
    "每日出貨箱數",
    "每日揀貨次數",
    "體積等級",
    "重量等級",
    "是否液體",
    "是否易碎",
    "是否季節性",
    "目前儲位區",
    "目前距離主要作業區m",
    "備註"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(f"缺少必要欄位：{missing_columns}")

# -----------------------------
# 4. 數值欄位轉換
# -----------------------------
df["每日出貨箱數"] = pd.to_numeric(
    df["每日出貨箱數"],
    errors="coerce"
).fillna(0)

df["每日揀貨次數"] = pd.to_numeric(
    df["每日揀貨次數"],
    errors="coerce"
).fillna(0)

df["目前距離主要作業區m"] = pd.to_numeric(
    df["目前距離主要作業區m"],
    errors="coerce"
).fillna(50)

# -----------------------------
# 5. 依每日揀貨次數進行 ABC 分類
# -----------------------------
df_sorted = df.sort_values(
    by="每日揀貨次數",
    ascending=False
).reset_index(drop=True)

total_picking_count = df_sorted["每日揀貨次數"].sum()

if total_picking_count == 0:
    raise ValueError("每日揀貨次數總和為 0，無法進行 ABC 分類。")

df_sorted["出貨占比"] = df_sorted["每日揀貨次數"] / total_picking_count
df_sorted["累積出貨占比"] = df_sorted["出貨占比"].cumsum()

def abc_classification(cumulative_ratio):
    if cumulative_ratio <= 0.80:
        return "A 類"
    elif cumulative_ratio <= 0.95:
        return "B 類"
    else:
        return "C 類"

df_sorted["ABC類別"] = df_sorted["累積出貨占比"].apply(abc_classification)

# -----------------------------
# 6. 依 ABC 類別、體積、重量與商品限制產生 AI 建議儲位
# -----------------------------
def recommend_zone(row):
    abc_class = row["ABC類別"]
    volume = str(row["體積等級"])
    weight = str(row["重量等級"])
    is_liquid = str(row["是否液體"])
    is_fragile = str(row["是否易碎"])
    is_seasonal = str(row["是否季節性"])

    # A 類：高頻商品，優先靠近主要作業區
    if abc_class == "A 類":
        if weight == "重" or is_liquid == "是":
            return "近端低層棧板區"
        elif volume == "大":
            return "近端大貨位"
        else:
            return "近端一般儲位"

    # B 類：中頻商品，安排在中段區域
    elif abc_class == "B 類":
        if weight == "重" or is_liquid == "是" or is_fragile == "是":
            return "中段低層儲位"
        elif volume == "大":
            return "中段大貨位"
        else:
            return "中段一般儲位"

    # C 類：低頻商品，安排在遠端或彈性儲位
    else:
        if is_seasonal == "是":
            return "遠端彈性儲位"
        elif volume == "大":
            return "遠端大貨位"
        else:
            return "遠端或高層儲位"

df_sorted["AI建議儲位"] = df_sorted.apply(recommend_zone, axis=1)

# -----------------------------
# 7. 理論建議距離設定
# -----------------------------
def recommended_distance(zone):
    zone = str(zone)

    if "近端" in zone:
        return 15
    elif "中段" in zone:
        return 35
    elif "遠端" in zone:
        return 60
    else:
        return 50

df_sorted["理論建議距離主要作業區m"] = df_sorted["AI建議儲位"].apply(
    recommended_distance
)

# --------------------------------------------------
# 修正重點：
# 如果商品目前已經比理論建議距離更靠近作業區，
# 則不強制移遠，避免出現負的節省距離成本。
# --------------------------------------------------
df_sorted["建議距離主要作業區m"] = df_sorted.apply(
    lambda row: min(
        row["目前距離主要作業區m"],
        row["理論建議距離主要作業區m"]
    ),
    axis=1
)

def final_slotting_decision(row):
    if row["建議距離主要作業區m"] < row["目前距離主要作業區m"]:
        return f"建議調整至{row['AI建議儲位']}"
    else:
        return "目前位置可保留"

df_sorted["儲位調整建議"] = df_sorted.apply(
    final_slotting_decision,
    axis=1
)

# -----------------------------
# 8. 距離成本計算
# -----------------------------
df_sorted["改善前距離成本"] = (
    df_sorted["每日揀貨次數"] *
    df_sorted["目前距離主要作業區m"]
)

df_sorted["改善後距離成本"] = (
    df_sorted["每日揀貨次數"] *
    df_sorted["建議距離主要作業區m"]
)

df_sorted["節省距離成本"] = (
    df_sorted["改善前距離成本"] -
    df_sorted["改善後距離成本"]
)

# 避免節省距離成本出現負值
df_sorted["節省距離成本"] = df_sorted["節省距離成本"].clip(lower=0)

before_total = df_sorted["改善前距離成本"].sum()
after_total = df_sorted["改善後距離成本"].sum()
saving_total = df_sorted["節省距離成本"].sum()
saving_rate = saving_total / before_total if before_total != 0 else 0

# -----------------------------
# 9. ABC 類別顏色設定
# -----------------------------
class_colors = {
    "A 類": "#2E86DE",
    "B 類": "#F39C12",
    "C 類": "#27AE60"
}

df_sorted["類別顏色"] = df_sorted["ABC類別"].map({
    "A 類": "藍色",
    "B 類": "橘色",
    "C 類": "綠色"
})

df_sorted["Color_Code"] = df_sorted["ABC類別"].map(class_colors)

def color_abc_class(value):
    if value == "A 類":
        return "background-color: #2E86DE; color: white; font-weight: bold;"
    elif value == "B 類":
        return "background-color: #F39C12; color: white; font-weight: bold;"
    elif value == "C 類":
        return "background-color: #27AE60; color: white; font-weight: bold;"
    else:
        return ""

# -----------------------------
# 10. 顯示 ABC 分析結果表
# -----------------------------
display_columns = [
    "SKU編號",
    "商品名稱",
    "商品類別",
    "每日出貨箱數",
    "每日揀貨次數",
    "體積等級",
    "重量等級",
    "是否液體",
    "是否易碎",
    "是否季節性",
    "目前儲位區",
    "目前距離主要作業區m",
    "出貨占比",
    "累積出貨占比",
    "ABC類別",
    "類別顏色",
    "AI建議儲位",
    "理論建議距離主要作業區m",
    "建議距離主要作業區m",
    "改善前距離成本",
    "改善後距離成本",
    "節省距離成本",
    "儲位調整建議",
    "備註"
]

print("SKU 商品 ABC 分類結果")

display(
    df_sorted[display_columns]
    .style
    .map(color_abc_class, subset=["ABC類別"])
    .format({
        "出貨占比": "{:.2%}",
        "累積出貨占比": "{:.2%}",
        "目前距離主要作業區m": "{:.0f}",
        "理論建議距離主要作業區m": "{:.0f}",
        "建議距離主要作業區m": "{:.0f}",
        "改善前距離成本": "{:,.0f}",
        "改善後距離成本": "{:,.0f}",
        "節省距離成本": "{:,.0f}"
    })
)

# -----------------------------
# 11. ABC 分類摘要表
# -----------------------------
summary = df_sorted.groupby("ABC類別").agg(
    SKU數量=("SKU編號", "count"),
    每日出貨箱數總和=("每日出貨箱數", "sum"),
    每日揀貨次數總和=("每日揀貨次數", "sum"),
    出貨占比=("出貨占比", "sum"),
    平均目前距離m=("目前距離主要作業區m", "mean"),
    平均理論建議距離m=("理論建議距離主要作業區m", "mean"),
    平均實際建議距離m=("建議距離主要作業區m", "mean"),
    改善前距離成本=("改善前距離成本", "sum"),
    改善後距離成本=("改善後距離成本", "sum"),
    節省距離成本=("節省距離成本", "sum")
).reset_index()

abc_order = {
    "A 類": 1,
    "B 類": 2,
    "C 類": 3
}

summary["排序"] = summary["ABC類別"].map(abc_order)
summary = summary.sort_values("排序").drop(columns=["排序"])

print("ABC 分類摘要表")

display(
    summary
    .style
    .map(color_abc_class, subset=["ABC類別"])
    .format({
        "出貨占比": "{:.2%}",
        "平均目前距離m": "{:.1f}",
        "平均理論建議距離m": "{:.1f}",
        "平均實際建議距離m": "{:.1f}",
        "改善前距離成本": "{:,.0f}",
        "改善後距離成本": "{:,.0f}",
        "節省距離成本": "{:,.0f}"
    })
)

# -----------------------------
# 12. 整體改善結果
# -----------------------------
print("整體改善結果")
print("--------------------------------")
print(f"改善前每日總距離成本：{before_total:,.0f}")
print(f"改善後每日總距離成本：{after_total:,.0f}")
print(f"每日可節省距離成本：{saving_total:,.0f}")
print(f"改善比例：{saving_rate:.2%}")

# ==========================================================
# 以下圖表維持英文，方便放入簡報
# ==========================================================

legend_patches = [
    mpatches.Patch(color="#2E86DE", label="A Class"),
    mpatches.Patch(color="#F39C12", label="B Class"),
    mpatches.Patch(color="#27AE60", label="C Class")
]

# -----------------------------
# Chart 1: Daily picking count by SKU
# -----------------------------
plt.figure(figsize=(14, 6))

plt.bar(
    df_sorted["SKU編號"],
    df_sorted["每日揀貨次數"],
    color=df_sorted["Color_Code"]
)

plt.title("Daily Picking Count by SKU", fontsize=16)
plt.xlabel("SKU", fontsize=12)
plt.ylabel("Daily Picking Count", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(handles=legend_patches, title="ABC Class")
plt.tight_layout()
plt.show()

# -----------------------------
# Chart 2: Cumulative picking ratio
# -----------------------------
plt.figure(figsize=(14, 6))

plt.plot(
    df_sorted["SKU編號"],
    df_sorted["累積出貨占比"],
    marker="o",
    linewidth=2,
    color="#8E44AD",
    label="Cumulative Picking Ratio"
)

plt.axhline(
    0.80,
    color="#2E86DE",
    linestyle="--",
    linewidth=2,
    label="A Class Threshold: 80%"
)

plt.axhline(
    0.95,
    color="#F39C12",
    linestyle="--",
    linewidth=2,
    label="B Class Threshold: 95%"
)

plt.title("Cumulative Picking Ratio for ABC Classification", fontsize=16)
plt.xlabel("SKU", fontsize=12)
plt.ylabel("Cumulative Picking Ratio", fontsize=12)
plt.xticks(rotation=45)
plt.ylim(0, 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Chart 3: Before and after distance cost
# -----------------------------
comparison = pd.DataFrame({
    "Scenario": ["Before Re-slotting", "After Re-slotting"],
    "Total_Daily_Distance_Cost": [before_total, after_total]
})

plt.figure(figsize=(8, 6))

plt.bar(
    comparison["Scenario"],
    comparison["Total_Daily_Distance_Cost"],
    color=["#E74C3C", "#27AE60"]
)

plt.title("Before and After Total Daily Picking Distance Cost", fontsize=16)
plt.xlabel("Scenario", fontsize=12)
plt.ylabel("Total Daily Distance Cost", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for index, value in enumerate(comparison["Total_Daily_Distance_Cost"]):
    plt.text(
        index,
        value,
        f"{value:,.0f}",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()
plt.show()

# -----------------------------
# Chart 4: Distance cost saved by SKU
# -----------------------------
plt.figure(figsize=(14, 6))

plt.bar(
    df_sorted["SKU編號"],
    df_sorted["節省距離成本"],
    color=df_sorted["Color_Code"]
)

plt.title("Distance Cost Saved by SKU", fontsize=16)
plt.xlabel("SKU", fontsize=12)
plt.ylabel("Distance Cost Saved", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(handles=legend_patches, title="ABC Class")
plt.tight_layout()
plt.show()

# -----------------------------
# Chart 5: Number of SKUs by ABC Class
# -----------------------------
class_count = df_sorted["ABC類別"].value_counts().reset_index()
class_count.columns = ["ABC類別", "SKU數量"]
class_count["排序"] = class_count["ABC類別"].map(abc_order)
class_count = class_count.sort_values("排序")

plt.figure(figsize=(8, 6))

plt.bar(
    class_count["ABC類別"].replace({
        "A 類": "A Class",
        "B 類": "B Class",
        "C 類": "C Class"
    }),
    class_count["SKU數量"],
    color=class_count["ABC類別"].map(class_colors)
)

plt.title("Number of SKUs by ABC Class", fontsize=16)
plt.xlabel("ABC Class", fontsize=12)
plt.ylabel("Number of SKUs", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for index, value in enumerate(class_count["SKU數量"]):
    plt.text(
        index,
        value,
        str(value),
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()
plt.show()

# -----------------------------
# 13. 匯出分析結果 Excel
# -----------------------------
output_file = "Warehouse_ABC_分析結果_v3_fixed.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_sorted[display_columns].to_excel(
        writer,
        sheet_name="ABC分析結果",
        index=False
    )

    summary.to_excel(
        writer,
        sheet_name="ABC摘要",
        index=False
    )

files.download(output_file)
