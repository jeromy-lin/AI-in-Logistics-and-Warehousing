# ==========================================
# 主題： ABC 分類法：倉儲 SKU 儲位分析
# 作者：國立雲林科技大學電機系 林家仁
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# -----------------------------
# 1. 上傳 Excel 檔案
# -----------------------------
from google.colab import files

uploaded = files.upload()
file_name = list(uploaded.keys())[0]

print("已上傳檔案：", file_name)

# -----------------------------
# 2. 讀取 Excel 的「SKU資料輸入」工作表
# -----------------------------
# header=2 表示第 3 列是欄位名稱
df = pd.read_excel(
    file_name,
    sheet_name="SKU資料輸入",
    header=2
)

# 移除空白列
df = df.dropna(subset=["SKU編號"])

print("原始 SKU 資料")
display(df)

# -----------------------------
# 3. 檢查必要欄位
# -----------------------------
required_columns = [
    "SKU編號",
    "商品名稱",
    "每日揀貨次數",
    "商品體積",
    "重量kg",
    "目前距離出貨口m"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"缺少必要欄位：{col}")

# -----------------------------
# 4. 依每日揀貨次數排序
# -----------------------------
df_sorted = df.sort_values(
    by="每日揀貨次數",
    ascending=False
).reset_index(drop=True)

# -----------------------------
# 5. 計算出貨占比與累積出貨占比
# -----------------------------
total_picking_count = df_sorted["每日揀貨次數"].sum()

df_sorted["出貨占比"] = df_sorted["每日揀貨次數"] / total_picking_count
df_sorted["累積出貨占比"] = df_sorted["出貨占比"].cumsum()

# -----------------------------
# 6. ABC 分類
# -----------------------------
# A 類：累積出貨占比 <= 80%
# B 類：累積出貨占比 <= 95%
# C 類：累積出貨占比 > 95%

def abc_classification(cumulative_ratio):
    if cumulative_ratio <= 0.80:
        return "A 類"
    elif cumulative_ratio <= 0.95:
        return "B 類"
    else:
        return "C 類"

df_sorted["ABC類別"] = df_sorted["累積出貨占比"].apply(abc_classification)

# -----------------------------
# 7. 建議儲位區域
# -----------------------------
def recommend_storage_zone(abc_class):
    if abc_class == "A 類":
        return "近端儲位：靠近出貨口與包裝區"
    elif abc_class == "B 類":
        return "中段儲位：一般揀貨區"
    else:
        return "遠端或高層儲位：低頻商品區"

df_sorted["建議儲位區域"] = df_sorted["ABC類別"].apply(recommend_storage_zone)

# -----------------------------
# 8. 建議距離
# -----------------------------
# 這裡是假設值，可依實際倉庫修改
# A 類建議放 15 公尺內
# B 類建議放 35 公尺左右
# C 類可放 60 公尺左右

def recommended_distance(abc_class):
    if abc_class == "A 類":
        return 15
    elif abc_class == "B 類":
        return 35
    else:
        return 60

df_sorted["建議距離出貨口m"] = df_sorted["ABC類別"].apply(recommended_distance)

# -----------------------------
# 9. 計算改善前後距離成本
# -----------------------------
# 距離成本 = 每日揀貨次數 × 距離出貨口距離

df_sorted["改善前距離成本"] = (
    df_sorted["每日揀貨次數"] *
    df_sorted["目前距離出貨口m"]
)

df_sorted["改善後距離成本"] = (
    df_sorted["每日揀貨次數"] *
    df_sorted["建議距離出貨口m"]
)

df_sorted["節省距離成本"] = (
    df_sorted["改善前距離成本"] -
    df_sorted["改善後距離成本"]
)

before_total = df_sorted["改善前距離成本"].sum()
after_total = df_sorted["改善後距離成本"].sum()
saving_total = before_total - after_total
saving_rate = saving_total / before_total

# -----------------------------
# 10. 顏色設定
# -----------------------------
class_colors = {
    "A 類": "#2E86DE",   # 藍色
    "B 類": "#F39C12",   # 橘色
    "C 類": "#27AE60"    # 綠色
}

df_sorted["類別顏色"] = df_sorted["ABC類別"].map({
    "A 類": "藍色",
    "B 類": "橘色",
    "C 類": "綠色"
})

df_sorted["Color_Code"] = df_sorted["ABC類別"].map(class_colors)

# -----------------------------
# 11. 表格顯示顏色
# -----------------------------
def color_abc_class(value):
    if value == "A 類":
        return "background-color: #2E86DE; color: white; font-weight: bold;"
    elif value == "B 類":
        return "background-color: #F39C12; color: white; font-weight: bold;"
    elif value == "C 類":
        return "background-color: #27AE60; color: white; font-weight: bold;"
    else:
        return ""

display_columns = [
    "SKU編號",
    "商品名稱",
    "每日揀貨次數",
    "商品體積",
    "重量kg",
    "目前距離出貨口m",
    "出貨占比",
    "累積出貨占比",
    "ABC類別",
    "類別顏色",
    "建議儲位區域",
    "建議距離出貨口m",
    "改善前距離成本",
    "改善後距離成本",
    "節省距離成本"
]

print("ABC 分類結果與儲位建議")

display(
    df_sorted[display_columns]
    .style
    .map(color_abc_class, subset=["ABC類別"])
    .format({
        "出貨占比": "{:.2%}",
        "累積出貨占比": "{:.2%}",
        "商品體積": "{:.1f}",
        "重量kg": "{:.1f}",
        "改善前距離成本": "{:,.0f}",
        "改善後距離成本": "{:,.0f}",
        "節省距離成本": "{:,.0f}"
    })
)

# -----------------------------
# 12. ABC 分類摘要表
# -----------------------------
summary = df_sorted.groupby("ABC類別").agg(
    SKU數量=("SKU編號", "count"),
    每日揀貨次數總和=("每日揀貨次數", "sum"),
    出貨占比=("出貨占比", "sum"),
    平均目前距離m=("目前距離出貨口m", "mean"),
    平均建議距離m=("建議距離出貨口m", "mean"),
    改善前距離成本=("改善前距離成本", "sum"),
    改善後距離成本=("改善後距離成本", "sum"),
    節省距離成本=("節省距離成本", "sum")
).reset_index()

print("ABC 分類摘要表")

display(
    summary
    .style
    .map(color_abc_class, subset=["ABC類別"])
    .format({
        "出貨占比": "{:.2%}",
        "平均目前距離m": "{:.1f}",
        "平均建議距離m": "{:.1f}",
        "改善前距離成本": "{:,.0f}",
        "改善後距離成本": "{:,.0f}",
        "節省距離成本": "{:,.0f}"
    })
)

# -----------------------------
# 13. 顯示整體改善結果
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

for abc_class, color in class_colors.items():
    english_label = abc_class.replace("類", "Class")
    plt.bar([], [], color=color, label=english_label)

plt.legend(title="ABC Class")
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

for abc_class, color in class_colors.items():
    english_label = abc_class.replace("類", "Class")
    plt.bar([], [], color=color, label=english_label)

plt.legend(title="ABC Class")
plt.tight_layout()
plt.show()

# -----------------------------
# Chart 5: Total daily picking count by ABC class
# -----------------------------
summary_order = ["A 類", "B 類", "C 類"]
summary_plot = summary.set_index("ABC類別").loc[summary_order].reset_index()

plt.figure(figsize=(8, 6))

plt.bar(
    ["A Class", "B Class", "C Class"],
    summary_plot["每日揀貨次數總和"],
    color=[class_colors["A 類"], class_colors["B 類"], class_colors["C 類"]]
)

plt.title("Total Daily Picking Count by ABC Class", fontsize=16)
plt.xlabel("ABC Class", fontsize=12)
plt.ylabel("Total Daily Picking Count", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for index, value in enumerate(summary_plot["每日揀貨次數總和"]):
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
# 14. 匯出分析結果 Excel
# -----------------------------
output_file = "ABC_分析結果.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_sorted[display_columns].to_excel(writer, sheet_name="ABC分析結果", index=False)
    summary.to_excel(writer, sheet_name="ABC摘要", index=False)

files.download(output_file)
