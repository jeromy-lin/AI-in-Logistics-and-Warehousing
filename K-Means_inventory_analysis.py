# ==========================================
# 倉儲 SKU 商品 K-Means 分群分析
# 作者：國立雲林科技大學電機系 林家仁
# 倉儲 以50種 SKU 商品 進行分群設計
# 適用檔案：Warehouse_KMeans_SKU_template_50.xlsx
# 以8種元素進行 分群設計 說明PSU搭配K-Means 分群設計結果
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from google.colab import files

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
# 5. 文字特徵轉成數值特徵
# -----------------------------
volume_mapping = {
    "小": 1,
    "中": 2,
    "大": 3
}

weight_mapping = {
    "輕": 1,
    "中": 2,
    "重": 3
}

yes_no_mapping = {
    "否": 0,
    "是": 1
}

df["體積分數"] = df["體積等級"].map(volume_mapping).fillna(0)
df["重量分數"] = df["重量等級"].map(weight_mapping).fillna(0)
df["液體標記"] = df["是否液體"].map(yes_no_mapping).fillna(0)
df["易碎標記"] = df["是否易碎"].map(yes_no_mapping).fillna(0)
df["季節性標記"] = df["是否季節性"].map(yes_no_mapping).fillna(0)

print("K-Means 前處理後資料")
display(
    df[
        [
            "SKU編號",
            "商品名稱",
            "每日出貨箱數",
            "每日揀貨次數",
            "體積等級",
            "體積分數",
            "重量等級",
            "重量分數",
            "是否液體",
            "液體標記",
            "是否易碎",
            "易碎標記",
            "是否季節性",
            "季節性標記",
            "目前距離主要作業區m"
        ]
    ]
)

# -----------------------------
# 6. 選擇 K-Means 使用的特徵欄位
# -----------------------------
feature_columns = [
    "每日出貨箱數",
    "每日揀貨次數",
    "體積分數",
    "重量分數",
    "液體標記",
    "易碎標記",
    "季節性標記",
    "目前距離主要作業區m"
]

X = df[feature_columns]

# -----------------------------
# 7. 標準化
# -----------------------------
# K-Means 會受到數值大小影響，因此需要標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 8. Elbow Method 找適合的 K 值
# -----------------------------
inertia_values = []
k_range = range(2, 9)

for k_test in k_range:
    kmeans_test = KMeans(
        n_clusters=k_test,
        random_state=42,
        n_init=10
    )
    kmeans_test.fit(X_scaled)
    inertia_values.append(kmeans_test.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(
    list(k_range),
    inertia_values,
    marker="o",
    linewidth=2
)

plt.title("Elbow Method for Choosing K", fontsize=16)
plt.xlabel("Number of Clusters K", fontsize=12)
plt.ylabel("Inertia", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

print("Elbow Method 說明")
print("--------------------------------------------------")
print("K-Means 需要先指定 K 值，也就是要分成幾群。")
print("Elbow Method 會觀察 K 從 2 到 8 時，群內誤差 Inertia 的下降情形。")
print("若曲線在某個 K 值後開始趨於平緩，代表再增加群數的效益有限。")
print("本教學範例先以 K=5 進行，對應五種倉儲儲位策略。")

# -----------------------------
# 9. 執行 K-Means 分群
# -----------------------------
# 教學範例建議先使用 K = 5
# K=5 是根據倉儲儲位策略設計：
# 1. 高頻重物或液體
# 2. 高頻大體積
# 3. 中頻一般商品
# 4. 低頻季節性商品
# 5. 低頻備品或小品項

k = 5

kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)

df["KMeans群組"] = kmeans.fit_predict(X_scaled)

# 為了教學閱讀方便，群組編號改成 Cluster 1 ~ Cluster 5
df["KMeans群組名稱"] = df["KMeans群組"].apply(
    lambda x: f"Cluster {x + 1}"
)

# -----------------------------
# 9-1. 設定不同 Cluster 顏色
# -----------------------------
cluster_colors = {
    0: "#2E86DE",  # 藍色
    1: "#F39C12",  # 橘色
    2: "#27AE60",  # 綠色
    3: "#8E44AD",  # 紫色
    4: "#E74C3C"   # 紅色
}

df["Cluster顏色"] = df["KMeans群組"].map(cluster_colors)

# -----------------------------
# 9-2. 整理每一群的 SKU 品項
# -----------------------------
cluster_sku_list = df.groupby("KMeans群組名稱").agg(
    SKU數量=("SKU編號", "count"),
    SKU清單=("SKU編號", lambda x: "、".join(x.astype(str))),
    商品清單=("商品名稱", lambda x: "、".join(x.astype(str)))
).reset_index()

print("各 Cluster 對應 SKU 品項")
display(cluster_sku_list)

# -----------------------------
# 10. PCA 降維，用於 2D 視覺化
# -----------------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

df["PCA1"] = pca_result[:, 0]
df["PCA2"] = pca_result[:, 1]

# -----------------------------
# 11. 依分群特徵產生群組說明與儲位建議
# -----------------------------
cluster_summary = df.groupby("KMeans群組").agg(
    SKU數量=("SKU編號", "count"),
    平均每日出貨箱數=("每日出貨箱數", "mean"),
    平均每日揀貨次數=("每日揀貨次數", "mean"),
    平均體積分數=("體積分數", "mean"),
    平均重量分數=("重量分數", "mean"),
    液體比例=("液體標記", "mean"),
    易碎比例=("易碎標記", "mean"),
    季節性比例=("季節性標記", "mean"),
    平均目前距離m=("目前距離主要作業區m", "mean")
).reset_index()

def explain_cluster(row):
    picking = row["平均每日揀貨次數"]
    volume = row["平均體積分數"]
    weight = row["平均重量分數"]
    liquid = row["液體比例"]
    fragile = row["易碎比例"]
    seasonal = row["季節性比例"]

    if picking >= 60 and (weight >= 2.5 or liquid >= 0.4):
        return "高頻重物或液體商品"
    elif picking >= 50 and volume >= 2.4:
        return "高頻大體積商品"
    elif picking >= 25:
        return "中頻一般或補貨商品"
    elif seasonal >= 0.4:
        return "低頻季節性商品"
    elif fragile >= 0.4:
        return "低頻易碎或需保護商品"
    else:
        return "低頻一般備品或小品項"

def recommend_storage(row):
    cluster_type = row["群組特性說明"]

    if cluster_type == "高頻重物或液體商品":
        return "近端低層棧板區"
    elif cluster_type == "高頻大體積商品":
        return "近端大貨位"
    elif cluster_type == "中頻一般或補貨商品":
        return "中段一般儲位"
    elif cluster_type == "低頻季節性商品":
        return "遠端彈性儲位"
    elif cluster_type == "低頻易碎或需保護商品":
        return "遠端低層保護儲位"
    else:
        return "遠端或高層儲位"

cluster_summary["群組特性說明"] = cluster_summary.apply(
    explain_cluster,
    axis=1
)

cluster_summary["建議儲位策略"] = cluster_summary.apply(
    recommend_storage,
    axis=1
)

cluster_summary["KMeans群組名稱"] = cluster_summary["KMeans群組"].apply(
    lambda x: f"Cluster {x + 1}"
)

cluster_summary["Cluster顏色"] = cluster_summary["KMeans群組"].map(cluster_colors)

print("K-Means 群組摘要")
display(
    cluster_summary[
        [
            "KMeans群組名稱",
            "Cluster顏色",
            "SKU數量",
            "平均每日出貨箱數",
            "平均每日揀貨次數",
            "平均體積分數",
            "平均重量分數",
            "液體比例",
            "易碎比例",
            "季節性比例",
            "平均目前距離m",
            "群組特性說明",
            "建議儲位策略"
        ]
    ].style.format({
        "平均每日出貨箱數": "{:.1f}",
        "平均每日揀貨次數": "{:.1f}",
        "平均體積分數": "{:.2f}",
        "平均重量分數": "{:.2f}",
        "液體比例": "{:.2%}",
        "易碎比例": "{:.2%}",
        "季節性比例": "{:.2%}",
        "平均目前距離m": "{:.1f}"
    })
)

# 將群組說明合併回原始資料
df = df.merge(
    cluster_summary[
        [
            "KMeans群組",
            "群組特性說明",
            "建議儲位策略"
        ]
    ],
    on="KMeans群組",
    how="left"
)

# -----------------------------
# 12. 顯示 SKU 分群結果
# -----------------------------
result_columns = [
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
    "KMeans群組名稱",
    "Cluster顏色",
    "群組特性說明",
    "建議儲位策略",
    "備註"
]

print("SKU K-Means 分群結果")
display(df[result_columns])

# -----------------------------
# 12-1. 不同 Cluster 的 SKU 品項明細
# -----------------------------
cluster_detail_columns = [
    "KMeans群組名稱",
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
    "群組特性說明",
    "建議儲位策略"
]

df_cluster_detail = df[cluster_detail_columns].sort_values(
    by=["KMeans群組名稱", "每日揀貨次數"],
    ascending=[True, False]
)

print("不同 Cluster 的 SKU 品項明細")
display(df_cluster_detail)

# ==========================================================
# 以下圖表維持英文，方便放入簡報
# ==========================================================

# -----------------------------
# Chart 1: K-Means PCA Scatter Plot
# 每一個 Cluster 用不同顏色顯示
# -----------------------------
plt.figure(figsize=(11, 8))

for cluster_id in sorted(df["KMeans群組"].unique()):
    cluster_data = df[df["KMeans群組"] == cluster_id]

    plt.scatter(
        cluster_data["PCA1"],
        cluster_data["PCA2"],
        s=100,
        alpha=0.85,
        color=cluster_colors[cluster_id],
        label=f"Cluster {cluster_id + 1}"
    )

    # 標註 SKU 編號
    for i in cluster_data.index:
        plt.text(
            df.loc[i, "PCA1"],
            df.loc[i, "PCA2"],
            df.loc[i, "SKU編號"],
            fontsize=8,
            ha="center",
            va="bottom"
        )

plt.title("K-Means Clustering Result by PCA", fontsize=16)
plt.xlabel("PCA Component 1", fontsize=12)
plt.ylabel("PCA Component 2", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# -----------------------------
# Chart 2: SKU Count by Cluster
# -----------------------------
cluster_count = df["KMeans群組名稱"].value_counts().sort_index()

cluster_count_colors = []
for cluster_name in cluster_count.index:
    cluster_number = int(cluster_name.replace("Cluster ", "")) - 1
    cluster_count_colors.append(cluster_colors[cluster_number])

plt.figure(figsize=(8, 6))

plt.bar(
    cluster_count.index,
    cluster_count.values,
    color=cluster_count_colors
)

plt.title("Number of SKUs by Cluster", fontsize=16)
plt.xlabel("Cluster", fontsize=12)
plt.ylabel("Number of SKUs", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for index, value in enumerate(cluster_count.values):
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
# Chart 3: Average Picking Count by Cluster
# -----------------------------
avg_picking = df.groupby("KMeans群組名稱")["每日揀貨次數"].mean().sort_index()

avg_picking_colors = []
for cluster_name in avg_picking.index:
    cluster_number = int(cluster_name.replace("Cluster ", "")) - 1
    avg_picking_colors.append(cluster_colors[cluster_number])

plt.figure(figsize=(8, 6))

plt.bar(
    avg_picking.index,
    avg_picking.values,
    color=avg_picking_colors
)

plt.title("Average Daily Picking Count by Cluster", fontsize=16)
plt.xlabel("Cluster", fontsize=12)
plt.ylabel("Average Daily Picking Count", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for index, value in enumerate(avg_picking.values):
    plt.text(
        index,
        value,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()
plt.show()

# -----------------------------
# Chart 4: Average Weight Score by Cluster
# -----------------------------
avg_weight = df.groupby("KMeans群組名稱")["重量分數"].mean().sort_index()

avg_weight_colors = []
for cluster_name in avg_weight.index:
    cluster_number = int(cluster_name.replace("Cluster ", "")) - 1
    avg_weight_colors.append(cluster_colors[cluster_number])

plt.figure(figsize=(8, 6))

plt.bar(
    avg_weight.index,
    avg_weight.values,
    color=avg_weight_colors
)

plt.title("Average Weight Score by Cluster", fontsize=16)
plt.xlabel("Cluster", fontsize=12)
plt.ylabel("Average Weight Score", fontsize=12)
plt.ylim(0, 3.2)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for index, value in enumerate(avg_weight.values):
    plt.text(
        index,
        value,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()
plt.show()

# -----------------------------
# Chart 5: Average Volume Score by Cluster
# -----------------------------
avg_volume = df.groupby("KMeans群組名稱")["體積分數"].mean().sort_index()

avg_volume_colors = []
for cluster_name in avg_volume.index:
    cluster_number = int(cluster_name.replace("Cluster ", "")) - 1
    avg_volume_colors.append(cluster_colors[cluster_number])

plt.figure(figsize=(8, 6))

plt.bar(
    avg_volume.index,
    avg_volume.values,
    color=avg_volume_colors
)

plt.title("Average Volume Score by Cluster", fontsize=16)
plt.xlabel("Cluster", fontsize=12)
plt.ylabel("Average Volume Score", fontsize=12)
plt.ylim(0, 3.2)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for index, value in enumerate(avg_volume.values):
    plt.text(
        index,
        value,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()
plt.show()

# -----------------------------
# Chart 6: Average Distance to Main Operation Area by Cluster
# -----------------------------
avg_distance = df.groupby("KMeans群組名稱")["目前距離主要作業區m"].mean().sort_index()

avg_distance_colors = []
for cluster_name in avg_distance.index:
    cluster_number = int(cluster_name.replace("Cluster ", "")) - 1
    avg_distance_colors.append(cluster_colors[cluster_number])

plt.figure(figsize=(8, 6))

plt.bar(
    avg_distance.index,
    avg_distance.values,
    color=avg_distance_colors
)

plt.title("Average Distance to Main Operation Area by Cluster", fontsize=16)
plt.xlabel("Cluster", fontsize=12)
plt.ylabel("Average Distance (m)", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for index, value in enumerate(avg_distance.values):
    plt.text(
        index,
        value,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=11
    )

plt.tight_layout()
plt.show()

# -----------------------------
# 13. 匯出 K-Means 分群結果 Excel
# -----------------------------
output_file = "Warehouse_KMeans_分群結果_50.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df[result_columns + [
        "體積分數",
        "重量分數",
        "液體標記",
        "易碎標記",
        "季節性標記",
        "PCA1",
        "PCA2"
    ]].to_excel(
        writer,
        sheet_name="KMeans分群結果",
        index=False
    )

    cluster_summary[
        [
            "KMeans群組名稱",
            "Cluster顏色",
            "SKU數量",
            "平均每日出貨箱數",
            "平均每日揀貨次數",
            "平均體積分數",
            "平均重量分數",
            "液體比例",
            "易碎比例",
            "季節性比例",
            "平均目前距離m",
            "群組特性說明",
            "建議儲位策略"
        ]
    ].to_excel(
        writer,
        sheet_name="群組摘要",
        index=False
    )

    cluster_sku_list.to_excel(
        writer,
        sheet_name="各群SKU清單",
        index=False
    )

    df_cluster_detail.to_excel(
        writer,
        sheet_name="各群SKU明細",
        index=False
    )

files.download(output_file)
