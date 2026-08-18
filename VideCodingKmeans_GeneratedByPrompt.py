# ==============================================================================
# 作者：國立雲林科技大學電機系 林家仁
# 企業物流客戶 K-Means 分群分析與視覺化 (Google Colab 完整執行檔)
# ==============================================================================

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ------------------------------------------------------------------------------
# 1. 檔案上傳與自動讀取
# ------------------------------------------------------------------------------
print("=== 步驟 1: 請上傳 Excel 檔案 ===")
uploaded = files.upload()

# 自動取得第一個上傳檔案的檔名與內容
uploaded_filename = list(uploaded.keys())[0]
print(f"\n成功讀取檔案: {uploaded_filename}")

target_sheet = 'KMeans_Features'
df = pd.read_excel(io.BytesIO(uploaded[uploaded_filename]), sheet_name=target_sheet)

# ------------------------------------------------------------------------------
# 2. 基本資料檢視
# ------------------------------------------------------------------------------
print("\n=== 步驟 2: 資料基本檢視 ===")
print(f"資料筆數 (Total Rows): {len(df)}")
print(f"欄位名稱 (Columns): {df.columns.tolist()}")
print("\n前 5 筆資料 (First 5 Rows):")
print(df.head())

# ------------------------------------------------------------------------------
# 3. 定義識別欄位與物流特徵
# ------------------------------------------------------------------------------
id_cols = ['Customer_ID', 'Customer', 'Month']
feature_cols = [
    'Monthly_Volume', 'Avg_Weight', 'Avg_Size', 'S60_Ratio', 'S90_Ratio',
    'S120_Ratio', 'S150_Ratio', 'Delivery_City_Count', 'Remote_Ratio',
    'Cold_Ratio', 'COD_Ratio', 'TimeSlot_Ratio', 'Redelivery_Ratio', 'Failure_Ratio'
]

# 確保所有特徵皆存在於資料中
missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Excel 中缺少以下欄位: {missing_cols}")

# ------------------------------------------------------------------------------
# 4. 特徵標準化 (StandardScaler)
# ------------------------------------------------------------------------------
X = df[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------------------
# 5. 測試不同 K 值 (K=2, 3, 4, 5) 並計算指標
# ------------------------------------------------------------------------------
k_range = [2, 3, 4, 5]
inertias = []
silhouette_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil_score)
    print(f"K = {k} | Inertia = {km.inertia_:.4f} | Silhouette Score = {sil_score:.4f}")

# ------------------------------------------------------------------------------
# 6. 圖表繪製 1: Elbow Method
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertias, marker='o', color='#1f77b4', linewidth=2, markersize=8)
plt.title('Elbow Method for K-Means', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Inertia (WCSS)', fontsize=12)
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 7. 圖表繪製 2: Silhouette Score
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouette_scores, marker='s', color='#2ca02c', linewidth=2, markersize=8)
plt.title('Silhouette Score by Number of Clusters', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Silhouette Score', fontsize=12)
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 8. 自動選擇最佳 K 並重新執行 K-Means
# ------------------------------------------------------------------------------
best_k_idx = int(np.argmax(silhouette_scores))
best_k = k_range[best_k_idx]
print(f"\n根據 Silhouette Score 最高值，自動選擇最佳群數 Best K = {best_k}")

kmeans_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans_best.fit_predict(X_scaled)

# 加入 Cluster 編號至原始資料 (標示為 Cluster 1 ~ Best K)
df['Cluster'] = [f"Cluster {l+1}" for l in cluster_labels]

# ------------------------------------------------------------------------------
# 9. PCA 降維 (2 維) 與 PCA 分群圖
# ------------------------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled)
df['PC1'] = pca_coords[:, 0]
df['PC2'] = pca_coords[:, 1]

plt.figure(figsize=(10, 7))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for i in range(best_k):
    cluster_name = f"Cluster {i+1}"
    sub_df = df[df['Cluster'] == cluster_name]
    plt.scatter(
        sub_df['PC1'], sub_df['PC2'],
        c=colors[i % len(colors)],
        label=cluster_name,
        s=90, alpha=0.85, edgecolors='k', linewidth=0.8
    )

# 加上 Customer 名稱與適當偏移，避免嚴重重疊
for idx, row in df.iterrows():
    plt.annotate(
        row['Customer'],
        (row['PC1'], row['PC2']),
        xytext=(5, 4), textcoords='offset points',
        fontsize=8, alpha=0.85
    )

plt.title('K-Means Customer Segmentation with PCA', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Clusters', loc='best', frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 10. Cluster Profile 與 Z-score 標準化比較圖
# ------------------------------------------------------------------------------
# 原始特徵平均值 Profile
cluster_profile_raw = df.groupby('Cluster')[feature_cols].mean()

# 將 Cluster Profile 做 Z-score 標準化，避免因量綱不同導致特徵被壓縮
profile_scaler = StandardScaler()
profile_std_values = profile_scaler.fit_transform(cluster_profile_raw)
cluster_profile_std = pd.DataFrame(
    profile_std_values,
    index=cluster_profile_raw.index,
    columns=feature_cols
)

# 指定比較的重點特徵 (按要求包含指定的 8 項特徵)
compare_features = [
    'Monthly_Volume', 'Avg_Weight', 'Avg_Size', 'Remote_Ratio',
    'Cold_Ratio', 'COD_Ratio', 'TimeSlot_Ratio', 'Redelivery_Ratio'
]

# 繪製 Standardized Cluster Characteristics Comparison
plt.figure(figsize=(12, 6))
x_axis = np.arange(len(compare_features))
bar_width = 0.8 / best_k

for i in range(best_k):
    cluster_name = f"Cluster {i+1}"
    y_vals = cluster_profile_std.loc[cluster_name, compare_features].values
    plt.bar(
        x_axis + i * bar_width,
        y_vals,
        width=bar_width,
        label=cluster_name,
        color=colors[i % len(colors)],
        edgecolor='black',
        linewidth=0.6,
        alpha=0.85
    )

plt.axhline(0, color='red', linestyle='--', linewidth=1.2, label='Overall Average (Z=0)')
plt.title('Standardized Cluster Characteristics Comparison', fontsize=14, fontweight='bold', pad=12)
plt.xlabel('Logistics Features', fontsize=12)
plt.ylabel('Standardized Value (Z-score)', fontsize=12)
plt.xticks(x_axis + bar_width * (best_k - 1) / 2, compare_features, rotation=25, ha='right', fontsize=10)
plt.legend(title='Clusters', loc='upper right', bbox_to_anchor=(1.15, 1.0))
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 11. 計算 Customer Main Cluster
# ------------------------------------------------------------------------------
# 每一家企業在所有月份中最常出現的 Cluster (眾數 Mode)
main_cluster_series = df.groupby(['Customer_ID', 'Customer'])['Cluster'].agg(
    lambda x: x.mode()[0]
).reset_index()

customer_main_cluster = main_cluster_series.rename(columns={'Cluster': 'Main_Cluster'})

# 建立 Model Selection 統計結果表
model_selection_df = pd.DataFrame({
    'K': k_range,
    'Inertia': inertias,
    'Silhouette_Score': silhouette_scores,
    'Is_Best_K': [k == best_k for k in k_range]
})

# ------------------------------------------------------------------------------
# 12. 匯出 Excel (kmeans_customer_segmentation.xlsx) 並下載
# ------------------------------------------------------------------------------
output_filename = 'kmeans_customer_segmentation.xlsx'

with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Segmentation_Result', index=False)
    customer_main_cluster.to_excel(writer, sheet_name='Customer_Main_Cluster', index=False)
    cluster_profile_raw.reset_index().to_excel(writer, sheet_name='Cluster_Profile', index=False)
    cluster_profile_std.reset_index().to_excel(writer, sheet_name='Standardized_Cluster_Profile', index=False)
    model_selection_df.to_excel(writer, sheet_name='Model_Selection', index=False)

print(f"\n成功匯出結果至 {output_filename}")
print("包含工作表: Segmentation_Result, Customer_Main_Cluster, Cluster_Profile, Standardized_Cluster_Profile, Model_Selection")

# 下載檔案至本地端
files.download(output_filename)
