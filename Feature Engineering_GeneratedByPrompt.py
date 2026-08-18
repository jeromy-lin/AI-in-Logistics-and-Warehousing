# ==============================================================================
# 宅配物流 WMS 企業客戶特徵工程 (Customer-Month Feature Engineering)
# 目標：將「一筆包裹一筆紀錄」轉換為「Customer-Month 特徵」並產生下一階段 K-Means Excel
# 作者：國立雲林科技大學電機系 林家仁
# ==============================================================================

import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# ------------------------------------------------------------------------------
# 第一區：Excel 檔案上傳與自動辨識
# ------------------------------------------------------------------------------
print("=== STEP 1: 請選擇上傳 WMS 原始資料 Excel 檔案 ===")
uploaded = files.upload()

if not uploaded:
    raise ValueError("未上傳任何檔案，請重新執行並選擇檔案！")

# 自動取得上傳的檔案名稱
file_name = list(uploaded.keys())[0]
print(f"\n成功讀取上傳檔案：{file_name}")

# 自動讀取 Excel Sheet Names
excel_file = pd.ExcelFile(io.BytesIO(uploaded[file_name]))
available_sheets = excel_file.sheet_names
print(f"可用的工作表 (Sheet Names)：{available_sheets}")

# 自動尋找包含 WMS 原始配送資料的工作表
target_sheet = None
for sheet in available_sheets:
    if 'WMS' in sheet or '原始' in sheet or 'Data' in sheet.upper():
        target_sheet = sheet
        break

if target_sheet is None:
    print("\n[錯誤] 找不到名稱包含 WMS 或原始資料的工作表！")
    print(f"目前檔案內所有可用的 Sheet Names 為：{available_sheets}")
    raise KeyError("請確認工作表名稱後重新執行。")

print(f"已鎖定目標工作表：[{target_sheet}]")
df_raw = pd.read_excel(io.BytesIO(uploaded[file_name]), sheet_name=target_sheet)

# ------------------------------------------------------------------------------
# 第二區：原始 WMS 資料檢查
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("=== STEP 2: 原始 WMS 資料檢查結果 ===")
print("="*50)
print(f"1. Excel 檔案名稱：{file_name}")
print(f"2. 使用的工作表名稱：{target_sheet}")
print(f"3. 原始資料筆數：{len(df_raw):,} 筆")
print(f"4. 欄位數量：{df_raw.shape[1]} 個")
print(f"5. 所有欄位名稱：\n   {df_raw.columns.tolist()}")
print("\n6. 前 5 筆資料預覽：")
display(df_raw.head())

print("\n7. 各欄位缺失值數量：")
print(df_raw.isnull().sum())

# ------------------------------------------------------------------------------
# 第三區：資料前處理與 Customer-Month 特徵建立
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("=== STEP 3: 資料轉換與 Customer-Month 特徵建立 ===")
print("="*50)

# 日期格式轉換與建立 YYYY-MM 月份
df_raw['出貨日期'] = pd.to_datetime(df_raw['出貨日期'])
df_raw['Month'] = df_raw['出貨日期'].dt.strftime('%Y-%m')

# 重複資料檢查
duplicate_count = df_raw.duplicated().sum()
print(f"重複資料筆數：{duplicate_count} 筆")
if duplicate_count > 0:
    df_raw = df_raw.drop_duplicates()

# 依 [客戶代碼, 客戶名稱, Month] 進行 Customer-Month 群組彙整
grouped = df_raw.groupby(['客戶代碼', '客戶名稱', 'Month'])

df_cm = grouped.agg(
    Monthly_Volume=('包裹編號', 'count'),
    Avg_Weight=('重量kg', 'mean'),
    Avg_Size=('尺寸總和cm', 'mean'),
    S60_Count=('尺寸級距', lambda x: (x == 's60').sum()),
    S90_Count=('尺寸級距', lambda x: (x == 's90').sum()),
    S120_Count=('尺寸級距', lambda x: (x == 's120').sum()),
    S150_Count=('尺寸級距', lambda x: (x == 's150').sum()),
    Delivery_City_Count=('收件縣市', 'nunique'),
    Remote_Count=('配送區域類型', lambda x: (x == '偏遠').sum()),
    Cold_Count=('溫層', lambda x: (x == '低溫').sum()),
    COD_Count=('COD', lambda x: (x == 'Y').sum()),
    TimeSlot_Count=('指定時段', lambda x: (x != '無').sum()),
    Redelivery_Count=('配送次數', lambda x: (x >= 2).sum()),
    Failure_Count=('配送結果', lambda x: (x == '配送失敗').sum())
).reset_index()

# 計算特徵比例
df_cm['S60_Ratio'] = df_cm['S60_Count'] / df_cm['Monthly_Volume']
df_cm['S90_Ratio'] = df_cm['S90_Count'] / df_cm['Monthly_Volume']
df_cm['S120_Ratio'] = df_cm['S120_Count'] / df_cm['Monthly_Volume']
df_cm['S150_Ratio'] = df_cm['S150_Count'] / df_cm['Monthly_Volume']
df_cm['Remote_Ratio'] = df_cm['Remote_Count'] / df_cm['Monthly_Volume']
df_cm['Cold_Ratio'] = df_cm['Cold_Count'] / df_cm['Monthly_Volume']
df_cm['COD_Ratio'] = df_cm['COD_Count'] / df_cm['Monthly_Volume']
df_cm['TimeSlot_Ratio'] = df_cm['TimeSlot_Count'] / df_cm['Monthly_Volume']
df_cm['Redelivery_Ratio'] = df_cm['Redelivery_Count'] / df_cm['Monthly_Volume']
df_cm['Failure_Ratio'] = df_cm['Failure_Count'] / df_cm['Monthly_Volume']

# 數值適度四捨五入 (平均值與比例)
round_cols = ['Avg_Weight', 'Avg_Size', 'S60_Ratio', 'S90_Ratio', 'S120_Ratio', 'S150_Ratio', 
              'Remote_Ratio', 'Cold_Ratio', 'COD_Ratio', 'TimeSlot_Ratio', 'Redelivery_Ratio', 'Failure_Ratio']
df_cm[round_cols] = df_cm[round_cols].round(4)

print(f"\n轉換完成：Original WMS Records ({len(df_raw):,} 筆) → Customer-Month Records ({len(df_cm)} 筆)")
print("\nCustomer-Month 資料預覽（前 5 筆）：")
display(df_cm.head())

# ------------------------------------------------------------------------------
# 第四區：建立 KMeans_Features 與英文客戶對照
# ------------------------------------------------------------------------------

# 中英文客戶名稱映射表 (供繪圖與英文工作表使用，避免中文字型顯示問題)
customer_en_map = {
    '全聯福利中心': 'PX Mart',
    'momo購物網': 'momo',
    'PChome': 'PChome',
    '食品企業A': 'Food Company A',
    '製造企業B': 'Manufacturing Company B'
}

df_kmeans = pd.DataFrame({
    'Customer_ID': df_cm['客戶代碼'],
    'Customer': df_cm['客戶名稱'].map(customer_en_map).fillna(df_cm['客戶名稱']),
    'Month': df_cm['Month'],
    'Monthly_Volume': df_cm['Monthly_Volume'],
    'Avg_Weight': df_cm['Avg_Weight'],
    'Avg_Size': df_cm['Avg_Size'],
    'S60_Ratio': df_cm['S60_Ratio'],
    'S90_Ratio': df_cm['S90_Ratio'],
    'S120_Ratio': df_cm['S120_Ratio'],
    'S150_Ratio': df_cm['S150_Ratio'],
    'Delivery_City_Count': df_cm['Delivery_City_Count'],
    'Remote_Ratio': df_cm['Remote_Ratio'],
    'Cold_Ratio': df_cm['Cold_Ratio'],
    'COD_Ratio': df_cm['COD_Ratio'],
    'TimeSlot_Ratio': df_cm['TimeSlot_Ratio'],
    'Redelivery_Ratio': df_cm['Redelivery_Ratio'],
    'Failure_Ratio': df_cm['Failure_Ratio']
})

# ------------------------------------------------------------------------------
# 第五區：Matplotlib 全英文圖表繪製 (無中文字型，適合展示)
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("=== STEP 4: 產生全英文分析圖表 ===")
print("="*50)

# 準備繪圖使用的 DataFrame (包含英文名稱)
df_plot = df_cm.copy()
df_plot['Customer_EN'] = df_plot['客戶名稱'].map(customer_en_map).fillna(df_plot['客戶名稱'])
customers = df_plot['Customer_EN'].unique()
months = sorted(df_plot['Month'].unique())

# 色彩設定 (簡潔專業)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Chart 1: Monthly Shipment Volume by Customer (Grouped Bar Chart)
plt.figure(figsize=(10, 5))
x = np.arange(len(months))
width = 0.15

for i, cust in enumerate(customers):
    cust_data = df_plot[df_plot['Customer_EN'] == cust]
    volumes = [cust_data[cust_data['Month'] == m]['Monthly_Volume'].values[0] if len(cust_data[cust_data['Month'] == m]) > 0 else 0 for m in months]
    plt.bar(x + i * width, volumes, width, label=cust, color=colors[i % len(colors)])

plt.title('Monthly Shipment Volume by Customer', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Month', fontsize=11)
plt.ylabel('Shipment Volume (Packages)', fontsize=11)
plt.xticks(x + width * (len(customers) - 1) / 2, months)
plt.legend(title='Customer', loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Chart 2: Customer Logistics Characteristics (Averaged Ratio Profiles)
plt.figure(figsize=(11, 5.5))
ratio_cols = ['Remote_Ratio', 'Cold_Ratio', 'COD_Ratio', 'TimeSlot_Ratio', 'Redelivery_Ratio']
ratio_labels = ['Remote Ratio', 'Cold Ratio', 'COD Ratio', 'Time Slot Ratio', 'Redelivery Ratio']

cust_ratio_avg = df_plot.groupby('Customer_EN')[ratio_cols].mean()

x_ratio = np.arange(len(ratio_labels))
width_ratio = 0.15

for i, cust in enumerate(customers):
    vals = cust_ratio_avg.loc[cust, ratio_cols].values
    plt.bar(x_ratio + i * width_ratio, vals, width_ratio, label=cust, color=colors[i % len(colors)])

plt.title('Customer Logistics Characteristics (Average Ratio Profiles)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Logistics Feature', fontsize=11)
plt.ylabel('Ratio', fontsize=11)
plt.xticks(x_ratio + width_ratio * (len(customers) - 1) / 2, ratio_labels)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.legend(title='Customer', loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Chart 3: Average Package Weight by Customer (Bar Chart)
plt.figure(figsize=(9, 5))
avg_weights = df_plot.groupby('Customer_EN')['Avg_Weight'].mean().reindex(customers)

bars3 = plt.bar(customers, avg_weights, color=colors, width=0.5)
plt.title('Average Package Weight by Customer', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Customer', fontsize=11)
plt.ylabel('Average Weight (kg)', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars3:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f} kg', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Chart 4: Average Package Size by Customer (Bar Chart)
plt.figure(figsize=(9, 5))
avg_sizes = df_plot.groupby('Customer_EN')['Avg_Size'].mean().reindex(customers)

bars4 = plt.bar(customers, avg_sizes, color=colors, width=0.5)
plt.title('Average Package Size by Customer (L + W + H)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Customer', fontsize=11)
plt.ylabel('Average Size Sum (cm)', fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars4:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f} cm', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
# 第六區：匯出 Excel 與自動下載 (customer_monthly_features.xlsx)
# ------------------------------------------------------------------------------
print("\n" + "="*50)
print("=== STEP 5: 匯出多 Sheet Excel 檔案 ===")
print("="*50)

output_filename = 'customer_monthly_features.xlsx'

# 1. Customer_Profile 工作表
df_profile = df_plot.groupby(['客戶代碼', '客戶名稱', 'Customer_EN']).agg(
    Total_Shipment_6M=('Monthly_Volume', 'sum'),
    Avg_Monthly_Volume=('Monthly_Volume', 'mean'),
    Avg_Weight=('Avg_Weight', 'mean'),
    Avg_Size=('Avg_Size', 'mean'),
    Avg_Remote_Ratio=('Remote_Ratio', 'mean'),
    Avg_Cold_Ratio=('Cold_Ratio', 'mean'),
    Avg_COD_Ratio=('COD_Ratio', 'mean'),
    Avg_TimeSlot_Ratio=('TimeSlot_Ratio', 'mean'),
    Avg_Redelivery_Ratio=('Redelivery_Ratio', 'mean'),
    Avg_Failure_Ratio=('Failure_Ratio', 'mean')
).reset_index().round(4)

# 2. Monthly_Volume 工作表 (Pivot Table)
df_monthly_pivot = df_plot.pivot_table(
    index=['客戶代碼', '客戶名稱'], 
    columns='Month', 
    values='Monthly_Volume', 
    aggfunc='sum'
).reset_index()

# 寫入 Excel
with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    df_cm.to_excel(writer, sheet_name='Customer-Month', index=False)
    df_kmeans.to_excel(writer, sheet_name='KMeans_Features', index=False)
    df_profile.to_excel(writer, sheet_name='Customer_Profile', index=False)
    df_monthly_pivot.to_excel(writer, sheet_name='Monthly_Volume', index=False)

print(f"Excel 檔案 [{output_filename}] 已成功建立！")
print("包含工作表：")
print("  1. Customer-Month")
print("  2. KMeans_Features (準備供下一步驟 K-Means 使用)")
print("  3. Customer_Profile")
print("  4. Monthly_Volume")

# 自動啟動瀏覽器下載
print("\n正在下載產生的 Excel 檔案至您的電腦...")
files.download(output_filename)
print("=== 特徵工程流程順利完成！ ===")
