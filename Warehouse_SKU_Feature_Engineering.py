# ============================================================
# 主題：# WMS 原始配送資料 → Customer-Month 客戶月特徵表
# 功能：
# 1. 使用 files.upload() 上傳 Excel
# 2. 自動讀取上傳檔案
# 3. 讀取工作表：WMS原始資料
# 4. 將大量包裹資料依「客戶 + 月份」進行彙整
# 5. 建立後續 K-Means 所需的 Customer-Month Features
# 6. 顯示分析結果與英文圖表
# 7. 輸出新的 Excel
#
# 執行環境：Google Colab 免費版 CPU
#
# ============================================================


# ============================================================
# STEP 1：載入套件
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from google.colab import files


# ============================================================
# STEP 2：上傳 Excel
# ============================================================

print("==============================================")
print("請上傳 WMS Excel 檔案")
print("==============================================")

uploaded = files.upload()

# 自動取得第一個上傳的 Excel 檔案
file_name = list(uploaded.keys())[0]

print("\n已成功上傳：")
print(file_name)


# ============================================================
# STEP 3：讀取 WMS 原始資料
# ============================================================

sheet_name = "WMS原始資料"

try:

    df = pd.read_excel(
        file_name,
        sheet_name=sheet_name
    )

except Exception as e:

    print("\n讀取 Excel 發生錯誤：")
    print(e)

    raise


# ============================================================
# STEP 4：顯示原始資料基本資訊
# ============================================================

print("\n")
print("==============================================")
print("WMS 原始資料基本資訊")
print("==============================================")

print("資料筆數：", len(df))

print("欄位數量：", len(df.columns))

print("\n資料欄位：")

for col in df.columns:

    print("-", col)


print("\n前 5 筆 WMS 資料：")

display(
    df.head()
)


# ============================================================
# STEP 5：必要欄位檢查
# ============================================================

required_columns = [

    "出貨日期",

    "客戶代碼",

    "客戶名稱",

    "包裹編號",

    "尺寸級距",

    "尺寸總和cm",

    "重量kg",

    "收件縣市",

    "配送區域類型",

    "溫層",

    "COD",

    "指定時段",

    "配送結果",

    "配送次數"

]


missing_columns = [

    col

    for col in required_columns

    if col not in df.columns

]


if len(missing_columns) > 0:

    print("\n")

    print("⚠ Excel 缺少以下欄位：")

    for col in missing_columns:

        print("-", col)

    raise ValueError(
        "Excel 欄位不完整，請確認 WMS 原始資料格式。"
    )

else:

    print("\n✓ 欄位檢查完成")

    print("所有必要欄位皆存在。")


# ============================================================
# STEP 6：資料清理
# ============================================================

# ------------------------------------------------------------
# 日期轉換
# ------------------------------------------------------------

df["出貨日期"] = pd.to_datetime(

    df["出貨日期"],

    errors="coerce"

)


# 移除日期錯誤資料
df = df.dropna(

    subset=["出貨日期"]

)


# ------------------------------------------------------------
# 建立月份欄位
# ------------------------------------------------------------

df["月份"] = (

    df["出貨日期"]

    .dt.to_period("M")

    .astype(str)

)


# ------------------------------------------------------------
# 數值欄位轉換
# ------------------------------------------------------------

df["重量kg"] = pd.to_numeric(

    df["重量kg"],

    errors="coerce"

)


df["尺寸總和cm"] = pd.to_numeric(

    df["尺寸總和cm"],

    errors="coerce"

)


df["配送次數"] = pd.to_numeric(

    df["配送次數"],

    errors="coerce"

).fillna(1)


# ============================================================
# STEP 7：建立分析 Feature Flags
# ============================================================

# ------------------------------------------------------------
# 尺寸級距
# ------------------------------------------------------------

df["s60_flag"] = (

    df["尺寸級距"] == "s60"

).astype(int)


df["s90_flag"] = (

    df["尺寸級距"] == "s90"

).astype(int)


df["s120_flag"] = (

    df["尺寸級距"] == "s120"

).astype(int)


df["s150_flag"] = (

    df["尺寸級距"] == "s150"

).astype(int)


# ------------------------------------------------------------
# 偏遠配送
# ------------------------------------------------------------

df["remote_flag"] = (

    df["配送區域類型"] == "偏遠"

).astype(int)


# ------------------------------------------------------------
# 低溫配送
# ------------------------------------------------------------

df["cold_flag"] = (

    df["溫層"] == "低溫"

).astype(int)


# ------------------------------------------------------------
# COD
# ------------------------------------------------------------

df["cod_flag"] = (

    df["COD"]

    .astype(str)

    .str.upper()

    == "Y"

).astype(int)


# ------------------------------------------------------------
# 指定時段
# ------------------------------------------------------------

df["timeslot_flag"] = (

    df["指定時段"]

    .astype(str)

    != "無"

).astype(int)


# ------------------------------------------------------------
# 二次配送
# ------------------------------------------------------------

df["redelivery_flag"] = (

    df["配送次數"] >= 2

).astype(int)


# ------------------------------------------------------------
# 配送失敗
# ------------------------------------------------------------

df["failure_flag"] = (

    df["配送結果"] == "配送失敗"

).astype(int)


# ============================================================
# STEP 8：依「客戶 + 月份」進行彙整
# ============================================================

customer_monthly = (

    df

    .groupby(

        [

            "客戶代碼",

            "客戶名稱",

            "月份"

        ],

        as_index=False

    )

    .agg(

        # ----------------------------------------------------
        # 出貨量
        # ----------------------------------------------------

        月出貨件數=(

            "包裹編號",

            "count"

        ),


        # ----------------------------------------------------
        # 平均重量
        # ----------------------------------------------------

        平均重量kg=(

            "重量kg",

            "mean"

        ),


        # ----------------------------------------------------
        # 平均尺寸
        # ----------------------------------------------------

        平均尺寸總和cm=(

            "尺寸總和cm",

            "mean"

        ),


        # ----------------------------------------------------
        # 尺寸結構
        # ----------------------------------------------------

        s60比例=(

            "s60_flag",

            "mean"

        ),


        s90比例=(

            "s90_flag",

            "mean"

        ),


        s120比例=(

            "s120_flag",

            "mean"

        ),


        s150比例=(

            "s150_flag",

            "mean"

        ),


        # ----------------------------------------------------
        # 配送區域
        # ----------------------------------------------------

        配送縣市數=(

            "收件縣市",

            "nunique"

        ),


        偏遠配送比例=(

            "remote_flag",

            "mean"

        ),


        # ----------------------------------------------------
        # 溫層
        # ----------------------------------------------------

        低溫配送比例=(

            "cold_flag",

            "mean"

        ),


        # ----------------------------------------------------
        # COD
        # ----------------------------------------------------

        COD比例=(

            "cod_flag",

            "mean"

        ),


        # ----------------------------------------------------
        # 指定配送時段
        # ----------------------------------------------------

        指定時段比例=(

            "timeslot_flag",

            "mean"

        ),


        # ----------------------------------------------------
        # 二次配送
        # ----------------------------------------------------

        二次配送比例=(

            "redelivery_flag",

            "mean"

        ),


        # ----------------------------------------------------
        # 配送失敗
        # ----------------------------------------------------

        配送失敗比例=(

            "failure_flag",

            "mean"

        )

    )

)


# ============================================================
# STEP 9：整理數值格式
# ============================================================

customer_monthly["平均重量kg"] = (

    customer_monthly["平均重量kg"]

    .round(2)

)


customer_monthly["平均尺寸總和cm"] = (

    customer_monthly["平均尺寸總和cm"]

    .round(2)

)


# ------------------------------------------------------------
# 百分比欄位
# ------------------------------------------------------------

ratio_columns = [

    "s60比例",

    "s90比例",

    "s120比例",

    "s150比例",

    "偏遠配送比例",

    "低溫配送比例",

    "COD比例",

    "指定時段比例",

    "二次配送比例",

    "配送失敗比例"

]


for col in ratio_columns:

    customer_monthly[col] = (

        customer_monthly[col]

        * 100

    ).round(2)


# ============================================================
# STEP 10：顯示 Customer-Month 結果
# ============================================================

print("\n")

print("==============================================")
print("Customer-Month 特徵轉換完成")
print("==============================================")


print(

    "原始 WMS 資料筆數：",

    len(df)

)


print(

    "企業客戶數：",

    customer_monthly["客戶名稱"].nunique()

)


print(

    "月份數：",

    customer_monthly["月份"].nunique()

)


print(

    "Customer-Month 資料筆數：",

    len(customer_monthly)

)


print("\n前 20 筆 Customer-Month 資料：")


display(

    customer_monthly.head(20)

)


# ============================================================
# STEP 11：建立英文顯示名稱
# ============================================================

# 中文企業名稱對應英文名稱
customer_name_map = {

    "全聯福利中心":

    "PX Mart",


    "momo購物網":

    "momo",


    "PChome":

    "PChome",


    "食品企業A":

    "Food Company A",


    "製造企業B":

    "Manufacturing Company B"

}


# 建立英文名稱欄位
customer_monthly["Customer_EN"] = (

    customer_monthly["客戶名稱"]

    .map(customer_name_map)

    .fillna(customer_monthly["客戶名稱"])

)


# ============================================================
# STEP 12：各企業月出貨量比較表
# ============================================================

volume_table = (

    customer_monthly

    .pivot(

        index="月份",

        columns="Customer_EN",

        values="月出貨件數"

    )

)


print("\n")

print("==============================================")
print("各企業月出貨量")
print("==============================================")


display(

    volume_table

)


# ============================================================
# STEP 13：圖表 1
# Monthly Shipment Volume
# ============================================================

plt.figure(

    figsize=(12, 6)

)


for customer in customer_monthly["Customer_EN"].unique():

    temp = customer_monthly[

        customer_monthly["Customer_EN"]

        == customer

    ]


    plt.plot(

        temp["月份"],

        temp["月出貨件數"],

        marker="o",

        linewidth=2,

        label=customer

    )


plt.title(

    "Monthly Shipment Volume by Customer",

    fontsize=16

)


plt.xlabel(

    "Month",

    fontsize=12

)


plt.ylabel(

    "Shipment Volume",

    fontsize=12

)


plt.legend(

    title="Customer"

)


plt.grid(

    alpha=0.3

)


plt.xticks(

    rotation=45

)


plt.tight_layout()


plt.show()


# ============================================================
# STEP 14：建立平均物流特徵
# ============================================================

customer_profile = (

    customer_monthly

    .groupby(

        [

            "客戶代碼",

            "Customer_EN"

        ],

        as_index=False

    )

    .agg(

        Avg_Monthly_Volume=(

            "月出貨件數",

            "mean"

        ),

        Avg_Weight=(

            "平均重量kg",

            "mean"

        ),

        Avg_Size=(

            "平均尺寸總和cm",

            "mean"

        ),

        Remote_Ratio=(

            "偏遠配送比例",

            "mean"

        ),

        Cold_Ratio=(

            "低溫配送比例",

            "mean"

        ),

        COD_Ratio=(

            "COD比例",

            "mean"

        ),

        TimeSlot_Ratio=(

            "指定時段比例",

            "mean"

        ),

        Redelivery_Ratio=(

            "二次配送比例",

            "mean"

        ),

        Failure_Ratio=(

            "配送失敗比例",

            "mean"

        )

    )

)


# 四捨五入
numeric_cols = customer_profile.select_dtypes(

    include=np.number

).columns


customer_profile[numeric_cols] = (

    customer_profile[numeric_cols]

    .round(2)

)


print("\n")

print("==============================================")
print("Customer Profile Summary")
print("==============================================")


display(

    customer_profile

)


# ============================================================
# STEP 15：圖表 2
# Customer Logistics Characteristics
# ============================================================

features_to_plot = [

    "Remote_Ratio",

    "Cold_Ratio",

    "COD_Ratio",

    "TimeSlot_Ratio",

    "Redelivery_Ratio"

]


plot_data = (

    customer_profile

    .set_index("Customer_EN")[features_to_plot]

)


ax = plot_data.plot(

    kind="bar",

    figsize=(13, 7)

)


plt.title(

    "Customer Logistics Characteristics",

    fontsize=16

)


plt.xlabel(

    "Customer",

    fontsize=12

)


plt.ylabel(

    "Ratio (%)",

    fontsize=12

)


plt.legend(

    title="Feature"

)


plt.xticks(

    rotation=25,

    ha="right"

)


plt.grid(

    axis="y",

    alpha=0.3

)


plt.tight_layout()


plt.show()


# ============================================================
# STEP 16：圖表 3
# Average Weight
# ============================================================

plt.figure(

    figsize=(10, 6)

)


plt.bar(

    customer_profile["Customer_EN"],

    customer_profile["Avg_Weight"]

)


plt.title(

    "Average Package Weight by Customer",

    fontsize=16

)


plt.xlabel(

    "Customer",

    fontsize=12

)


plt.ylabel(

    "Average Weight (kg)",

    fontsize=12

)


plt.xticks(

    rotation=25,

    ha="right"

)


plt.grid(

    axis="y",

    alpha=0.3

)


plt.tight_layout()


plt.show()


# ============================================================
# STEP 17：圖表 4
# Average Package Size
# ============================================================

plt.figure(

    figsize=(10, 6)

)


plt.bar(

    customer_profile["Customer_EN"],

    customer_profile["Avg_Size"]

)


plt.title(

    "Average Package Size by Customer",

    fontsize=16

)


plt.xlabel(

    "Customer",

    fontsize=12

)


plt.ylabel(

    "Average Length + Width + Height (cm)",

    fontsize=12

)


plt.xticks(

    rotation=25,

    ha="right"

)


plt.grid(

    axis="y",

    alpha=0.3

)


plt.tight_layout()


plt.show()


# ============================================================
# STEP 18：建立英文版 K-Means 準備表
# ============================================================

# 後續 STEP 3 K-Means 可以直接使用這張表

kmeans_features = customer_monthly[[

    "客戶代碼",

    "Customer_EN",

    "月份",

    "月出貨件數",

    "平均重量kg",

    "平均尺寸總和cm",

    "s60比例",

    "s90比例",

    "s120比例",

    "s150比例",

    "配送縣市數",

    "偏遠配送比例",

    "低溫配送比例",

    "COD比例",

    "指定時段比例",

    "二次配送比例",

    "配送失敗比例"

]].copy()


# 將欄位改成英文
kmeans_features.columns = [

    "Customer_ID",

    "Customer",

    "Month",

    "Monthly_Volume",

    "Avg_Weight",

    "Avg_Size",

    "S60_Ratio",

    "S90_Ratio",

    "S120_Ratio",

    "S150_Ratio",

    "Delivery_City_Count",

    "Remote_Ratio",

    "Cold_Ratio",

    "COD_Ratio",

    "TimeSlot_Ratio",

    "Redelivery_Ratio",

    "Failure_Ratio"

]


print("\n")

print("==============================================")
print("K-Means Feature Dataset")
print("==============================================")


display(

    kmeans_features.head(20)

)


# ============================================================
# STEP 19：輸出 Excel
# ============================================================

output_file = (

    "customer_monthly_features.xlsx"

)


with pd.ExcelWriter(

    output_file,

    engine="openpyxl"

) as writer:


    # --------------------------------------------------------
    # 中文 Customer-Month
    # --------------------------------------------------------

    customer_monthly.drop(

        columns=["Customer_EN"]

    ).to_excel(

        writer,

        sheet_name="Customer-Month",

        index=False

    )


    # --------------------------------------------------------
    # 英文 K-Means 特徵資料
    # --------------------------------------------------------

    kmeans_features.to_excel(

        writer,

        sheet_name="KMeans_Features",

        index=False

    )


    # --------------------------------------------------------
    # 客戶平均特徵
    # --------------------------------------------------------

    customer_profile.to_excel(

        writer,

        sheet_name="Customer_Profile",

        index=False

    )


    # --------------------------------------------------------
    # 月出貨量
    # --------------------------------------------------------

    volume_table.to_excel(

        writer,

        sheet_name="Monthly_Volume"

    )


print("\n")

print("==============================================")
print("Excel 輸出完成")
print("==============================================")


print(

    "檔案名稱：",

    output_file

)


# ============================================================
# STEP 20：下載 Excel
# ============================================================

files.download(

    output_file

)
