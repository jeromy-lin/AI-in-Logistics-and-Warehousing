# ============================================================
# 物流業 WMS｜AI 智慧定價戰情看板
# Full Google Colab + XGBoost + Gradio Web App
# 作者：國立雲林科技大學電機系 林家仁
# 功能：
# 1. 上傳 WMS Excel
# 2. 自動讀取 WMS_Package_Data
# 3. Customer-based Train/Test Split
# 4. 訓練 XGBoost Cost-to-Serve Model
# 5. 顯示 R² / MAE / RMSE / MAPE
# 6. 啟動 Gradio Web App
# 7. 中文化操作介面
# 8. 動態 Target Margin
# 9. Enterprise Minimum Price
# 10. AI Suggested Price
# 11. Final Quote
# 12. Expected Margin
# 13. Markup
# 14. Pricing Status
# 15. 中文自動解說
# ============================================================


# ============================================================
# STEP 0｜安裝套件
# ============================================================

!pip -q install xgboost gradio openpyxl scikit-learn


# ============================================================
# STEP 1｜載入套件
# ============================================================

import pandas as pd
import numpy as np
import gradio as gr
import warnings

warnings.filterwarnings("ignore")

from google.colab import files

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from xgboost import XGBRegressor


# ============================================================
# STEP 2｜上傳 WMS Excel
# ============================================================

print("📁 請上傳物流業 WMS Excel 檔案")
print("")

uploaded = files.upload()

filename = list(uploaded.keys())[0]

print("")
print("✅ 已成功上傳：", filename)


# ============================================================
# STEP 3｜讀取 WMS_Package_Data
# ============================================================

df = pd.read_excel(
    filename,
    sheet_name="WMS_Package_Data"
)

# 清除 Excel 多餘欄位
df = df.loc[
    :,
    ~df.columns.astype(str).str.startswith("Unnamed")
]

if "index" in df.columns:
    df = df.drop(columns=["index"])


print("")
print("=" * 50)
print("WMS DATA SUMMARY")
print("=" * 50)

print(
    "Package Records :",
    f"{len(df):,}"
)

print(
    "Customers       :",
    f"{df['Customer_ID'].nunique():,}"
)

print(
    "Columns         :",
    len(df.columns)
)


# ============================================================
# STEP 4｜定義 Features
# ============================================================

numeric_features = [

    "Monthly_Target_Packages",

    "Weight_kg",

    "Distance_km",

    "Remote_Flag",

    "COD_Flag",

    "COD_Amount",

    "Insurance_Flag",

    "Declared_Value",

    "Redelivery_Flag",

    "SameDay_Flag",

    "StorePickup_Flag",

    "Pickup_Density",

    "Customer_Cold_Ratio",

    "Customer_Remote_Ratio",

    "Customer_Redelivery_Rate"
]


categorical_features = [

    "Package_Size",

    "Temperature",

    "Destination_Region",

    "Customer_Type",

    "Industry"
]


# ============================================================
# STEP 5｜檢查 Excel 欄位
# ============================================================

required_columns = (

    numeric_features

    + categorical_features

    + [

        "Actual_Operational_Cost",

        "Customer_ID"
    ]
)


missing_columns = [

    col
    for col in required_columns
    if col not in df.columns
]


if len(missing_columns) > 0:

    raise ValueError(
        "Excel 缺少以下欄位：\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# STEP 6｜建立 X / y
# ============================================================

X = df[
    numeric_features
    + categorical_features
].copy()


y = df[
    "Actual_Operational_Cost"
].copy()


# ============================================================
# STEP 7｜資料前處理
# ============================================================

numeric_transformer = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_transformer = Pipeline(

    steps=[

        (
            "imputer",

            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",

            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",

            numeric_transformer,

            numeric_features
        ),

        (
            "cat",

            categorical_transformer,

            categorical_features
        )
    ]
)


# ============================================================
# STEP 8｜Customer-Based Train / Test Split
# ============================================================

groups = df[
    "Customer_ID"
]


splitter = GroupShuffleSplit(

    n_splits=1,

    test_size=0.25,

    random_state=42
)


train_idx, test_idx = next(

    splitter.split(
        X,
        y,
        groups
    )
)


X_train = X.iloc[
    train_idx
].copy()


X_test = X.iloc[
    test_idx
].copy()


y_train = y.iloc[
    train_idx
].copy()


y_test = y.iloc[
    test_idx
].copy()


training_customers = (
    df
    .iloc[train_idx][
        "Customer_ID"
    ]
    .nunique()
)


testing_customers = (
    df
    .iloc[test_idx][
        "Customer_ID"
    ]
    .nunique()
)


print("")
print("=" * 50)
print("TRAIN / TEST SPLIT")
print("=" * 50)

print(
    "Training Records   :",
    f"{len(X_train):,}"
)

print(
    "Testing Records    :",
    f"{len(X_test):,}"
)

print(
    "Training Customers :",
    training_customers
)

print(
    "Testing Customers  :",
    testing_customers
)


# ============================================================
# STEP 9｜建立 XGBoost
# ============================================================

model = Pipeline(

    steps=[

        (
            "preprocessor",

            preprocessor
        ),

        (
            "xgboost",

            XGBRegressor(

                n_estimators=500,

                learning_rate=0.05,

                max_depth=5,

                subsample=0.85,

                colsample_bytree=0.85,

                reg_lambda=1.0,

                reg_alpha=0.0,

                objective="reg:squarederror",

                eval_metric="rmse",

                random_state=42,

                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# STEP 10｜訓練 XGBoost
# ============================================================

print("")
print("🚀 Training XGBoost Model...")

model.fit(
    X_train,
    y_train
)

print("✅ XGBoost Model Training Completed")


# ============================================================
# STEP 11｜Testing Prediction
# ============================================================

pred_test = model.predict(
    X_test
)


# ============================================================
# STEP 12｜模型績效
# ============================================================

r2 = r2_score(
    y_test,
    pred_test
)


mae = mean_absolute_error(
    y_test,
    pred_test
)


rmse = np.sqrt(

    mean_squared_error(
        y_test,
        pred_test
    )
)


mape = (

    mean_absolute_percentage_error(
        y_test,
        pred_test
    )

    * 100
)


print("")
print("=" * 50)
print("XGBOOST MODEL PERFORMANCE")
print("=" * 50)

print(
    f"R²   : {r2:.4f}"
)

print(
    f"MAE  : ${mae:.2f}"
)

print(
    f"RMSE : ${rmse:.2f}"
)

print(
    f"MAPE : {mape:.2f}%"
)


# ============================================================
# STEP 13｜預設資料
# ============================================================

default_row = df.iloc[0]


# ============================================================
# STEP 14｜智慧定價函數
# ============================================================

def smart_pricing(

    monthly_volume,

    customer_type,

    industry,

    package_size,

    weight,

    temperature,

    region,

    remote,

    distance,

    cod,

    cod_amount,

    insurance,

    declared_value,

    redelivery,

    same_day,

    pickup_density,

    target_margin,

    minimum_price
):


    # --------------------------------------------------------
    # 基本型別轉換
    # --------------------------------------------------------

    target_margin = float(
        target_margin
    )

    minimum_price = float(
        minimum_price
    )

    monthly_volume = float(
        monthly_volume
    )

    weight = float(
        weight
    )

    distance = float(
        distance
    )

    cod_amount = float(
        cod_amount
    )

    declared_value = float(
        declared_value
    )

    pickup_density = float(
        pickup_density
    )


    # --------------------------------------------------------
    # 毛利率防呆
    # --------------------------------------------------------

    if target_margin >= 100:

        error_html = """

        <div style="
            background:#FDEDEC;
            border-left:7px solid #C0392B;
            padding:20px;
            border-radius:12px;
            font-size:18px;
        ">

            ❌ 目標毛利率必須低於 100%

        </div>

        """

        return (
            error_html,
            pd.DataFrame()
        )


    # --------------------------------------------------------
    # 建立單筆 Package 資料
    # --------------------------------------------------------

    new_data = pd.DataFrame({

        "Monthly_Target_Packages": [
            monthly_volume
        ],

        "Weight_kg": [
            weight
        ],

        "Distance_km": [
            distance
        ],

        "Remote_Flag": [

            1
            if remote == "Yes"
            else 0
        ],

        "COD_Flag": [

            1
            if cod == "Yes"
            else 0
        ],

        "COD_Amount": [
            cod_amount
        ],

        "Insurance_Flag": [

            1
            if insurance == "Yes"
            else 0
        ],

        "Declared_Value": [
            declared_value
        ],

        "Redelivery_Flag": [

            1
            if redelivery == "Yes"
            else 0
        ],

        "SameDay_Flag": [

            1
            if same_day == "Yes"
            else 0
        ],

        "StorePickup_Flag": [

            int(
                default_row[
                    "StorePickup_Flag"
                ]
            )
        ],

        "Pickup_Density": [
            pickup_density
        ],

        "Customer_Cold_Ratio": [

            float(
                default_row[
                    "Customer_Cold_Ratio"
                ]
            )
        ],

        "Customer_Remote_Ratio": [

            float(
                default_row[
                    "Customer_Remote_Ratio"
                ]
            )
        ],

        "Customer_Redelivery_Rate": [

            float(
                default_row[
                    "Customer_Redelivery_Rate"
                ]
            )
        ],

        "Package_Size": [
            package_size
        ],

        "Temperature": [
            temperature
        ],

        "Destination_Region": [
            region
        ],

        "Customer_Type": [
            customer_type
        ],

        "Industry": [
            industry
        ]
    })


    # ========================================================
    # XGBoost 預測成本
    # ========================================================

    predicted_cost = float(

        model.predict(
            new_data
        )[0]
    )


    # ========================================================
    # Target Margin
    # ========================================================

    margin_decimal = (

        target_margin

        / 100
    )


    # ========================================================
    # AI Suggested Price
    # ========================================================

    ai_suggested_price = (

        predicted_cost

        /

        (
            1
            - margin_decimal
        )
    )


    # ========================================================
    # Enterprise Minimum Pricing Rule
    # ========================================================

    final_quote = max(

        ai_suggested_price,

        minimum_price
    )


    # ========================================================
    # Expected Margin
    # ========================================================

    expected_margin = (

        (
            final_quote
            - predicted_cost
        )

        /

        final_quote

        * 100
    )


    # ========================================================
    # Markup
    # ========================================================

    markup = (

        (
            final_quote
            - predicted_cost
        )

        /

        predicted_cost

        * 100
    )


    # ========================================================
    # Pricing Status + 中文解說
    # ========================================================

    if minimum_price > ai_suggested_price:

        pricing_status = (
            "🟠 企業規則保護｜Rule Protected"
        )

        pricing_reason = (

            "企業最低報價高於 AI 建議價，"
            "因此系統採用企業最低報價，"
            "避免報價低於公司的價格底線。"
        )

        status_color = "#FFF4E5"

        status_border = "#E39B35"


        explanation = f"""

        XGBoost 預測這筆物流服務的
        <b>Cost-to-Serve 為 ${predicted_cost:,.2f}</b>。

        目前設定的
        <b>目標毛利率為 {target_margin:.0f}%</b>，

        因此 AI 依照成本與毛利率計算出的
        <b>建議報價為 ${ai_suggested_price:,.2f}</b>。

        但是企業設定的最低報價為
        <b>${minimum_price:,.2f}</b>，
        高於 AI 建議價。

        因此系統啟動
        <b>Enterprise Minimum Pricing Rule</b>，
        最終建議報價採用
        <b>${final_quote:,.2f}</b>。

        依照這個價格計算，
        預估毛利率為
        <b>{expected_margin:.2f}%</b>。

        """


    else:

        pricing_status = (
            "🟢 毛利達標｜Margin Protected"
        )

        pricing_reason = (

            "AI 建議價格已達成目前設定的"
            "目標毛利率，且未低於企業最低報價。"
        )

        status_color = "#EFF7EC"

        status_border = "#7F9273"


        explanation = f"""

        XGBoost 預測這筆物流服務的
        <b>Cost-to-Serve 為 ${predicted_cost:,.2f}</b>。

        目前設定的
        <b>目標毛利率為 {target_margin:.0f}%</b>，

        因此 AI 計算出的
        <b>建議報價為 ${ai_suggested_price:,.2f}</b>。

        此價格高於企業最低報價
        <b>${minimum_price:,.2f}</b>，

        因此系統直接採用
        <b>AI Suggested Price</b>
        作為最終建議報價。

        最終報價為
        <b>${final_quote:,.2f}</b>，

        預估毛利率為
        <b>{expected_margin:.2f}%</b>。

        """


    # ========================================================
    # HTML Dashboard
    # ========================================================

    html = f"""

    <!-- ================================================== -->
    <!-- Title -->
    <!-- ================================================== -->

    <div style="
        background:linear-gradient(
            135deg,
            #FFF8F0,
            #F7C9A9
        );
        padding:22px 26px;
        border-radius:16px;
        margin-bottom:18px;
        border-left:8px solid #C96F5B;
    ">

        <div style="
            font-size:28px;
            font-weight:900;
            color:#4F3A30;
        ">
            AI 智慧定價結果
        </div>

        <div style="
            color:#746C67;
            font-size:14px;
            margin-top:7px;
            line-height:1.8;
        ">

            XGBoost 成本預測
            →
            目標毛利率
            →
            AI 建議報價
            →
            企業最低報價
            →
            最終建議報價

        </div>

    </div>


    <!-- ================================================== -->
    <!-- KPI Cards -->
    <!-- ================================================== -->

    <div style="
        display:grid;
        grid-template-columns:
        repeat(5,minmax(145px,1fr));
        gap:12px;
    ">


        <!-- Predicted Cost -->

        <div style="
            background:#EFF7EC;
            padding:18px;
            border-radius:14px;
            text-align:center;
        ">

            <div style="
                font-size:13px;
                color:#666;
            ">
                AI 預測成本
            </div>

            <div style="
                font-size:30px;
                font-weight:900;
                color:#4F3A30;
                margin-top:6px;
            ">
                ${predicted_cost:,.2f}
            </div>

            <div style="
                font-size:12px;
                color:#746C67;
                margin-top:7px;
            ">
                XGBoost 預測的<br>
                Cost-to-Serve
            </div>

        </div>


        <!-- Target Margin -->

        <div style="
            background:#FFF4DC;
            padding:18px;
            border-radius:14px;
            text-align:center;
        ">

            <div style="
                font-size:13px;
                color:#666;
            ">
                目標毛利率
            </div>

            <div style="
                font-size:30px;
                font-weight:900;
                color:#4F3A30;
                margin-top:6px;
            ">
                {target_margin:.0f}%
            </div>

            <div style="
                font-size:12px;
                color:#746C67;
                margin-top:7px;
            ">
                Target Margin
            </div>

        </div>


        <!-- AI Suggested Price -->

        <div style="
            background:#FCEEE5;
            padding:18px;
            border-radius:14px;
            text-align:center;
        ">

            <div style="
                font-size:13px;
                color:#666;
            ">
                AI 建議報價
            </div>

            <div style="
                font-size:30px;
                font-weight:900;
                color:#4F3A30;
                margin-top:6px;
            ">
                ${ai_suggested_price:,.2f}
            </div>

            <div style="
                font-size:12px;
                color:#746C67;
                margin-top:7px;
            ">
                Cost ÷ (1 − Margin)
            </div>

        </div>


        <!-- Final Quote -->

        <div style="
            background:#F8E3DF;
            padding:18px;
            border-radius:14px;
            text-align:center;
            border:2px solid #E9B5A8;
        ">

            <div style="
                font-size:13px;
                color:#666;
            ">
                最終建議報價
            </div>

            <div style="
                font-size:34px;
                font-weight:900;
                color:#C96F5B;
                margin-top:4px;
            ">
                ${final_quote:,.2f}
            </div>

            <div style="
                font-size:12px;
                color:#746C67;
                margin-top:6px;
            ">
                AI 建議價與企業底價<br>
                取較高者
            </div>

        </div>


        <!-- Expected Margin -->

        <div style="
            background:#EEEAF7;
            padding:18px;
            border-radius:14px;
            text-align:center;
        ">

            <div style="
                font-size:13px;
                color:#666;
            ">
                預估毛利率
            </div>

            <div style="
                font-size:30px;
                font-weight:900;
                color:#4F3A30;
                margin-top:6px;
            ">
                {expected_margin:.1f}%
            </div>

            <div style="
                font-size:12px;
                color:#746C67;
                margin-top:7px;
            ">
                Expected Margin
            </div>

        </div>

    </div>


    <!-- ================================================== -->
    <!-- Pricing Status -->
    <!-- ================================================== -->

    <div style="
        background:{status_color};
        border-left:7px solid {status_border};
        padding:17px 20px;
        border-radius:12px;
        margin-top:18px;
    ">

        <div style="
            font-size:20px;
            font-weight:900;
            color:#4F3A30;
        ">
            {pricing_status}
        </div>

        <div style="
            font-size:14px;
            color:#6D5142;
            margin-top:6px;
            line-height:1.8;
        ">
            {pricing_reason}
        </div>

    </div>


    <!-- ================================================== -->
    <!-- System Explanation -->
    <!-- ================================================== -->

    <div style="
        background:#FFFFFF;
        border:1px solid #EADDD3;
        padding:20px 22px;
        border-radius:14px;
        margin-top:18px;
    ">

        <div style="
            font-size:20px;
            font-weight:900;
            color:#4F3A30;
            margin-bottom:10px;
        ">
            📘 系統自動解說
        </div>

        <div style="
            font-size:15px;
            color:#5F514A;
            line-height:2;
        ">

            {explanation}

        </div>

    </div>


    <!-- ================================================== -->
    <!-- Formula -->
    <!-- ================================================== -->

    <div style="
        background:#FFF8F0;
        padding:18px 20px;
        border-radius:12px;
        margin-top:16px;
    ">

        <div style="
            font-size:18px;
            font-weight:800;
            color:#4F3A30;
            margin-bottom:10px;
        ">
            🧮 定價公式
        </div>

        <div style="
            font-size:15px;
            color:#6D5142;
            line-height:2;
        ">

            <b>AI 建議報價：</b><br>

            AI Suggested Price
            =
            Predicted Cost ÷
            (1 − Target Margin)

            <br><br>

            <b>最終建議報價：</b><br>

            Final Quote
            =
            MAX(
            AI Suggested Price,
            Enterprise Minimum Price
            )

        </div>

    </div>


    <!-- ================================================== -->
    <!-- Metric Explanation -->
    <!-- ================================================== -->

    <div style="
        background:#F9F5F1;
        padding:20px 22px;
        border-radius:12px;
        margin-top:16px;
    ">

        <div style="
            font-size:18px;
            font-weight:900;
            color:#4F3A30;
            margin-bottom:10px;
        ">
            📖 指標怎麼看？
        </div>

        <div style="
            font-size:14px;
            color:#6D5142;
            line-height:2;
        ">

            <b>AI 預測成本：</b>
            XGBoost 預估完成這筆物流服務所需的營運成本。
            <br>

            <b>目標毛利率：</b>
            企業希望這筆物流服務達成的最低毛利目標。
            <br>

            <b>AI 建議報價：</b>
            根據 AI 預測成本與目標毛利率所計算的建議價格。
            <br>

            <b>企業最低報價：</b>
            公司規定不能低於的價格底線。
            <br>

            <b>最終建議報價：</b>
            AI 建議價與企業最低報價兩者取較高者。
            <br>

            <b>預估毛利率：</b>
            以最終建議報價與 AI 預測成本計算的實際預估毛利率。
            <br>

            <b>Markup：</b>
            售價相較於成本增加多少百分比。
            Markup 與 Margin 的分母不同，因此數值不會相同。

        </div>

    </div>

    """


    # ========================================================
    # Summary Table
    # ========================================================

    result_table = pd.DataFrame({

        "定價指標 Pricing Metric": [

            "AI 預測成本",

            "目標毛利率",

            "AI 建議報價",

            "企業最低報價",

            "最終建議報價",

            "預估毛利率",

            "Markup",

            "定價狀態"
        ],

        "結果 Result": [

            f"${predicted_cost:,.2f}",

            f"{target_margin:.0f}%",

            f"${ai_suggested_price:,.2f}",

            f"${minimum_price:,.2f}",

            f"${final_quote:,.2f}",

            f"{expected_margin:.2f}%",

            f"{markup:.2f}%",

            pricing_status
        ]
    })


    return (
        html,
        result_table
    )


# ============================================================
# STEP 15｜建立 Gradio Web App
# ============================================================

with gr.Blocks(
    title="物流業 WMS AI 智慧定價戰情看板"
) as app:


    # ========================================================
    # Header
    # ========================================================

    gr.HTML("""

    <div style="
        background:linear-gradient(
            135deg,
            #FFF8F0,
            #F7C9A9
        );
        padding:25px 30px;
        border-radius:18px;
        margin-bottom:20px;
        border-left:9px solid #C96F5B;
    ">

        <div style="
            font-size:32px;
            font-weight:900;
            color:#4F3A30;
        ">
            🚚 物流業 WMS｜AI 智慧定價戰情看板
        </div>

        <div style="
            color:#746C67;
            font-size:16px;
            margin-top:8px;
        ">
            XGBoost Cost-to-Serve Prediction
            + Dynamic Smart Pricing
        </div>

        <div style="
            color:#6D5142;
            font-size:14px;
            margin-top:12px;
        ">

            操作流程：
            客戶條件
            →
            包裹條件
            →
            配送條件
            →
            定價策略
            →
            AI 最終建議報價

        </div>

    </div>

    """)


    # ========================================================
    # Model Performance
    # ========================================================

    gr.Markdown(
        "## 📊 XGBoost 模型績效"
    )


    gr.HTML(f"""

    <div style="
        display:grid;
        grid-template-columns:repeat(4,1fr);
        gap:12px;
        margin-bottom:22px;
    ">

        <div style="
            background:#EFF7EC;
            padding:16px;
            border-radius:12px;
            text-align:center;
        ">

            <div>
                R²
            </div>

            <div style="
                font-size:28px;
                font-weight:900;
            ">
                {r2:.4f}
            </div>

            <small>
                解釋能力，越高越好
            </small>

        </div>


        <div style="
            background:#FFF4DC;
            padding:16px;
            border-radius:12px;
            text-align:center;
        ">

            <div>
                MAE
            </div>

            <div style="
                font-size:28px;
                font-weight:900;
            ">
                ${mae:.2f}
            </div>

            <small>
                平均成本誤差
            </small>

        </div>


        <div style="
            background:#FCEEE5;
            padding:16px;
            border-radius:12px;
            text-align:center;
        ">

            <div>
                RMSE
            </div>

            <div style="
                font-size:28px;
                font-weight:900;
            ">
                ${rmse:.2f}
            </div>

            <small>
                大型誤差風險
            </small>

        </div>


        <div style="
            background:#EEEAF7;
            padding:16px;
            border-radius:12px;
            text-align:center;
        ">

            <div>
                MAPE
            </div>

            <div style="
                font-size:28px;
                font-weight:900;
            ">
                {mape:.2f}%
            </div>

            <small>
                平均百分比誤差
            </small>

        </div>

    </div>

    """)


    # ========================================================
    # Tabs
    # ========================================================

    with gr.Tabs():


        # ====================================================
        # ① Customer
        # ====================================================

        with gr.Tab(
            "① 客戶條件 Customer"
        ):


            gr.Markdown(
                """
                ### 👤 客戶條件

                這一頁回答：

                **「這是哪一種類型的客戶？」**

                出貨量與客戶型態可能影響規模經濟與物流成本。
                """
            )


            with gr.Row():


                monthly_volume = gr.Slider(

                    minimum=int(
                        df[
                            "Monthly_Target_Packages"
                        ].min()
                    ),

                    maximum=int(
                        df[
                            "Monthly_Target_Packages"
                        ].max()
                    ),

                    value=int(
                        default_row[
                            "Monthly_Target_Packages"
                        ]
                    ),

                    step=100,

                    label=(
                        "每月預估出貨量 "
                        "Monthly Package Volume"
                    )
                )


                customer_type = gr.Dropdown(

                    choices=sorted(

                        df[
                            "Customer_Type"
                        ]

                        .dropna()

                        .astype(str)

                        .unique()

                        .tolist()
                    ),

                    value=str(
                        default_row[
                            "Customer_Type"
                        ]
                    ),

                    label=(
                        "客戶類型 Customer Type"
                    )
                )


                industry = gr.Dropdown(

                    choices=sorted(

                        df[
                            "Industry"
                        ]

                        .dropna()

                        .astype(str)

                        .unique()

                        .tolist()
                    ),

                    value=str(
                        default_row[
                            "Industry"
                        ]
                    ),

                    label=(
                        "產業別 Industry"
                    )
                )


        # ====================================================
        # ② Package
        # ====================================================

        with gr.Tab(
            "② 包裹條件 Package"
        ):


            gr.Markdown(
                """
                ### 📦 包裹條件

                這一頁回答：

                **「我們要送的是什麼貨？」**

                包裹材積、重量、溫層、COD 與保險等，
                都可能直接影響 Cost-to-Serve。
                """
            )


            with gr.Row():


                package_size = gr.Dropdown(

                    choices=sorted(

                        df[
                            "Package_Size"
                        ]

                        .dropna()

                        .astype(str)

                        .unique()

                        .tolist()
                    ),

                    value=str(
                        default_row[
                            "Package_Size"
                        ]
                    ),

                    label=(
                        "包裹材積 Package Size"
                    )
                )


                weight = gr.Slider(

                    minimum=float(
                        df[
                            "Weight_kg"
                        ].min()
                    ),

                    maximum=float(
                        df[
                            "Weight_kg"
                        ].max()
                    ),

                    value=float(
                        default_row[
                            "Weight_kg"
                        ]
                    ),

                    step=0.1,

                    label=(
                        "包裹重量 Weight (kg)"
                    )
                )


                temperature = gr.Dropdown(

                    choices=sorted(

                        df[
                            "Temperature"
                        ]

                        .dropna()

                        .astype(str)

                        .unique()

                        .tolist()
                    ),

                    value=str(
                        default_row[
                            "Temperature"
                        ]
                    ),

                    label=(
                        "配送溫層 Temperature"
                    )
                )


            with gr.Row():


                cod = gr.Radio(

                    choices=[
                        "No",
                        "Yes"
                    ],

                    value=(

                        "Yes"

                        if int(
                            default_row[
                                "COD_Flag"
                            ]
                        ) == 1

                        else "No"
                    ),

                    label=(
                        "是否 COD 代收"
                    )
                )


                cod_amount = gr.Number(

                    value=float(
                        default_row[
                            "COD_Amount"
                        ]
                    ),

                    label=(
                        "COD 代收金額"
                    )
                )


                insurance = gr.Radio(

                    choices=[
                        "No",
                        "Yes"
                    ],

                    value=(

                        "Yes"

                        if int(
                            default_row[
                                "Insurance_Flag"
                            ]
                        ) == 1

                        else "No"
                    ),

                    label=(
                        "是否投保 Insurance"
                    )
                )


                declared_value = gr.Number(

                    value=float(
                        default_row[
                            "Declared_Value"
                        ]
                    ),

                    label=(
                        "商品申報價值 Declared Value"
                    )
                )


        # ====================================================
        # ③ Delivery
        # ====================================================

        with gr.Tab(
            "③ 配送條件 Delivery"
        ):


            gr.Markdown(
                """
                ### 🚚 配送條件

                這一頁回答：

                **「這件貨要送到哪裡？怎麼送？」**

                距離、偏遠、二次配送與當日配送，
                都可能使履約成本增加。
                """
            )


            with gr.Row():


                region = gr.Dropdown(

                    choices=sorted(

                        df[
                            "Destination_Region"
                        ]

                        .dropna()

                        .astype(str)

                        .unique()

                        .tolist()
                    ),

                    value=str(
                        default_row[
                            "Destination_Region"
                        ]
                    ),

                    label=(
                        "配送區域 Destination Region"
                    )
                )


                remote = gr.Radio(

                    choices=[
                        "No",
                        "Yes"
                    ],

                    value=(

                        "Yes"

                        if int(
                            default_row[
                                "Remote_Flag"
                            ]
                        ) == 1

                        else "No"
                    ),

                    label=(
                        "是否偏遠配送 Remote Delivery"
                    )
                )


                distance = gr.Slider(

                    minimum=float(
                        df[
                            "Distance_km"
                        ].min()
                    ),

                    maximum=float(
                        df[
                            "Distance_km"
                        ].max()
                    ),

                    value=float(
                        default_row[
                            "Distance_km"
                        ]
                    ),

                    step=1,

                    label=(
                        "配送距離 Distance (km)"
                    )
                )


            with gr.Row():


                redelivery = gr.Radio(

                    choices=[
                        "No",
                        "Yes"
                    ],

                    value=(

                        "Yes"

                        if int(
                            default_row[
                                "Redelivery_Flag"
                            ]
                        ) == 1

                        else "No"
                    ),

                    label=(
                        "是否二次配送 Redelivery"
                    )
                )


                same_day = gr.Radio(

                    choices=[
                        "No",
                        "Yes"
                    ],

                    value=(

                        "Yes"

                        if int(
                            default_row[
                                "SameDay_Flag"
                            ]
                        ) == 1

                        else "No"
                    ),

                    label=(
                        "是否當日配送 Same-Day"
                    )
                )


                pickup_density = gr.Slider(

                    minimum=float(
                        df[
                            "Pickup_Density"
                        ].min()
                    ),

                    maximum=float(
                        df[
                            "Pickup_Density"
                        ].max()
                    ),

                    value=float(
                        default_row[
                            "Pickup_Density"
                        ]
                    ),

                    step=0.1,

                    label=(
                        "取件密度 Pickup Density"
                    )
                )


        # ====================================================
        # ④ Pricing
        # ====================================================

        with gr.Tab(
            "④ 定價策略 Pricing"
        ):


            gr.Markdown(
                """
                ### 💰 定價策略

                這一頁回答：

                **「這筆物流服務希望賺多少？最低可以賣多少？」**

                先調整 **目標毛利率 Target Margin**，
                再輸入 **企業最低報價**。

                系統會自動計算：

                **AI 預測成本  
                → AI 建議報價  
                → 企業最低報價檢查  
                → 最終建議報價**

                > AI 不直接猜售價，而是先預測 Cost-to-Serve，
                > 再加入毛利與企業報價規則。
                """
            )


            with gr.Row():


                target_margin = gr.Slider(

                    minimum=5,

                    maximum=50,

                    value=20,

                    step=1,

                    label=(
                        "目標毛利率 "
                        "Target Margin (%)"
                    )
                )


                minimum_price = gr.Number(

                    value=100,

                    label=(
                        "企業最低報價 "
                        "Enterprise Minimum Price"
                    )
                )


            calculate_button = gr.Button(

                "🚀 計算 AI 智慧建議報價",

                variant="primary",

                size="lg"
            )


    # ========================================================
    # Output
    # ========================================================

    gr.Markdown(
        "## 📊 ⑤ AI 智慧定價結果"
    )


    pricing_cards = gr.HTML()


    pricing_table = gr.Dataframe(

        headers=[

            "定價指標 Pricing Metric",

            "結果 Result"
        ],

        interactive=False
    )


    # ========================================================
    # Click Event
    # ========================================================

    calculate_button.click(

        fn=smart_pricing,

        inputs=[

            monthly_volume,

            customer_type,

            industry,

            package_size,

            weight,

            temperature,

            region,

            remote,

            distance,

            cod,

            cod_amount,

            insurance,

            declared_value,

            redelivery,

            same_day,

            pickup_density,

            target_margin,

            minimum_price
        ],

        outputs=[

            pricing_cards,

            pricing_table
        ]
    )


# ============================================================
# STEP 16｜啟動 Web App
# ============================================================

print("")
print("=" * 50)
print("🚀 啟動 AI Smart Pricing Web App")
print("=" * 50)

print("")
print("完成後請點擊下方 Gradio Public URL")
print("")


app.launch(
    share=True,
    debug=False
)
