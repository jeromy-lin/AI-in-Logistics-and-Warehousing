# ==========================================
# SVM 儲位類別預測範例
# 作者：國立雲林科技大學電機系 林家仁
# 使用資料：Warehouse_KMeans_SKU_predict_70.xlsx#
# 說明：
# 1. 仿照 K-Means 程式，支援 Colab 上傳 Excel 或本機讀檔
# 2. 使用 50 筆 Train 商品的「建議儲位策略」作為分類答案
# 3. 訓練 SVM 支援向量機模型
# 4. 預測 20 筆 Predict 新品應放在哪一類儲位
# 5. 匯出 Excel 結果，並繪製預測類別統計圖
# ==========================================

from __future__ import annotations

import argparse
import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:
    display = print

try:
    from google.colab import files

    IN_COLAB = True
except ImportError:
    files = None
    IN_COLAB = False

if IN_COLAB:
    BASE_DIR = Path.cwd()
else:
    try:
        BASE_DIR = Path(__file__).resolve().parent
    except NameError:
        BASE_DIR = Path.cwd()

os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib-cache"))

import matplotlib

if not IN_COLAB:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.metrics import auc, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import label_binarize
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.svm import SVC
except ImportError as exc:
    raise SystemExit(
        "缺少 scikit-learn 套件。請在 Colab 執行，或先安裝：pip install scikit-learn"
    ) from exc

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.svm._base")


DEFAULT_INPUT_FILE = "Warehouse_KMeans_SKU_predict_70.xlsx"
DEFAULT_OUTPUT_FILE = "Warehouse_SVM_儲位類別預測結果_70.xlsx"
SHEET_NAME = "SKU資料_70"
TARGET_COLUMN = "建議儲位策略"


def get_input_file(cli_input: str | None) -> str:
    """取得輸入檔案。Colab 會跳出上傳，本機則使用預設檔名。"""
    if cli_input:
        return cli_input

    if IN_COLAB:
        print("請上傳 Excel 檔案，例如：Warehouse_KMeans_SKU_predict_70.xlsx")
        uploaded = files.upload()
        if not uploaded:
            raise ValueError("沒有上傳任何檔案，程式停止。")
        file_name = list(uploaded.keys())[0]
        print("已上傳檔案：", file_name)
        return file_name

    default_path = Path(DEFAULT_INPUT_FILE)
    if default_path.exists():
        return str(default_path)

    script_folder_path = BASE_DIR / DEFAULT_INPUT_FILE
    if script_folder_path.exists():
        return str(script_folder_path)

    return DEFAULT_INPUT_FILE


def find_header_row(file_name: str, sheet_name: str = SHEET_NAME) -> int:
    """尋找包含 SKU編號 的表頭列，避免 Excel 前面有課程說明列。"""
    raw_df = pd.read_excel(file_name, sheet_name=sheet_name, header=None)

    for row_index in range(len(raw_df)):
        row_values = raw_df.iloc[row_index].astype(str).tolist()
        if "SKU編號" in row_values:
            return row_index

    raise ValueError("找不到表頭列，請確認工作表內有「SKU編號」欄位。")


def read_inventory_data(file_name: str) -> pd.DataFrame:
    """讀取 SKU 資料表。"""
    header_row = find_header_row(file_name)
    df = pd.read_excel(file_name, sheet_name=SHEET_NAME, header=header_row)
    df = df.dropna(how="all")
    df = df.dropna(subset=["SKU編號"])
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """檢查本範例需要的欄位是否存在。"""
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
        TARGET_COLUMN,
        "資料用途",
        "預測狀態",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError("缺少必要欄位：" + "、".join(missing_columns))


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """清理資料並建立 SVM 可用的數值特徵。"""
    df = df.copy()

    numeric_columns = ["每日出貨箱數", "每日揀貨次數"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    volume_map = {"小": 1, "中": 2, "大": 3}
    weight_map = {"輕": 1, "中": 2, "重": 3}
    yes_no_map = {"否": 0, "是": 1}

    df["體積分數"] = df["體積等級"].map(volume_map)
    df["重量分數"] = df["重量等級"].map(weight_map)
    df["液體標記"] = df["是否液體"].map(yes_no_map)
    df["易碎標記"] = df["是否易碎"].map(yes_no_map)
    df["季節性標記"] = df["是否季節性"].map(yes_no_map)

    return df


def build_svm_model() -> tuple[Pipeline, list[str], list[str]]:
    """建立 SVM 分類模型。"""
    numeric_features = [
        "每日出貨箱數",
        "每日揀貨次數",
        "體積分數",
        "重量分數",
        "液體標記",
        "易碎標記",
        "季節性標記",
    ]

    categorical_features = [
        "商品類別",
        "體積等級",
        "重量等級",
        "是否液體",
        "是否易碎",
        "是否季節性",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="未知")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    random_state=42,
                ),
            ),
        ]
    )

    feature_columns = numeric_features + categorical_features
    return model, feature_columns, numeric_features


def split_train_predict(
    df: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """切分已知答案的訓練資料與待預測新品資料。"""
    train_df = df[df["資料用途"].astype(str).str.strip().eq("Train")].copy()
    predict_df = df[df["資料用途"].astype(str).str.strip().eq("Predict")].copy()

    train_df = train_df.dropna(subset=[TARGET_COLUMN])
    train_df = train_df[train_df[TARGET_COLUMN].astype(str).str.strip().ne("")]

    if train_df.empty:
        raise ValueError("沒有可訓練資料，請確認 Train 列有「建議儲位策略」。")

    if predict_df.empty:
        raise ValueError("沒有待預測資料，請確認 Predict 列存在。")

    x_train_all = train_df[feature_columns]
    y_train_all = train_df[TARGET_COLUMN].astype(str)
    x_predict = predict_df[feature_columns]

    return train_df, predict_df, x_train_all, y_train_all, x_predict


def train_and_evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict[str, object]]:
    """訓練模型並輸出簡單評估結果。"""
    class_counts = y.value_counts()
    stratify_y = y if class_counts.min() >= 2 and len(class_counts) > 1 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify_y,
    )

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("\n=== SVM 模型評估 ===")
    print(f"測試集 Accuracy：{accuracy_score(y_test, y_pred):.2%}")
    print("\n分類報告：")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("混淆矩陣：")
    display(cm_df)

    eval_result = {
        "x_test": x_test,
        "y_test": y_test,
        "y_score": model.predict_proba(x_test),
        "classes": model.named_steps["svm"].classes_,
    }

    model.fit(x, y)
    return model, eval_result


def build_strategy_cluster_mapping(train_df: pd.DataFrame) -> dict[str, str]:
    """用訓練資料建立「儲位策略 -> KMeans 群組」對照。"""
    if "KMeans群組名稱" not in train_df.columns:
        return {}

    mapping_df = train_df.dropna(subset=[TARGET_COLUMN, "KMeans群組名稱"])
    if mapping_df.empty:
        return {}

    return (
        mapping_df.groupby(TARGET_COLUMN)["KMeans群組名稱"]
        .agg(lambda values: values.value_counts().index[0])
        .to_dict()
    )


def predict_new_skus(
    model: Pipeline,
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    x_predict: pd.DataFrame,
) -> pd.DataFrame:
    """預測新品儲位類別，並加入預測信心。"""
    predictions = model.predict(x_predict)
    probabilities = model.predict_proba(x_predict)
    confidence = probabilities.max(axis=1)

    result = predict_df.copy()
    result["SVM預測儲位策略"] = predictions
    strategy_cluster_mapping = build_strategy_cluster_mapping(train_df)
    result["對應KMeans群組"] = result["SVM預測儲位策略"].map(strategy_cluster_mapping).fillna("未對應")
    result["預測信心"] = confidence
    result["預測信心百分比"] = (confidence * 100).round(1).astype(str) + "%"
    result["模型說明"] = result["預測信心"].apply(
        lambda x: "高信心" if x >= 0.7 else ("中等信心" if x >= 0.5 else "低信心，建議人工複核")
    )

    output_columns = [
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
        "SVM預測儲位策略",
        "對應KMeans群組",
        "預測信心百分比",
        "模型說明",
        "備註",
    ]
    existing_columns = [col for col in output_columns if col in result.columns]
    return result[existing_columns]


def plot_prediction_summary(result_df: pd.DataFrame) -> None:
    """繪製新品預測結果統計圖。"""
    chinese_font = setup_chinese_font()

    count = result_df["SVM預測儲位策略"].value_counts()

    plt.figure(figsize=(9, 5))
    count.plot(kind="bar", color="#0B5CAD")
    plt.title("SVM 新品儲位類別預測統計", fontsize=14, fontproperties=chinese_font)
    plt.xlabel("預測儲位策略", fontproperties=chinese_font)
    plt.ylabel("新品數量", fontproperties=chinese_font)
    plt.xticks(rotation=25, ha="right")
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(chinese_font)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    show_or_save_plot("SVM_新品儲位類別預測統計.png")


def show_or_save_plot(file_name: str) -> None:
    """Colab 顯示圖表；本地端存成 PNG。"""
    if IN_COLAB:
        plt.show()
    else:
        chart_file = BASE_DIR / file_name
        plt.savefig(chart_file, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"已輸出圖表：{chart_file}")


def to_dense_array(data) -> np.ndarray:
    """將 sklearn 轉換後的稀疏矩陣或陣列轉為 numpy array。"""
    if hasattr(data, "toarray"):
        return data.toarray()
    return np.asarray(data)


def plot_roc_curve(eval_result: dict[str, object]) -> None:
    """繪製多類別 One-vs-Rest ROC Curve。"""
    chinese_font = setup_chinese_font()
    y_test = pd.Series(eval_result["y_test"])
    y_score = np.asarray(eval_result["y_score"])
    classes = np.asarray(eval_result["classes"])

    if len(classes) < 2:
        print("ROC Curve 需要至少 2 個類別，略過繪圖。")
        return

    y_test_bin = label_binarize(y_test, classes=classes)
    if y_test_bin.shape[1] == 1:
        y_test_bin = np.column_stack([1 - y_test_bin[:, 0], y_test_bin[:, 0]])

    plt.figure(figsize=(8.5, 6))
    plotted = False
    for class_index, class_name in enumerate(classes):
        if class_index >= y_test_bin.shape[1] or class_index >= y_score.shape[1]:
            continue

        positive_count = int(y_test_bin[:, class_index].sum())
        negative_count = len(y_test_bin) - positive_count
        if positive_count == 0 or negative_count == 0:
            continue

        fpr, tpr, _ = roc_curve(y_test_bin[:, class_index], y_score[:, class_index])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{class_name} AUC={auc_score:.2f}")
        plotted = True

    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    plt.title("SVM 多類別 ROC Curve", fontsize=14, fontproperties=chinese_font)
    plt.xlabel("False Positive Rate", fontproperties=chinese_font)
    plt.ylabel("True Positive Rate", fontproperties=chinese_font)
    plt.grid(linestyle="--", alpha=0.35)
    if plotted:
        legend = plt.legend(prop=chinese_font, fontsize=9)
        for text in legend.get_texts():
            text.set_fontproperties(chinese_font)
    else:
        plt.text(
            0.5,
            0.5,
            "測試集中類別樣本不足，無法繪製 ROC",
            ha="center",
            va="center",
            fontproperties=chinese_font,
        )
    plt.tight_layout()
    show_or_save_plot("SVM_ROC_Curve.png")


def build_pca_projection(
    model: Pipeline,
    x_train_all: pd.DataFrame,
    y_train_all: pd.Series,
    x_predict: pd.DataFrame,
    result_df: pd.DataFrame,
) -> tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """使用模型前處理後的特徵建立 PCA 2D 投影。"""
    preprocessor = model.named_steps["preprocess"]
    train_features = to_dense_array(preprocessor.transform(x_train_all))
    predict_features = to_dense_array(preprocessor.transform(x_predict))

    pca = PCA(n_components=2, random_state=42)
    train_xy = pca.fit_transform(train_features)
    predict_xy = pca.transform(predict_features)

    train_pca_df = pd.DataFrame(
        {
            "PCA1": train_xy[:, 0],
            "PCA2": train_xy[:, 1],
            "儲位策略": y_train_all.to_numpy(),
            "資料用途": "Train",
        }
    )
    predict_pca_df = pd.DataFrame(
        {
            "PCA1": predict_xy[:, 0],
            "PCA2": predict_xy[:, 1],
            "儲位策略": result_df["SVM預測儲位策略"].to_numpy(),
            "資料用途": "Predict",
        }
    )
    return pca, train_pca_df, predict_pca_df


def plot_pca_scatter(train_pca_df: pd.DataFrame, predict_pca_df: pd.DataFrame) -> None:
    """繪製 PCA 散佈圖。"""
    chinese_font = setup_chinese_font()
    all_labels = sorted(pd.concat([train_pca_df["儲位策略"], predict_pca_df["儲位策略"]]).unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_labels), 1)))
    color_map = dict(zip(all_labels, colors))

    plt.figure(figsize=(8.5, 6))
    for label in all_labels:
        train_subset = train_pca_df[train_pca_df["儲位策略"].eq(label)]
        predict_subset = predict_pca_df[predict_pca_df["儲位策略"].eq(label)]

        if not train_subset.empty:
            plt.scatter(
                train_subset["PCA1"],
                train_subset["PCA2"],
                s=58,
                alpha=0.8,
                color=color_map[label],
                label=f"{label} Train",
            )
        if not predict_subset.empty:
            plt.scatter(
                predict_subset["PCA1"],
                predict_subset["PCA2"],
                s=95,
                marker="X",
                edgecolor="#222222",
                linewidth=0.8,
                color=color_map[label],
                label=f"{label} Predict",
            )

    plt.title("SVM PCA 散佈圖：Train 與 Predict 分布", fontsize=14, fontproperties=chinese_font)
    plt.xlabel("PCA 1", fontproperties=chinese_font)
    plt.ylabel("PCA 2", fontproperties=chinese_font)
    plt.grid(linestyle="--", alpha=0.35)
    legend = plt.legend(prop=chinese_font, fontsize=8, loc="best")
    for text in legend.get_texts():
        text.set_fontproperties(chinese_font)
    plt.tight_layout()
    show_or_save_plot("SVM_PCA_散佈圖.png")


def plot_decision_boundary(train_pca_df: pd.DataFrame) -> None:
    """使用 PCA 二維座標訓練展示用 SVM，繪製決策邊界。"""
    chinese_font = setup_chinese_font()
    x_pca = train_pca_df[["PCA1", "PCA2"]].to_numpy()
    y = train_pca_df["儲位策略"].to_numpy()
    classes = np.unique(y)

    if len(classes) < 2:
        print("Decision Boundary 需要至少 2 個類別，略過繪圖。")
        return

    demo_svm = SVC(kernel="rbf", C=1.0, gamma="scale")
    demo_svm.fit(x_pca, y)

    x_min, x_max = x_pca[:, 0].min() - 0.8, x_pca[:, 0].max() + 0.8
    y_min, y_max = x_pca[:, 1].min() - 0.8, x_pca[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 350), np.linspace(y_min, y_max, 350))
    grid_predictions = demo_svm.predict(np.c_[xx.ravel(), yy.ravel()])
    class_to_number = {label: index for index, label in enumerate(demo_svm.classes_)}
    zz = np.array([class_to_number[label] for label in grid_predictions]).reshape(xx.shape)

    plt.figure(figsize=(8.5, 6))
    plt.contourf(xx, yy, zz, alpha=0.16, levels=np.arange(len(classes) + 1) - 0.5, cmap="tab10")

    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    for label, color in zip(classes, colors):
        subset = train_pca_df[train_pca_df["儲位策略"].eq(label)]
        plt.scatter(
            subset["PCA1"],
            subset["PCA2"],
            s=62,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=label,
        )

    support_vectors = demo_svm.support_vectors_
    plt.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=145,
        facecolors="none",
        edgecolors="#111111",
        linewidth=1.4,
        label="支援向量",
    )

    plt.title("SVM 決策邊界圖（PCA 二維展示）", fontsize=14, fontproperties=chinese_font)
    plt.xlabel("PCA 1", fontproperties=chinese_font)
    plt.ylabel("PCA 2", fontproperties=chinese_font)
    plt.grid(linestyle="--", alpha=0.25)
    legend = plt.legend(prop=chinese_font, fontsize=8, loc="best")
    for text in legend.get_texts():
        text.set_fontproperties(chinese_font)
    plt.tight_layout()
    show_or_save_plot("SVM_Decision_Boundary.png")


def export_result(train_df: pd.DataFrame, result_df: pd.DataFrame, output_file: str) -> None:
    """匯出 SVM 預測結果 Excel。"""
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="SVM新品預測結果", index=False)
        train_df.to_excel(writer, sheet_name="SVM訓練資料", index=False)

    print(f"\n已輸出 SVM 預測結果：{output_file}")

    if IN_COLAB:
        files.download(output_file)


def setup_chinese_font() -> font_manager.FontProperties:
    """設定中文字型，避免 Colab 圖表出現 Glyph missing 與方塊亂碼。"""
    preferred_fonts = [
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Microsoft JhengHei",
        "SimHei",
        "Arial Unicode MS",
    ]
    candidate_font_files = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msjh.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]

    for font_file in candidate_font_files:
        font_path = Path(font_file)
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_prop = font_manager.FontProperties(fname=str(font_path))
            plt.rcParams["font.sans-serif"] = [font_prop.get_name(), "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            warnings.filterwarnings("ignore", message="Glyph .* missing from font")
            return font_prop

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next((font for font in preferred_fonts if font in available_fonts), None)

    if selected_font is None and IN_COLAB:
        print("Colab 尚未找到中文字型，正在安裝 fonts-noto-cjk...")
        subprocess.run(["apt-get", "-qq", "update"], check=False)
        subprocess.run(["apt-get", "-qq", "install", "-y", "fonts-noto-cjk"], check=False)
        font_manager.fontManager = font_manager._load_fontmanager(try_read_cache=False)
        for font_file in candidate_font_files:
            font_path = Path(font_file)
            if font_path.exists():
                font_manager.fontManager.addfont(str(font_path))
                font_prop = font_manager.FontProperties(fname=str(font_path))
                plt.rcParams["font.sans-serif"] = [font_prop.get_name(), "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                warnings.filterwarnings("ignore", message="Glyph .* missing from font")
                return font_prop
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        selected_font = next((font for font in preferred_fonts if font in available_fonts), None)

    if selected_font:
        plt.rcParams["font.sans-serif"] = [selected_font, "DejaVu Sans"]
        font_prop = font_manager.FontProperties(family=selected_font)
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        print("提醒：目前環境找不到中文字型，圖表中文可能會顯示為方塊。")
        font_prop = font_manager.FontProperties(family="DejaVu Sans")

    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    return font_prop


def main() -> None:
    parser = argparse.ArgumentParser(description="Unit 4 SVM 儲位類別預測範例")
    parser.add_argument("--input", default=None, help="輸入 Excel 檔案路徑")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="輸出 Excel 檔案路徑")
    args, _ = parser.parse_known_args()

    file_name = get_input_file(args.input)
    if not IN_COLAB and not Path(file_name).exists():
        raise FileNotFoundError(f"找不到檔案：{file_name}")

    print("讀取檔案：", file_name)
    df = read_inventory_data(file_name)
    validate_columns(df)
    df = clean_data(df)

    print("\n=== SKU 資料預覽 ===")
    display(df.head())

    model, feature_columns, _ = build_svm_model()
    train_df, predict_df, x_train_all, y_train_all, x_predict = split_train_predict(df, feature_columns)

    print("\n=== 資料筆數 ===")
    print(f"Train 已知儲位策略資料：{len(train_df)} 筆")
    print(f"Predict 新品待預測資料：{len(predict_df)} 筆")

    print("\n=== SVM 使用特徵欄位 ===")
    for col in feature_columns:
        print("-", col)

    model, eval_result = train_and_evaluate(model, x_train_all, y_train_all)
    result_df = predict_new_skus(model, train_df, predict_df, x_predict)

    print("\n=== 新品 SVM 儲位類別預測結果 ===")
    display(result_df)

    plot_prediction_summary(result_df)
    plot_roc_curve(eval_result)
    _, train_pca_df, predict_pca_df = build_pca_projection(model, x_train_all, y_train_all, x_predict, result_df)
    plot_pca_scatter(train_pca_df, predict_pca_df)
    plot_decision_boundary(train_pca_df)
    export_result(train_df, result_df, args.output)


if __name__ == "__main__":
    main()
