# ============================================================
# 時間與成本函數設計建立
# 作者：國立雲林科技大學電機系 林家仁
# 工廠生產時間與成本函數分析
# Pareto Front 多目標最佳化之前導練習
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import interact, IntSlider

# 設定圖表樣式
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120


# ============================================================
# 1. 設定工廠生產參數
# ============================================================

最大生產數量 = 10

# 生產時間參數
單件生產時間 = 20       # 每件產品需要的生產時間，單位：分鐘
固定準備時間 = 30       # 設備啟動與生產準備時間，單位：分鐘

# 生產成本參數
單件變動成本 = 50       # 每生產一件產品增加的成本，單位：元
固定生產成本 = 100      # 不隨生產數量改變的固定成本，單位：元


# ============================================================
# 2. 建立總生產時間函數與總生產成本函數
# ============================================================

def 計算總生產時間(生產數量):
    """
    總生產時間函數：

    T(q) = 單件生產時間 × 生產數量 + 固定準備時間
    """
    return 單件生產時間 * 生產數量 + 固定準備時間


def 計算總生產成本(生產數量):
    """
    總生產成本函數：

    C(q) = 單件變動成本 × 生產數量 + 固定生產成本
    """
    return 單件變動成本 * 生產數量 + 固定生產成本


# ============================================================
# 3. 顯示生產參數與函數
# ============================================================

print("=" * 65)
print("工廠生產時間與成本函數分析")
print("=" * 65)

print("\n【生產參數】")
print(f"每件產品生產時間：{單件生產時間} 分鐘")
print(f"設備固定準備時間：{固定準備時間} 分鐘")
print(f"每件產品變動成本：{單件變動成本} 元")
print(f"固定生產成本：{固定生產成本} 元")

print("\n【總生產時間函數】")
print(f"T(q) = {單件生產時間}q + {固定準備時間}")

print("\n【總生產成本函數】")
print(f"C(q) = {單件變動成本}q + {固定生產成本}")


# ============================================================
# 4. 建立不同生產數量的時間與成本資料
# ============================================================

生產數量資料 = np.arange(1, 最大生產數量 + 1)

總生產時間資料 = 計算總生產時間(生產數量資料)
總生產成本資料 = 計算總生產成本(生產數量資料)

生產資料表 = pd.DataFrame({
    "生產數量（件）": 生產數量資料,
    "總生產時間（分鐘）": 總生產時間資料,
    "總生產成本（元）": 總生產成本資料
})

print("\n【不同生產數量的時間與成本】")
display(生產資料表)


# ============================================================
# 5. 顯示前 3 件產品的完整計算過程
# ============================================================

print("\n" + "=" * 65)
print("生產時間與成本計算範例")
print("=" * 65)

for 生產數量 in range(1, 4):

    總時間 = 計算總生產時間(生產數量)
    總成本 = 計算總生產成本(生產數量)

    print(f"\n生產數量：{生產數量} 件")

    print(
        f"總生產時間：T({生產數量}) "
        f"= {單件生產時間} × {生產數量} + {固定準備時間} "
        f"= {總時間} 分鐘"
    )

    print(
        f"總生產成本：C({生產數量}) "
        f"= {單件變動成本} × {生產數量} + {固定生產成本} "
        f"= {總成本} 元"
    )


# ============================================================
# 6. 繪製生產數量與總生產時間圖
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    生產數量資料,
    總生產時間資料,
    color="#1565C0",
    marker="o",
    markersize=8,
    linewidth=2.5,
    label="Total Production Time"
)

# 在每一個資料點上顯示總生產時間
for 生產數量, 總時間 in zip(
    生產數量資料,
    總生產時間資料
):
    plt.annotate(
        f"{總時間}",
        (生產數量, 總時間),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

plt.title(
    "Production Quantity vs. Total Production Time",
    fontsize=15,
    fontweight="bold"
)
plt.xlabel("Production Quantity (units)", fontsize=11)
plt.ylabel("Total Production Time (minutes)", fontsize=11)
plt.xticks(生產數量資料)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 7. 繪製生產數量與總生產成本圖
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    生產數量資料,
    總生產成本資料,
    color="#EF6C00",
    marker="o",
    markersize=8,
    linewidth=2.5,
    label="Total Production Cost"
)

# 在每一個資料點上顯示總生產成本
for 生產數量, 總成本 in zip(
    生產數量資料,
    總生產成本資料
):
    plt.annotate(
        f"{總成本}",
        (生產數量, 總成本),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )

plt.title(
    "Production Quantity vs. Total Production Cost",
    fontsize=15,
    fontweight="bold"
)
plt.xlabel("Production Quantity (units)", fontsize=11)
plt.ylabel("Total Production Cost (NTD)", fontsize=11)
plt.xticks(生產數量資料)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 8. 將時間與成本圖並排顯示
# ============================================================

圖表, 座標軸 = plt.subplots(1, 2, figsize=(15, 5.5))

# 左圖：生產數量與總生產時間
座標軸[0].plot(
    生產數量資料,
    總生產時間資料,
    color="#1565C0",
    marker="o",
    markersize=7,
    linewidth=2.5,
    label="Total Production Time"
)

座標軸[0].set_title(
    "Production Quantity vs. Time",
    fontsize=13,
    fontweight="bold"
)
座標軸[0].set_xlabel("Production Quantity (units)")
座標軸[0].set_ylabel("Total Production Time (minutes)")
座標軸[0].set_xticks(生產數量資料)
座標軸[0].grid(True, linestyle="--", alpha=0.4)
座標軸[0].legend()

# 右圖：生產數量與總生產成本
座標軸[1].plot(
    生產數量資料,
    總生產成本資料,
    color="#EF6C00",
    marker="o",
    markersize=7,
    linewidth=2.5,
    label="Total Production Cost"
)

座標軸[1].set_title(
    "Production Quantity vs. Cost",
    fontsize=13,
    fontweight="bold"
)
座標軸[1].set_xlabel("Production Quantity (units)")
座標軸[1].set_ylabel("Total Production Cost (NTD)")
座標軸[1].set_xticks(生產數量資料)
座標軸[1].grid(True, linestyle="--", alpha=0.4)
座標軸[1].legend()

plt.tight_layout()
plt.show()


# ============================================================
# 9. 將時間與成本標準化後放在同一張圖中
#
# 因為時間與成本的單位不同，不能直接比較原始數值，
# 因此先將兩者轉換成 0～1 之間的標準化數值。
# ============================================================

標準化時間 = (
    (總生產時間資料 - 總生產時間資料.min()) /
    (總生產時間資料.max() - 總生產時間資料.min())
)

標準化成本 = (
    (總生產成本資料 - 總生產成本資料.min()) /
    (總生產成本資料.max() - 總生產成本資料.min())
)

plt.figure(figsize=(10, 6))

plt.plot(
    生產數量資料,
    標準化時間,
    color="#1565C0",
    marker="o",
    markersize=8,
    linewidth=2.5,
    label="Normalized Production Time"
)

plt.plot(
    生產數量資料,
    標準化成本,
    color="#EF6C00",
    marker="s",
    markersize=8,
    linewidth=2.5,
    label="Normalized Production Cost"
)

plt.title(
    "Normalized Production Time and Cost",
    fontsize=15,
    fontweight="bold"
)
plt.xlabel("Production Quantity (units)", fontsize=11)
plt.ylabel("Normalized Value", fontsize=11)
plt.xticks(生產數量資料)
plt.ylim(-0.05, 1.10)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 10. 繪製總生產時間與總生產成本關係圖
# 此圖用來讓學生先理解：
# 每一種生產數量，都會對應一組時間與成本結果。
# 這張圖是 Pareto Front 的前導概念，
# 但目前還不能稱為真正的 Pareto Front。
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    總生產時間資料,
    總生產成本資料,
    color="#6A1B9A",
    marker="o",
    markersize=8,
    linewidth=2.5,
    label="Production Plan"
)

# 標示每個資料點所對應的生產數量
for 生產數量, 總時間, 總成本 in zip(
    生產數量資料,
    總生產時間資料,
    總生產成本資料
):
    plt.annotate(
        f"Q={生產數量}",
        (總時間, 總成本),
        xytext=(6, 7),
        textcoords="offset points",
        fontsize=9
    )

plt.title(
    "Time-Cost Relationship in Factory Production",
    fontsize=15,
    fontweight="bold"
)
plt.xlabel("Total Production Time (minutes)", fontsize=11)
plt.ylabel("Total Production Cost (NTD)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 11. 建立互動式生產時間與成本計算器
# ============================================================

def 互動式生產計算器(生產數量):

    總時間 = 計算總生產時間(生產數量)
    總成本 = 計算總生產成本(生產數量)

    print("=" * 60)
    print("工廠生產時間與成本計算結果")
    print("=" * 60)

    print(f"\n生產數量：{生產數量} 件")

    print("\n【總生產時間】")
    print(
        f"T({生產數量}) "
        f"= {單件生產時間} × {生產數量} + {固定準備時間} "
        f"= {總時間} 分鐘"
    )

    print("\n【總生產成本】")
    print(
        f"C({生產數量}) "
        f"= {單件變動成本} × {生產數量} + {固定生產成本} "
        f"= {總成本} 元"
    )

    # 建立英文圖表
    圖表, 座標軸 = plt.subplots(1, 2, figsize=(11, 4.5))

    # 總生產時間長條圖
    座標軸[0].bar(
        ["Total Time"],
        [總時間],
        color="#1565C0",
        width=0.5
    )

    座標軸[0].set_title(
        "Total Production Time",
        fontweight="bold"
    )
    座標軸[0].set_ylabel("Minutes")
    座標軸[0].set_ylim(
        0,
        計算總生產時間(20) * 1.15
    )
    座標軸[0].text(
        0,
        總時間 + 8,
        f"{總時間} min",
        ha="center",
        fontweight="bold"
    )
    座標軸[0].grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    # 總生產成本長條圖
    座標軸[1].bar(
        ["Total Cost"],
        [總成本],
        color="#EF6C00",
        width=0.5
    )

    座標軸[1].set_title(
        "Total Production Cost",
        fontweight="bold"
    )
    座標軸[1].set_ylabel("NTD")
    座標軸[1].set_ylim(
        0,
        計算總生產成本(20) * 1.15
    )
    座標軸[1].text(
        0,
        總成本 + 20,
        f"{總成本} NTD",
        ha="center",
        fontweight="bold"
    )
    座標軸[1].grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


interact(
    互動式生產計算器,
    生產數量=IntSlider(
        value=3,
        min=1,
        max=20,
        step=1,
        description="生產數量：",
        continuous_update=False
    )
)


# ============================================================
# 12. 計算新的生產計畫
# ============================================================

新生產數量 = 15

預估總時間 = 計算總生產時間(新生產數量)
預估總成本 = 計算總生產成本(新生產數量)

print("\n" + "=" * 65)
print("新生產計畫試算結果")
print("=" * 65)

print(f"生產數量：{新生產數量} 件")
print(f"總生產時間：{預估總時間} 分鐘")
print(f"總生產成本：{預估總成本} 元")

print("\n【結果說明】")
print(
    f"當工廠生產 {新生產數量} 件產品時，"
    f"預估總生產時間為 {預估總時間} 分鐘，"
    f"預估總生產成本為 {預估總成本} 元。"
)

