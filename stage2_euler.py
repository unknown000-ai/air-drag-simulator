import numpy as np
import matplotlib.pyplot as plt


# =========================
# 初始條件
# =========================

g = 9.81
y0 = 100.0
v0 = 0.0

dt = 0.001


# =========================
# 初始狀態
# =========================

t = 0.0
y = y0
v = v0


# 用 list 儲存每一步資料
time_list = [t]
height_list = [y]
velocity_list = [v]


# =========================
# Euler Method
# =========================

while y > 0:

    # 更新位置
    y = y + v * dt

    # 更新速度
    v = v - g * dt

    # 更新時間
    t = t + dt

    # 儲存資料
    time_list.append(t)
    height_list.append(y)
    velocity_list.append(v)


# =========================
# 顯示 Euler 結果
# =========================

print("===== Euler 模擬結果 =====")
print(f"落地時間：約 {t:.3f} s")
print(f"落地速度：約 {v:.3f} m/s")


# =========================
# 建立解析解
# =========================

time_array = np.array(time_list)

y_exact = y0 + v0 * time_array - 0.5 * g * time_array**2

v_exact = v0 - g * time_array


# =========================
# 高度圖：Euler vs Exact
# =========================

plt.figure()

plt.plot(
    time_list,
    height_list,
    label="Euler"
)

plt.plot(
    time_array,
    y_exact,
    "--",
    label="Exact"
)

plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Euler vs Exact: Height")

plt.legend()
plt.grid()


# =========================
# 速度圖：Euler vs Exact
# =========================

plt.figure()

plt.plot(
    time_list,
    velocity_list,
    label="Euler"
)

plt.plot(
    time_array,
    v_exact,
    "--",
    label="Exact"
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Euler vs Exact: Velocity")

plt.legend()
plt.grid()


# 一次顯示兩張圖
plt.show()