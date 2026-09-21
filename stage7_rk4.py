import numpy as np
import matplotlib.pyplot as plt


# =========================
# 基本參數
# =========================

g = 9.81

m = 1.0
rho = 1.225
Cd = 0.47
A = 0.10

y0 = 100.0
v0 = 0.0

dt = 0.05


# =========================
# 定義微分方程
# =========================

def derivatives(y, v):

    # 空氣阻力
    F_drag = -0.5 * rho * Cd * A * v * abs(v)

    # dy/dt
    dydt = v

    # dv/dt
    dvdt = -g + F_drag / m

    return dydt, dvdt


# =========================
# 初始狀態
# =========================

t = 0.0
y = y0
v = v0


time_list = [t]
height_list = [y]
velocity_list = [v]


# =========================
# RK4 Method
# =========================

while y > 0:

    # k1
    k1_y, k1_v = derivatives(y, v)

    # k2
    k2_y, k2_v = derivatives(
        y + 0.5 * dt * k1_y,
        v + 0.5 * dt * k1_v
    )

    # k3
    k3_y, k3_v = derivatives(
        y + 0.5 * dt * k2_y,
        v + 0.5 * dt * k2_v
    )

    # k4
    k4_y, k4_v = derivatives(
        y + dt * k3_y,
        v + dt * k3_v
    )

    # 更新位置
    y = y + (dt / 6.0) * (
        k1_y
        + 2 * k2_y
        + 2 * k3_y
        + k4_y
    )

    # 更新速度
    v = v + (dt / 6.0) * (
        k1_v
        + 2 * k2_v
        + 2 * k3_v
        + k4_v
    )

    # 更新時間
    t = t + dt

    # 儲存資料
    time_list.append(t)
    height_list.append(y)
    velocity_list.append(v)


# =========================
# 顯示結果
# =========================

print("===== RK4 平方阻力模擬 =====")

print(f"落地時間：約 {t:.3f} s")
print(f"落地速度：約 {v:.3f} m/s")


# =========================
# 畫高度圖
# =========================

plt.figure()

plt.plot(
    time_list,
    height_list
)

plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("RK4: Height vs Time")

plt.grid()


# =========================
# 畫速度圖
# =========================

plt.figure()

plt.plot(
    time_list,
    velocity_list
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("RK4: Velocity vs Time")

plt.grid()


plt.show()