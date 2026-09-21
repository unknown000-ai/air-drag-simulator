import numpy as np
import matplotlib.pyplot as plt


# =========================
# 基本參數
# =========================

g = 9.81

m = 1.0
b = 0.5

y0 = 100.0
v0 = 0.0

dt = 0.01


# =========================
# 真空解析解
# =========================

t_hit_vacuum = (
    v0 + np.sqrt(v0**2 + 2 * g * y0)
) / g

t_vacuum = np.linspace(
    0,
    t_hit_vacuum,
    500
)

y_vacuum = (
    y0
    + v0 * t_vacuum
    - 0.5 * g * t_vacuum**2
)

v_vacuum = (
    v0
    - g * t_vacuum
)


# =========================
# 線性阻力 Euler
# =========================

t = 0.0
y = y0
v = v0

time_drag = [t]
height_drag = [y]
velocity_drag = [v]


while y > 0:

    a = -g - (b / m) * v

    y = y + v * dt

    v = v + a * dt

    t = t + dt

    time_drag.append(t)
    height_drag.append(y)
    velocity_drag.append(v)


# =========================
# 終端速度
# =========================

v_terminal = -m * g / b


# =========================
# 顯示數值結果
# =========================

print("===== 真空 =====")
print(
    f"落地時間：{t_hit_vacuum:.3f} s"
)

print(
    f"落地速度：{v_vacuum[-1]:.3f} m/s"
)


print("\n===== 線性空氣阻力 =====")
print(
    f"落地時間：約 {t:.3f} s"
)

print(
    f"落地速度：約 {v:.3f} m/s"
)

print(
    f"終端速度：{v_terminal:.3f} m/s"
)


# =========================
# 高度比較
# =========================

plt.figure()

plt.plot(
    t_vacuum,
    y_vacuum,
    label="Vacuum"
)

plt.plot(
    time_drag,
    height_drag,
    label="Linear Drag"
)

plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Height: Vacuum vs Linear Drag")

plt.legend()
plt.grid()


# =========================
# 速度比較
# =========================

plt.figure()

plt.plot(
    t_vacuum,
    v_vacuum,
    label="Vacuum"
)

plt.plot(
    time_drag,
    velocity_drag,
    label="Linear Drag"
)

plt.axhline(
    y=v_terminal,
    linestyle="--",
    label="Terminal Velocity"
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity: Vacuum vs Linear Drag")

plt.legend()
plt.grid()


plt.show()