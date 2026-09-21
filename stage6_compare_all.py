import numpy as np
import matplotlib.pyplot as plt


# =========================
# 基本參數
# =========================

g = 9.81

m = 1.0

# 線性阻力
b = 0.5

# 平方阻力
rho = 1.225
Cd = 0.47
A = 0.10

# 初始條件
y0 = 100.0
v0 = 0.0

dt = 0.01


# =========================
# 1. 真空
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
# 2. 線性空氣阻力
# =========================

t = 0.0
y = y0
v = v0

time_linear = [t]
height_linear = [y]
velocity_linear = [v]


while y > 0:

    a = -g - (b / m) * v

    y = y + v * dt
    v = v + a * dt
    t = t + dt

    time_linear.append(t)
    height_linear.append(y)
    velocity_linear.append(v)


t_hit_linear = t
v_hit_linear = v


# =========================
# 3. 平方空氣阻力
# =========================

t = 0.0
y = y0
v = v0

time_quad = [t]
height_quad = [y]
velocity_quad = [v]


while y > 0:

    F_drag = (
        -0.5
        * rho
        * Cd
        * A
        * v
        * abs(v)
    )

    a = -g + F_drag / m

    y = y + v * dt
    v = v + a * dt
    t = t + dt

    time_quad.append(t)
    height_quad.append(y)
    velocity_quad.append(v)


t_hit_quad = t
v_hit_quad = v


# =========================
# 終端速度
# =========================

v_terminal_linear = -m * g / b

v_terminal_quad = -np.sqrt(
    2 * m * g
    / (rho * Cd * A)
)


# =========================
# 顯示結果
# =========================

print("===== Vacuum =====")
print(
    f"落地時間：{t_hit_vacuum:.3f} s"
)
print(
    f"落地速度：{v_vacuum[-1]:.3f} m/s"
)


print("\n===== Linear Drag =====")
print(
    f"落地時間：約 {t_hit_linear:.3f} s"
)
print(
    f"落地速度：約 {v_hit_linear:.3f} m/s"
)
print(
    f"終端速度：{v_terminal_linear:.3f} m/s"
)


print("\n===== Quadratic Drag =====")
print(
    f"落地時間：約 {t_hit_quad:.3f} s"
)
print(
    f"落地速度：約 {v_hit_quad:.3f} m/s"
)
print(
    f"終端速度：{v_terminal_quad:.3f} m/s"
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
    time_linear,
    height_linear,
    label="Linear Drag"
)

plt.plot(
    time_quad,
    height_quad,
    label="Quadratic Drag"
)

plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Height Comparison")

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
    time_linear,
    velocity_linear,
    label="Linear Drag"
)

plt.plot(
    time_quad,
    velocity_quad,
    label="Quadratic Drag"
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity Comparison")

plt.legend()
plt.grid()


plt.show()