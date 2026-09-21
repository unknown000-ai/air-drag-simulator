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

dt = 0.5


# =========================
# 微分方程
# =========================

def derivatives(y, v):

    F_drag = -0.5 * rho * Cd * A * v * abs(v)

    dydt = v
    dvdt = -g + F_drag / m

    return dydt, dvdt


# =========================
# Euler Method
# =========================

t = 0.0
y = y0
v = v0

time_euler = [t]
height_euler = [y]
velocity_euler = [v]


while y > 0:

    dydt, dvdt = derivatives(y, v)

    y = y + dydt * dt
    v = v + dvdt * dt
    t = t + dt

    time_euler.append(t)
    height_euler.append(y)
    velocity_euler.append(v)


t_hit_euler = t
v_hit_euler = v


# =========================
# RK4 Method
# =========================

t = 0.0
y = y0
v = v0

time_rk4 = [t]
height_rk4 = [y]
velocity_rk4 = [v]


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

    y = y + (dt / 6.0) * (
        k1_y
        + 2 * k2_y
        + 2 * k3_y
        + k4_y
    )

    v = v + (dt / 6.0) * (
        k1_v
        + 2 * k2_v
        + 2 * k3_v
        + k4_v
    )

    t = t + dt

    time_rk4.append(t)
    height_rk4.append(y)
    velocity_rk4.append(v)


t_hit_rk4 = t
v_hit_rk4 = v


# =========================
# 顯示結果
# =========================

print("===== Euler =====")
print(f"落地時間：約 {t_hit_euler:.3f} s")
print(f"落地速度：約 {v_hit_euler:.3f} m/s")


print("\n===== RK4 =====")
print(f"落地時間：約 {t_hit_rk4:.3f} s")
print(f"落地速度：約 {v_hit_rk4:.3f} m/s")


# =========================
# 高度比較
# =========================

plt.figure()

plt.plot(
    time_euler,
    height_euler,
    marker="o",
    label="Euler"
)

plt.plot(
    time_rk4,
    height_rk4,
    marker="s",
    label="RK4"
)

plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title(f"Height: Euler vs RK4, dt = {dt}")

plt.legend()
plt.grid()


# =========================
# 速度比較
# =========================

plt.figure()

plt.plot(
    time_euler,
    velocity_euler,
    marker="o",
    label="Euler"
)

plt.plot(
    time_rk4,
    velocity_rk4,
    marker="s",
    label="RK4"
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title(f"Velocity: Euler vs RK4, dt = {dt}")

plt.legend()
plt.grid()


plt.show()