import matplotlib.pyplot as plt


# =========================
# 基本參數
# =========================

g = 9.81          # 重力加速度 (m/s^2)

m = 1.0           # 質量 (kg)
rho = 1.225       # 空氣密度 (kg/m^3)
Cd = 0.47         # 阻力係數
A = 0.10          # 截面積 (m^2)

y0 = 100.0        # 初始高度 (m)
v0 = 0.0          # 初速度 (m/s)

dt = 0.01         # 時間間隔 (s)


# =========================
# 初始狀態
# =========================

t = 0.0
y = y0
v = v0


time_list = [t]
height_list = [y]
velocity_list = [v]
acceleration_list = [-g]
drag_force_list = [0.0]


# =========================
# Euler Method
# =========================

while y > 0:

    # 空氣阻力
    F_drag = -0.5 * rho * Cd * A * v * abs(v)

    # 總加速度
    a = -g + F_drag / m

    # 更新位置
    y = y + v * dt

    # 更新速度
    v = v + a * dt

    # 更新時間
    t = t + dt

    # 儲存資料
    time_list.append(t)
    height_list.append(y)
    velocity_list.append(v)
    acceleration_list.append(a)
    drag_force_list.append(F_drag)


# =========================
# 理論終端速度
# =========================

v_terminal = -(
    (2 * m * g) / (rho * Cd * A)
) ** 0.5


# =========================
# 顯示結果
# =========================

print("===== 平方空氣阻力模擬 =====")

print(f"質量：{m:.2f} kg")
print(f"空氣密度：{rho:.3f} kg/m^3")
print(f"阻力係數 Cd：{Cd:.2f}")
print(f"截面積：{A:.3f} m^2")

print(f"落地時間：約 {t:.3f} s")
print(f"落地速度：約 {v:.3f} m/s")

print(
    f"理論終端速度：{v_terminal:.3f} m/s"
)


# =========================
# 高度圖
# =========================

plt.figure()

plt.plot(
    time_list,
    height_list
)

plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Quadratic Drag: Height vs Time")

plt.grid()


# =========================
# 速度圖
# =========================

plt.figure()

plt.plot(
    time_list,
    velocity_list,
    label="Velocity"
)

plt.axhline(
    y=v_terminal,
    linestyle="--",
    label="Terminal Velocity"
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Quadratic Drag: Velocity vs Time")

plt.legend()
plt.grid()


# =========================
# 加速度圖
# =========================

plt.figure()

plt.plot(
    time_list,
    acceleration_list
)

plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Quadratic Drag: Acceleration vs Time")

plt.grid()


# =========================
# 空氣阻力圖
# =========================

plt.figure()

plt.plot(
    time_list,
    drag_force_list
)

plt.xlabel("Time (s)")
plt.ylabel("Drag Force (N)")
plt.title("Quadratic Drag Force vs Time")

plt.grid()


plt.show()