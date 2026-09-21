import matplotlib.pyplot as plt


# =========================
# 初始條件
# =========================

g = 9.81        # 重力加速度 (m/s^2)
m = 1.0         # 質量 (kg)
b = 0.5         # 線性阻力係數 (kg/s)

y0 = 100.0      # 初始高度 (m)
v0 = 0.0        # 初速度 (m/s)

dt = 0.01       # 時間間隔 (s)


# =========================
# 初始狀態
# =========================

t = 0.0
y = y0
v = v0


# 儲存資料
time_list = [t]
height_list = [y]
velocity_list = [v]
acceleration_list = [-g]


# =========================
# Euler Method
# =========================

while y > 0:

    # ---------------------
    # 計算加速度
    # ---------------------

    a = -g - (b / m) * v


    # ---------------------
    # 更新位置
    # ---------------------

    y = y + v * dt


    # ---------------------
    # 更新速度
    # ---------------------

    v = v + a * dt


    # ---------------------
    # 更新時間
    # ---------------------

    t = t + dt


    # ---------------------
    # 儲存資料
    # ---------------------

    time_list.append(t)
    height_list.append(y)
    velocity_list.append(v)
    acceleration_list.append(a)


# =========================
# 終端速度
# =========================

v_terminal = -m * g / b


# =========================
# 顯示結果
# =========================

print("===== 線性空氣阻力模擬 =====")

print(f"質量：{m:.2f} kg")
print(f"阻力係數 b：{b:.2f} kg/s")

print(f"落地時間：約 {t:.3f} s")
print(f"落地速度：約 {v:.3f} m/s")

print(f"理論終端速度：{v_terminal:.3f} m/s")


# =========================
# 高度 - 時間圖
# =========================

plt.figure()

plt.plot(
    time_list,
    height_list
)

plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Linear Drag: Height vs Time")

plt.grid()


# =========================
# 速度 - 時間圖
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
    label="Terminal velocity"
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Linear Drag: Velocity vs Time")

plt.legend()
plt.grid()


# =========================
# 加速度 - 時間圖
# =========================

plt.figure()

plt.plot(
    time_list,
    acceleration_list
)

plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.title("Linear Drag: Acceleration vs Time")

plt.grid()


# =========================
# 顯示所有圖
# =========================

plt.show()