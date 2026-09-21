import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# 基本參數
# -------------------------

g = 9.81       # 重力加速度 m/s^2
y0 = 100.0     # 初始高度 m
v0 = 0.0       # 初速度 m/s


# -------------------------
# 計算落地時間
# -------------------------

t_hit = (v0 + np.sqrt(v0**2 + 2 * g * y0)) / g


# -------------------------
# 建立時間資料
# -------------------------

t = np.linspace(0, t_hit, 500)


# -------------------------
# 計算高度與速度
# -------------------------

y = y0 + v0 * t - 0.5 * g * t**2

v = v0 - g * t


# -------------------------
# 計算落地速度
# -------------------------

v_hit = v0 - g * t_hit


# -------------------------
# 顯示結果
# -------------------------

print("===== 模擬結果 =====")
print(f"初始高度：{y0:.2f} m")
print(f"初速度：{v0:.2f} m/s")
print(f"落地時間：{t_hit:.3f} s")
print(f"落地速度：{v_hit:.3f} m/s")


# -------------------------
# 高度 - 時間圖
# -------------------------

plt.figure()

plt.plot(t, y)

plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Height vs Time")

plt.grid()

# -------------------------
# 速度 - 時間圖
# -------------------------

plt.figure()

plt.plot(t, v)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity vs Time")

plt.grid()

plt.show()