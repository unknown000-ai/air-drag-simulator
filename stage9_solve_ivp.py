import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


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


# =========================
# 微分方程
# =========================

def equations(t, state):

    y = state[0]
    v = state[1]

    F_drag = -0.5 * rho * Cd * A * v * abs(v)

    dydt = v
    dvdt = -g + F_drag / m

    return [dydt, dvdt]


# =========================
# 落地事件
# =========================

def hit_ground(t, state):

    y = state[0]

    return y


# 只偵測由正高度穿越到負高度
hit_ground.direction = -1

# 偵測到落地就停止
hit_ground.terminal = True


# =========================
# 初始條件
# =========================

initial_state = [
    y0,
    v0
]


# =========================
# 求解
# =========================

solution = solve_ivp(
    equations,
    [0, 30],
    initial_state,
    events=hit_ground,
    max_step=0.05,
    rtol=1e-8,
    atol=1e-10
)


# =========================
# 取出結果
# =========================

t = solution.t

y = solution.y[0]
v = solution.y[1]


# =========================
# 落地時間、速度
# =========================

if len(solution.t_events[0]) > 0:

    t_hit = solution.t_events[0][0]

    y_hit = solution.y_events[0][0][0]
    v_hit = solution.y_events[0][0][1]

else:

    t_hit = None
    y_hit = None
    v_hit = None


# =========================
# 終端速度
# =========================

v_terminal = -np.sqrt(
    2 * m * g
    / (rho * Cd * A)
)


# =========================
# 顯示結果
# =========================

print("===== solve_ivp 平方阻力 =====")

print(f"理論終端速度：{v_terminal:.3f} m/s")

if t_hit is not None:

    print(f"落地時間：{t_hit:.6f} s")
    print(f"落地速度：{v_hit:.6f} m/s")

else:

    print("模擬時間內沒有落地")


# =========================
# 高度圖
# =========================

plt.figure()

plt.plot(
    t,
    y
)

plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("solve_ivp: Height vs Time")

plt.grid()


# =========================
# 速度圖
# =========================

plt.figure()

plt.plot(
    t,
    v,
    label="Velocity"
)

plt.axhline(
    y=v_terminal,
    linestyle="--",
    label="Terminal Velocity"
)

plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("solve_ivp: Velocity vs Time")

plt.legend()
plt.grid()


plt.show()