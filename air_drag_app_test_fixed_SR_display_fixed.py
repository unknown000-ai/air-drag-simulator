import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, brentq
from scipy.signal import savgol_filter

# 光速為 SI 制的精確定義值。此模型採用「實驗室／空氣靜止參考系」。
C_LIGHT = 299_792_458.0


def lorentz_gamma(v):
    """計算洛倫茲因子；輸入必須符合 |v| < c。"""
    v = np.asarray(v, dtype=float)
    if np.any(np.abs(v) >= C_LIGHT):
        raise ValueError("相對論模型要求初速度與估計速度皆小於光速。")
    return 1.0 / np.sqrt(1.0 - (v / C_LIGHT)**2)


def momentum_per_mass(v):
    """u = p/m = gamma*v，不使用『相對論質量』。"""
    return lorentz_gamma(v) * np.asarray(v, dtype=float)


def velocity_from_u(u):
    """由 u=gamma*v 取回實驗室座標速度；可避免積分數值超光速。"""
    u = np.asarray(u, dtype=float)
    return u / np.hypot(1.0, u / C_LIGHT)


def relative_acceleration_correction(v):
    """1 - 1/gamma^3；在 v << c 時也可避免直接相減的消去誤差。"""
    beta2 = (np.asarray(v, dtype=float) / C_LIGHT)**2
    if np.any(beta2 >= 1):
        raise ValueError("速度不可達到光速。")
    return -np.expm1(1.5 * np.log1p(-beta2))


def motion_rhs(t, state, m, g, drag_function, relativistic):
    """一維實驗室時間；SR 時 state=[y,u]，牛頓時 state=[y,v]。"""
    y, second = state
    v = float(velocity_from_u(second)) if relativistic else second
    # F_D 是空氣相對實驗室靜止時，量測／擬合的三維力（非普適高速定律）。
    return [v, -g + drag_function(v) / m]


def initial_state(y0, v0, relativistic):
    return [y0, float(momentum_per_mass(v0)) if relativistic else v0]

# python -m streamlit run app.py

# ============================================================
# 1. 網頁設定
# ============================================================

st.set_page_config(
    page_title="空氣阻力運動模擬",
    page_icon="📉",
    layout="wide"
)

st.title("空氣阻力下的落體運動、阻力律辨識與相對論動量修正")

st.write(
    "比較真空、線性阻力、平方阻力與可變冪次阻力模型；"
    "並可選擇牛頓或狹義相對論動量方程，研究低速實驗的適用範圍。"
)

st.info(
    "本程式定義向上為正方向，因此向下運動時速度為負。"
)

PLOT_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "responsive": True
}


# ============================================================
# 2. 預設物體
# ============================================================

OBJECT_PRESETS = {

    "自訂": None,

    "棒球（近似）": {
        "m_input": 0.145,
        "Cd_input": 0.47,
        "A_input": 0.0042,
        "L_input": 0.073
    },

    "羽毛（示意）": {
        "m_input": 0.003,
        "Cd_input": 1.20,
        "A_input": 0.010,
        "L_input": 0.10
    },

    "雨滴（約 2 mm，示意）": {
        "m_input": 4.2e-6,
        "Cd_input": 0.60,
        "A_input": 3.14e-6,
        "L_input": 0.002
    }
}


DEFAULTS = {

    "m_input": 1.0,
    "Cd_input": 0.47,
    "A_input": 0.10,
    "L_input": 0.10,

    "y0_input": 100.0,
    "v0_input": 0.0,

    "g_input": 9.81,

    "b_input": 0.5,

    "rho_input": 1.225,
    "mu_input": 1.81e-5,

    "vc_input": 10.0,
    "p_input": 4.0,
    "n_high_input": 1.8
}


for key, value in DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = value


if "has_run" not in st.session_state:
    st.session_state.has_run = False


if "fit_result" not in st.session_state:
    st.session_state.fit_result = None


def apply_object_preset():

    name = st.session_state.object_preset

    preset = OBJECT_PRESETS[name]

    if preset is not None:

        for key, value in preset.items():
            st.session_state[key] = value


# ============================================================
# 3. 物理說明
# ============================================================

with st.expander("📘 查看物理模型與公式"):

    st.markdown(
        r"""
### 基本運動方程

$$
\frac{dy}{dt}=v
$$

$$
m\frac{dv}{dt}=-mg+F_D
$$

---

### 真空

$$
F_D=0
$$

---

### 線性阻力

$$
F_D=-bv
$$

---

### 平方阻力

$$
F_D=
-\frac12\rho C_D A v|v|
$$

---

### 可變冪次阻力

本專題加入一個經驗模型：

$$
F_D=
-\frac12\rho C_{D0}A v|v|
\left[
1+
\left(
\frac{|v|}{v_c}
\right)^p
\right]^{
-\frac{2-n_\infty}{p}
}
$$

當

$$
|v|\ll v_c
$$

時，

$$
|F_D|\propto |v|^2
$$

而在高速極限：

$$
|v|\gg v_c
$$

時，

$$
|F_D|\propto |v|^{n_\infty}
$$

例如

$$
n_\infty=1.8
$$

就代表高速時阻力逐漸接近速度的 1.8 次方。

---

### 有效阻力指數

定義

$$
n_{\mathrm{eff}}
=
\frac{d\ln |F_D|}
{d\ln |v|}
$$

本模型可以得到：

$$
n_{\mathrm{eff}}
=
2-
(2-n_\infty)
\frac{
(|v|/v_c)^p
}{
1+(|v|/v_c)^p
}
$$

因此阻力的「次方」本身也可以隨速度改變。

---

### Reynolds number

$$
Re=
\frac{\rho |v|L}{\mu}
$$

其中：

- $\rho$：流體密度
- $L$：特徵長度
- $\mu$：動力黏滯係數

現實中的 $C_D$ 往往可能隨 $Re$ 改變，因此固定 $C_D$ 只是近似。
        """
    )


with st.expander("📗 狹義相對論的適用範圍與物理方程（請先閱讀）"):
    st.markdown(r"""
### 實驗室參考系的相對論動量方程

本頁固定「**空氣近似靜止的實驗室參考系**」，用實驗室座標時間與速度描述運動。
一維相對論動量為 $p_y=\gamma mv$，其中 $\gamma=(1-v^2/c^2)^{-1/2}$。
""")
    st.latex(r"\frac{d(\gamma mv)}{dt}=F_{\mathrm{ext}}+F_D")
    st.markdown(r"""
若地面附近近似把重力表示成實驗室外力 $F_{\mathrm{ext}}=-mg$，
則 $a=(-g+F_D/m)/\gamma^3$。為了避免接近光速時數值不穩定，
程式選用 $u=\gamma v$ 作為積分變數：
""")
    st.latex(r"\begin{aligned} \frac{dy}{dt}&=\frac{u}{\sqrt{1+(u/c)^2}} \\ \frac{du}{dt}&=-g+\frac{F_D}{m} \end{aligned}")
    st.markdown(r"""
**重要限制：**固定 $-mg$ 是一個理想化外力假設，並非廣義相對論重力理論。
因此高速模擬只用來研究相對論動量效應，不是精確的地球重力軌跡。

### 四種阻力模型怎麼修正？

- **真空**：$F_D=0$，可以直接比較牛頓與相對論動量。
- **線性阻力**：$F_D=-bv$，原本的低速現象學近似保留。
- **平方阻力**：$F_D=-\frac12\rho C_D A v|v|$，其中 $\rho,C_D,A$ 是空氣靜止系的低速經驗參數。
- **可變冪次阻力**：原本的 $C_{D0},v_c,p,n_\infty$ 仍是相同參考系下的經驗參數，並非相對論推導出的通用高速阻力定律。

若球形物體滿足 $Re\ll1$ 的爬流條件，Stokes 線性阻力可以寫成
$F_D=-6\pi\mu rv$（$b=6\pi\mu r$）；一般落體若雷諾數很高，就不能直接套用 Stokes 公式。

**不能只靠加入相對論而任意修改阻力係數。** 本程式不會自行把 $C_D$、$\rho$ 或 $F_D$ 乘上 $\gamma^k$。
在高速時，流體壓縮性、物體形狀、受熱、粒子撞擊及介質能動量傳遞，都可能讓原本經驗式失效。
要預測真實的高速空氣阻力，需要額外實驗或更完整的流體模型。

### 反推阻力與擬合

牛頓力學：$F_D=m(a+g)$。在本程式假設下，相對論模型為：
""")
    st.latex(r"F_D=m\gamma^3a+mg")
    st.markdown(r"""
此處 $a=dv/dt$ 是實驗室座標加速度。從位置資料作二次微分會放大測量雜訊；
普通落體速度下相對論效應遠小於這些誤差，不能單憑 $R^2$ 很高就宣稱觀測到了相對論。

在這個簡化模型中，若終端速度存在，它仍滿足 $F_D(v_T)=mg$，
因此在相同阻力參數下，牛頓與相對論解的終端速度相同，
差別在**接近終端速度的過程**。

除 $v/c$ 外，還要檢查 Mach number、Reynolds number 與空氣壓縮性等適用條件。
速度接近光速時，牛頓解僅是反事實的數學比較，並不代表物體真的可以超光速。
""")


# ============================================================
# 4. Sidebar
# ============================================================

st.sidebar.header("模擬設定")

dynamics_mode = st.sidebar.radio(
    "動力學框架（同時影響模擬、阻力律辨識、參數擬合）",
    ["牛頓力學", "狹義相對論動量修正（理想化）"],
    help="空氣阻力仍是低速經驗模型；這裡修正的是 p=gamma*m*v 的運動方程。",
)
use_sr = dynamics_mode != "牛頓力學"
compare_sr = st.sidebar.checkbox(
    "加入牛頓 vs 相對論結果比較",
    value=True,
    help="使用同一個阻力模型及其相同參數，獨立積分兩套方程。",
)



model = st.sidebar.selectbox(
    "選擇物理模型",
    [
        "真空",
        "線性空氣阻力",
        "平方空氣阻力",
        "可變冪次阻力"
    ]
)


compare_all = st.sidebar.checkbox(
    "同時比較所有模型",
    value=False
)


st.sidebar.subheader("物體")

st.sidebar.selectbox(
    "物體預設",
    list(OBJECT_PRESETS.keys()),
    key="object_preset",
    on_change=apply_object_preset
)


# ============================================================
# 5. 圖表設定
# ============================================================

st.sidebar.subheader("圖表顯示")

zoom_mode = st.sidebar.selectbox(
    "時間軸顯示方式",
    [
        "完整時間範圍",
        "聚焦主要區間",
        "自訂時間範圍"
    ],
    index=1
)


custom_t_min = 0.0
custom_t_max = 10.0


if zoom_mode == "自訂時間範圍":

    custom_t_min = st.sidebar.number_input(
        "起始時間 (s)",
        min_value=0.0,
        value=0.0
    )

    custom_t_max = st.sidebar.number_input(
        "結束時間 (s)",
        min_value=0.1,
        value=10.0
    )


# ============================================================
# 6. 參數表單
# ============================================================

with st.sidebar.form("simulation_form"):

    st.subheader("初始條件")

    y0 = st.number_input(
        "初始高度 y₀ (m)",
        min_value=0.001,
        key="y0_input"
    )

    v0 = st.number_input(
        "初速度 v₀ (m/s)",
        key="v0_input"
    )

    st.caption(
        "v₀ > 0：向上；v₀ < 0：向下"
    )


    st.subheader("基本參數")

    m = st.number_input(
        "質量 m (kg)",
        min_value=1e-8,
        format="%.6g",
        key="m_input"
    )

    g = st.number_input(
        "重力加速度 g (m/s²)",
        min_value=0.001,
        key="g_input"
    )


    # --------------------------------------------------------
    # 線性阻力
    # --------------------------------------------------------

    if model == "線性空氣阻力" or compare_all:

        st.subheader("線性阻力")

        b = st.number_input(
            "b (kg/s)",
            min_value=0.0,
            key="b_input"
        )

    else:

        b = st.session_state.b_input


    # --------------------------------------------------------
    # 空氣相關參數
    # --------------------------------------------------------

    if (
        model in ["平方空氣阻力", "可變冪次阻力"]
        or compare_all
    ):

        st.subheader("空氣參數")

        rho = st.number_input(
            "空氣密度 ρ (kg/m³)",
            min_value=0.0,
            format="%.6g",
            key="rho_input"
        )

        Cd = st.number_input(
            "基準阻力係數 Cd",
            min_value=0.0,
            format="%.6g",
            key="Cd_input"
        )

        A = st.number_input(
            "迎風截面積 A (m²)",
            min_value=0.0,
            format="%.6g",
            key="A_input"
        )

        L = st.number_input(
            "特徵長度 L (m)",
            min_value=1e-8,
            format="%.6g",
            key="L_input"
        )

        mu = st.number_input(
            "空氣動力黏滯係數 μ (Pa·s)",
            min_value=1e-10,
            format="%.6g",
            key="mu_input"
        )

    else:

        rho = st.session_state.rho_input
        Cd = st.session_state.Cd_input
        A = st.session_state.A_input
        L = st.session_state.L_input
        mu = st.session_state.mu_input


    # --------------------------------------------------------
    # 可變冪次參數
    # --------------------------------------------------------

    if model == "可變冪次阻力" or compare_all:

        st.subheader("可變冪次模型")

        vc = st.number_input(
            "轉變特徵速度 vc (m/s)",
            min_value=0.01,
            key="vc_input"
        )

        p = st.number_input(
            "過渡參數 p",
            min_value=0.1,
            key="p_input"
        )

        n_high = st.number_input(
            "高速極限指數 n∞",
            min_value=1.0,
            max_value=2.0,
            key="n_high_input"
        )

    else:

        vc = st.session_state.vc_input
        p = st.session_state.p_input
        n_high = st.session_state.n_high_input


    submitted = st.form_submit_button(
        "🚀 開始模擬",
        type="primary"
    )


if submitted:
    st.session_state.has_run = True


# ============================================================
# 7. 共用物理函式
# ============================================================

def quadratic_drag(v, rho, Cd, A):

    return (
        -0.5
        * rho
        * Cd
        * A
        * v
        * abs(v)
    )


def variable_drag(
    v,
    rho,
    Cd0,
    A,
    vc,
    p,
    n_high
):

    speed = abs(v)

    if speed == 0:
        return 0.0

    delta = 2.0 - n_high

    correction = (
        1.0
        +
        (speed / vc) ** p
    ) ** (-delta / p)

    return (
        -0.5
        * rho
        * Cd0
        * A
        * v
        * speed
        * correction
    )


def effective_exponent(
    speed,
    vc,
    p,
    n_high
):

    speed = np.asarray(speed)

    q = (
        speed / vc
    ) ** p

    return (
        2.0
        -
        (2.0 - n_high)
        * q
        / (1.0 + q)
    )


def effective_cd(
    speed,
    Cd0,
    vc,
    p,
    n_high
):

    speed = np.asarray(speed)

    delta = 2.0 - n_high

    return (
        Cd0
        *
        (
            1.0
            +
            (speed / vc) ** p
        ) ** (-delta / p)
    )


def reynolds_number(
    speed,
    rho,
    L,
    mu
):

    return (
        rho
        * np.asarray(speed)
        * L
        / mu
    )


# ============================================================
# 8. 一般 solve_ivp 解法
# ============================================================

def solve_drag_model(
    y0, v0, m, g, drag_function, max_time=2000, relativistic=None
):
    """在空氣靜止的實驗室參考系積分，event 偵測落地。"""
    if relativistic is None:
        relativistic = use_sr
    if relativistic and abs(v0) >= C_LIGHT:
        raise ValueError("狹義相對論初速度必須滿足 |v₀| < c")

    sol = solve_ivp(
        lambda t, state: motion_rhs(t, state, m, g, drag_function, relativistic),
        (0.0, float(max_time)),
        initial_state(y0, v0, relativistic),
        events=_ground_event,
        dense_output=True,
        max_step=0.05,
        rtol=1e-9,
        atol=1e-11,
    )
    if not sol.success:
        raise RuntimeError("ODE 求解失敗：" + sol.message)

    t = sol.t
    y = sol.y[0]
    second = sol.y[1]
    v = velocity_from_u(second) if relativistic else second
    gamma = lorentz_gamma(v) if relativistic else np.ones_like(v)
    drag_values = np.array([drag_function(float(q)) for q in v])
    acceleration = (-g + drag_values/m) / (gamma**3)
    t_hit = float(sol.t_events[0][0]) if len(sol.t_events[0]) else None
    v_hit = (
        float(velocity_from_u(sol.y_events[0][0][1]))
        if relativistic and t_hit is not None
        else float(sol.y_events[0][0][1]) if t_hit is not None else None
    )
    return {
        "t": t, "y": y, "v": v, "a": acceleration,
        "F_drag": drag_values, "t_hit": t_hit, "v_hit": v_hit,
        "gamma": gamma, "relativistic": relativistic,
        "_dense_state": sol.sol,
    }


def _ground_event(t, state):
    return state[0]


_ground_event.direction = -1
_ground_event.terminal = True


# ============================================================
# 9. 真空
# ============================================================

def solve_vacuum(
    y0,
    v0,
    g
):

    if use_sr:
        output = solve_drag_model(y0, v0, m, g, lambda v: 0.0, relativistic=True)
        output["v_terminal"] = None
        return output

    t_hit = (
        v0
        +
        np.sqrt(
            v0**2
            +
            2 * g * y0
        )
    ) / g


    t = np.linspace(
        0,
        t_hit,
        500
    )


    y = (
        y0
        +
        v0 * t
        -
        0.5 * g * t**2
    )


    v = (
        v0
        -
        g * t
    )


    return {
        "t": t,
        "y": y,
        "v": v,
        "a": np.full_like(t, -g),
        "F_drag": np.zeros_like(t),
        "t_hit": t_hit,
        "v_hit": v[-1],
        "v_terminal": None,
        "gamma": np.ones_like(t),
        "relativistic": False,
        "_dense_state": lambda sample_t: np.vstack([
            y0 + v0 * np.asarray(sample_t) - 0.5 * g * np.asarray(sample_t)**2,
            v0 - g * np.asarray(sample_t),
        ]),
    }


# ============================================================
# 10. 線性
# ============================================================

def solve_linear():

    result = solve_drag_model(
        y0,
        v0,
        m,
        g,
        lambda v: -b * v
    )

    if b > 0:

        result["v_terminal"] = (
            -m * g / b
        )

    else:

        result["v_terminal"] = None

    return result


# ============================================================
# 11. 平方
# ============================================================

def solve_quadratic():

    result = solve_drag_model(
        y0,
        v0,
        m,
        g,
        lambda v: quadratic_drag(
            v,
            rho,
            Cd,
            A
        )
    )


    if (
        rho > 0
        and Cd > 0
        and A > 0
    ):

        result["v_terminal"] = -np.sqrt(
            2 * m * g
            /
            (
                rho
                * Cd
                * A
            )
        )

    else:

        result["v_terminal"] = None


    return result


# ============================================================
# 12. 可變冪次
# ============================================================

def solve_variable():

    drag = lambda v: variable_drag(
        v,
        rho,
        Cd,
        A,
        vc,
        p,
        n_high
    )


    result = solve_drag_model(
        y0,
        v0,
        m,
        g,
        drag
    )


    # 數值求終端速度
    if (
        rho > 0
        and Cd > 0
        and A > 0
    ):

        def terminal_equation(speed):

            drag_mag = abs(
                variable_drag(
                    -speed,
                    rho,
                    Cd,
                    A,
                    vc,
                    p,
                    n_high
                )
            )

            return drag_mag - m * g


        try:

            upper = 10.0

            while (
                terminal_equation(upper) < 0
                and upper < 10000
            ):

                upper *= 2


            vt = brentq(
                terminal_equation,
                1e-8,
                upper
            )


            result["v_terminal"] = -vt

        except Exception:

            result["v_terminal"] = None

    else:

        result["v_terminal"] = None


    speed = np.abs(
        result["v"]
    )


    result["n_eff"] = effective_exponent(
        speed,
        vc,
        p,
        n_high
    )


    result["Cd_eff"] = effective_cd(
        speed,
        Cd,
        vc,
        p,
        n_high
    )


    result["Re"] = reynolds_number(
        speed,
        rho,
        L,
        mu
    )


    return result


# ============================================================
# 13. 圖表工具
# ============================================================

def apply_zoom(
    fig,
    all_times
):

    if zoom_mode == "完整時間範圍":

        fig.update_xaxes(
            autorange=True
        )


    elif zoom_mode == "聚焦主要區間":

        valid = [
            arr
            for arr in all_times
            if (
                arr is not None
                and len(arr) > 0
            )
        ]


        if valid:

            focus = min(
                arr[-1]
                for arr in valid
            )

            fig.update_xaxes(
                range=[
                    0,
                    focus * 1.2
                ]
            )


    elif (
        zoom_mode
        == "自訂時間範圍"
    ):

        if custom_t_max > custom_t_min:

            fig.update_xaxes(
                range=[
                    custom_t_min,
                    custom_t_max
                ]
            )


    return fig


def show_plot(fig):

    st.plotly_chart(
        fig,
        width="stretch",
        config=PLOT_CONFIG
    )


def plot_single(
    result,
    key,
    title,
    y_title,
    name
):

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=result["t"],
            y=result[key],
            mode="lines",
            name=name
        )
    )


    fig.update_layout(
        title=title,
        xaxis_title="時間 (s)",
        yaxis_title=y_title,
        hovermode="x unified",
        dragmode="zoom"
    )


    return apply_zoom(
        fig,
        [result["t"]]
    )


# ============================================================
# 14. 模擬
# ============================================================

if st.session_state.has_run:

    st.subheader("目前使用的動力學：" + ("狹義相對論動量修正" if use_sr else "牛頓力學"))
    st.caption("左右側設定中的『動力學框架』決定主圖採用哪種方程；下面的相對論比較會獨立計算兩個版本。")
    if y0 > 1e5:
        st.warning(
            "目前初始高度超過 100 km。程式仍使用常數 g，不能代表真實地球重力及大氣環境。"
            "此案例只適合作為理想化固定外力的數學比較。"
        )
    if abs(v0) > 0.01 * C_LIGHT:
        st.info(
            "這是高速理想化測試。請查看下方的『牛頓力學 vs 狹義相對論動量修正』，"
            "特別是初始加速度、加速度比較及 Δv、Δy 差值圖；單看高度主圖可能近似直線。"
        )

    if use_sr and abs(v0) >= C_LIGHT:
        st.error("狹義相對論模式下必須滿足 |v₀| < c，請修改初速度。")
        st.stop()

    if abs(v0) >= C_LIGHT and not use_sr:
        st.warning("牛頓模型容許形式上超光速的初值，但這只是無物理意義的反事實比較。")

    if abs(v0) > 100 and model != "真空":
        st.warning(
            "初速度超過約 100 m/s：實際空氣的阻力係數可能受到 Reynolds/Mach 數、"
            "壓縮性與物體形狀影響。請勿把固定 Cd 或可變冪次低速經驗式視為可靠的高速預測。"
        )

    if (
        model in [
            "平方空氣阻力",
            "可變冪次阻力"
        ]
        or compare_all
    ):

        if rho == 0:

            st.warning(
                "ρ = 0，因此所有與空氣密度有關的阻力都會消失。"
            )


        if Cd == 0:

            st.warning(
                "Cd = 0，因此平方與可變冪次阻力都會消失。"
            )


        if A == 0:

            st.warning(
                "A = 0，因此平方與可變冪次阻力都會消失。"
            )


    if not compare_all:


        if model == "真空":

            result = solve_vacuum(
                y0,
                v0,
                g
            )


        elif model == "線性空氣阻力":

            result = solve_linear()


        elif model == "平方空氣阻力":

            result = solve_quadratic()


        else:

            result = solve_variable()


        st.header(
            f"模擬結果：{model}｜{'相對論動量修正' if use_sr else '牛頓力學'}"
        )

        if result["t_hit"] is None:
            st.info(
                f"這組初始條件在目前的 {float(result['t'][-1]):.1f} 秒模擬範圍內沒有落地，"
                "因此沒有落地時間與落地速度。" + ("真空中沒有空氣阻力，終端速度本來就無法定義。" if model == "真空" else "")
            )
        initial_n_acc = -g + float(result["F_drag"][0]) / m
        initial_gamma = float(lorentz_gamma(v0)) if abs(v0) < C_LIGHT else float('nan')
        initial_sr_acc = initial_n_acc / initial_gamma**3 if np.isfinite(initial_gamma) else float('nan')
        metric_cols = st.columns(3)
        metric_cols[0].metric("初始洛倫茲因子 γ₀", f"{initial_gamma:.5g}" if np.isfinite(initial_gamma) else "無定義")
        metric_cols[1].metric("同外力下牛頓初始加速度", f"{initial_n_acc:.6g} m/s²")
        metric_cols[2].metric("同外力下 SR 初始加速度", f"{initial_sr_acc:.6g} m/s²" if np.isfinite(initial_sr_acc) else "無定義")

        c1, c2, c3 = st.columns(3)


        if result["t_hit"] is not None:

            c1.metric(
                "落地時間",
                f'{result["t_hit"]:.4f} s'
            )

            c2.metric(
                "落地速度",
                f'{result["v_hit"]:.4f} m/s'
            )


        if result["v_terminal"] is not None:

            c3.metric(
                "終端速度",
                f'{result["v_terminal"]:.4f} m/s'
            )

        else:

            c3.metric(
                "終端速度",
                "無 / 無法定義"
            )


        show_plot(
            plot_single(
                result,
                "y",
                "高度 vs 時間",
                "高度 (m)",
                model
            )
        )


        show_plot(
            plot_single(
                result,
                "v",
                "速度 vs 時間",
                "速度 (m/s)",
                model
            )
        )


        show_plot(
            plot_single(
                result,
                "a",
                "加速度 vs 時間",
                "加速度 (m/s²)",
                model
            )
        )


        show_plot(
            plot_single(
                result,
                "F_drag",
                "阻力 vs 時間",
                "阻力 (N)",
                model
            )
        )


        # ----------------------------------------------------
        # 可變冪次模型額外分析
        # ----------------------------------------------------

        if model == "可變冪次阻力":

            st.header(
                "可變阻力模型分析"
            )


            fig_n = go.Figure()

            fig_n.add_trace(
                go.Scatter(
                    x=result["t"],
                    y=result["n_eff"],
                    mode="lines",
                    name="Effective exponent"
                )
            )

            fig_n.update_layout(
                title="Effective Drag Exponent vs Time",
                xaxis_title="時間 (s)",
                yaxis_title="n_eff"
            )

            show_plot(
                apply_zoom(
                    fig_n,
                    [result["t"]]
                )
            )


            fig_re = go.Figure()

            fig_re.add_trace(
                go.Scatter(
                    x=result["Re"],
                    y=result["Cd_eff"],
                    mode="lines",
                    name="Effective Cd"
                )
            )

            fig_re.update_layout(
                title="Effective Cd vs Reynolds Number",
                xaxis_title="Reynolds Number",
                yaxis_title="Effective Cd"
            )

            show_plot(fig_re)


    # ========================================================
    # 所有模型比較
    # ========================================================

    else:

        vacuum = solve_vacuum(
            y0,
            v0,
            g
        )

        linear = solve_linear()

        quadratic = solve_quadratic()

        variable = solve_variable()


        models = {
        "真空自由落體": vacuum,
        "線性空氣阻力": linear,
        "平方空氣阻力": quadratic,
        "可變冪次阻力": variable,
        }


        st.header(
            "四種阻力模型比較"
        )


        cols = st.columns(4)


        for column, (
            name,
            result
        ) in zip(
            cols,
            models.items()
        ):

            with column:

                st.subheader(name)

                if result["t_hit"] is not None:

                    st.metric(
                        "落地時間",
                        f'{result["t_hit"]:.3f} s'
                    )

                    st.metric(
                        "落地速度",
                        f'{result["v_hit"]:.3f} m/s'
                    )


        comparison_times = [
            r["t"]
            for r in models.values()
        ]


        for key, title, ylabel in [

            (
                "y",
                "高度隨時間變化比較",
                "高度 (m)"
            ),

            (
                "v",
                "速度隨時間變化比較",
                "速度 (m/s)"
            ),

            (
                "a",
                "加速度隨時間變化比較",
                "加速度 (m/s²)"
            ),

            (
                "F_drag",
                "空氣阻力隨時間變化比較",
                "空氣阻力 (N)"
            )
        ]:


            fig = go.Figure()


            for name, result in models.items():

                fig.add_trace(
                    go.Scatter(
                        x=result["t"],
                        y=result[key],
                        mode="lines",
                        name=name
                    )
                )


            fig.update_layout(
                title=title,
                xaxis_title="時間 (s)",
                yaxis_title=ylabel,
                hovermode="x unified"
            )


            show_plot(
                apply_zoom(
                    fig,
                    comparison_times
                )
            )


        st.subheader(
            "可變模型的有效阻力指數"
        )


        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=variable["t"],
                y=variable["n_eff"],
                mode="lines"
            )
        )

        fig.update_layout(
            xaxis_title="時間 (s)",
            yaxis_title="n_eff"
        )

        show_plot(fig)



# ============================================================
# 14A. 同一阻力律下，牛頓與相對論的可驗證差異
# ============================================================

if st.session_state.has_run and compare_sr:
    st.divider()
    st.header("🌌 牛頓力學 vs 狹義相對論動量修正")
    st.caption(
        "同一空氣靜止實驗室參考系、同一外力 -mg、同一組低速阻力律。"
        "不同之處僅為 p=mv 與 p=γmv。不是完整的相對論大氣阻力／重力模型。"
    )

    drag_for_comparison = {
        "真空": lambda v: 0.0,
        "線性空氣阻力": lambda v: -b * v,
        "平方空氣阻力": lambda v: quadratic_drag(v, rho, Cd, A),
        "可變冪次阻力": lambda v: variable_drag(v, rho, Cd, A, vc, p, n_high),
    }[model]
    try:
        if abs(v0) >= C_LIGHT:
            st.error("比較相對論運動時，請設定 |v₀| < c。")
        else:
            nr = solve_drag_model(
                y0, v0, m, g, drag_for_comparison, relativistic=False
            )
            sr = solve_drag_model(
                y0, v0, m, g, drag_for_comparison, relativistic=True
            )
            common_end = min(float(nr["t"][-1]), float(sr["t"][-1]))
            common_t = np.linspace(0.0, common_end, 400)
            s_nr = nr["_dense_state"](common_t)
            s_sr = sr["_dense_state"](common_t)
            y_nr, v_nr = s_nr
            y_sr = s_sr[0]
            v_sr = velocity_from_u(s_sr[1])
            beta_max = float(max(np.max(abs(v_sr)), abs(v0)) / C_LIGHT)
            corr = float(np.max(relative_acceleration_correction(v_sr)))
            t_nr = nr["t_hit"]
            t_sr = sr["t_hit"]
            ca, cb, cc = st.columns(3)
            ca.metric("最高速度 |v|/c", f"{beta_max:.4e}")
            cb.metric("最大 1 − 1/γ³", f"{corr:.4e}")
            if t_nr is not None and t_sr is not None:
                cc.metric("SR − 牛頓落地時間", f"{t_sr - t_nr:+.5e} s")
            else:
                cc.metric("落地時間差", "指定時間內未全部落地")

            st.markdown("**直接比較：初始加速度（相同的外力）**")
            n_acc0 = -g + float(drag_for_comparison(v0)) / m
            sr_acc0 = n_acc0 / float(lorentz_gamma(v0))**3
            ad1, ad2, ad3 = st.columns(3)
            ad1.metric("牛頓初始加速度", f"{n_acc0:.6g} m/s²")
            ad2.metric("SR 初始加速度", f"{sr_acc0:.6g} m/s²")
            ad3.metric("SR / 牛頓（同外力）", f"{float(1/lorentz_gamma(v0)**3):.6g}")

            fig_y = go.Figure()
            fig_v = go.Figure()
            fig_d = go.Figure()
            fig_dv = go.Figure()
            fig_acc = go.Figure()
            for label, yy, vv in [
                ("牛頓", y_nr, v_nr), ("相對論動量修正", y_sr, v_sr)
            ]:
                fig_y.add_trace(go.Scatter(x=common_t, y=yy, mode="lines", name=label))
                fig_v.add_trace(go.Scatter(x=common_t, y=vv, mode="lines", name=label))
            fig_d.add_trace(go.Scatter(
                x=common_t, y=y_sr-y_nr, mode="lines", name="SR − 牛頓"
            ))
            fig_dv.add_trace(go.Scatter(
                x=common_t, y=v_sr-v_nr, mode="lines", name="SR − 牛頓"
            ))
            drag_nr = np.array([drag_for_comparison(float(q)) for q in v_nr])
            drag_sr = np.array([drag_for_comparison(float(q)) for q in v_sr])
            a_nr = -g + drag_nr/m
            a_sr = (-g + drag_sr/m) / lorentz_gamma(v_sr)**3
            fig_acc.add_trace(go.Scatter(x=common_t, y=a_nr, mode="lines", name="牛頓"))
            fig_acc.add_trace(go.Scatter(x=common_t, y=a_sr, mode="lines", name="SR"))
            fig_y.update_layout(title="高度—時間比較", xaxis_title="t (s)", yaxis_title="y (m)")
            fig_v.update_layout(title="速度—時間比較", xaxis_title="t (s)", yaxis_title="v (m/s)")
            fig_d.update_layout(title="高度差 Δy（SR − 牛頓）", xaxis_title="t (s)", yaxis_title="Δy (m)")
            fig_dv.update_layout(title="速度差 Δv（SR − 牛頓）", xaxis_title="t (s)", yaxis_title="Δv (m/s)")
            fig_acc.update_layout(title="加速度—時間：牛頓與 SR", xaxis_title="t (s)", yaxis_title="a (m/s²)")
            col1, col2 = st.columns(2)
            with col1:
                show_plot(fig_y)
            with col2:
                show_plot(fig_v)
            if beta_max < 1e-5:
                st.info(
                    "速度遠小於 c：理論修正非常微小。Δy 的直接浮點相減可能顯示 0 "
                    "或數值雜訊；請以右上方由 log1p/expm1 精確計算的 1−1/γ³ 作量級判斷，"
                    "不可將 Δy 圖的數值雜訊當作相對論的實驗訊號。"
                )
            else:
                show_plot(fig_d)
                show_plot(fig_dv)
            if beta_max >= 1e-5:
                show_plot(fig_acc)
            if y0 > 1e5:
                st.warning("這是固定 -mg 外力下的理想比較，不是高度達數十萬公尺以上時的真實地球重力軌跡。")
            if beta_max * C_LIGHT > 100 and model != "真空":
                st.warning(
                    "速度已超過約 100 m/s：此處的阻力是把原本經驗式延伸到未驗證速度的數學示範。"
                    "真實空氣的 Cd 可能隨 Re/Mach、壓縮性及熱效應明顯變化。"
                )
            st.caption(
                "實驗可測性：若理論修正小於觀測不確定度及數值誤差，"
                "則本專題不能用這些落體數據驗證狹義相對論。"
            )
    except Exception as comparison_error:
        st.error(f"相對論比較計算失敗：{comparison_error}")


# ============================================================
# 15. 阻力律辨識：由實驗資料反推出局部冪次
# ============================================================

st.divider()

st.header("🔎 阻力律辨識：分析運動過程中的有效冪次")

st.write(
    "這個模式不預先指定阻力一定是線性、平方或 1.8 次方。"
    "程式先由時間－高度資料估計速度與加速度，再由牛頓第二定律"
    "反推出阻力，最後分析阻力與速率之間的局部冪次關係。"
)

st.warning(
    "重要假設：牛頓模式 F_drag=m(a+g)，SR 動量模式 F_drag=mγ³a+mg；"
    "兩者都假設除了近似重力與空氣阻力外，"
    "沒有其他未建模外力。若存在風、浮力、推力、接觸力、物體姿態快速改變等效應，"
    "推得的『有效冪次』應視為整體動力學的經驗描述，而不是唯一的空氣阻力定律。"
)

with st.expander("📘 這個分析在做什麼？"):

    st.markdown(
        r"""
若某個局部區間近似滿足

$$
|F_D| = K |v|^n
$$

取自然對數後：

$$
\ln |F_D|
=
\ln K
+
n\ln |v|
$$

因此在局部的

$$
\ln |F_D|
\text{ 對 }
\ln |v|
$$

圖上，斜率就是有效冪次：

$$
n_{\mathrm{eff}}
=
\frac{d\ln |F_D|}
{d\ln |v|}
$$

程式不會預先指定 $n=1$、$2$ 或其他數值，而是使用滑動視窗局部迴歸估計
$n_{\mathrm{eff}}(t)$，再用分段常數模型與 BIC 自動判斷可能的轉變區段。
        """
    )


def _make_odd_window(value, n_points, polyorder=2):
    """把視窗修正成可用的奇數長度。"""

    value = int(value)

    if value % 2 == 0:
        value += 1

    max_window = n_points if n_points % 2 == 1 else n_points - 1

    min_window = polyorder + 2

    if min_window % 2 == 0:
        min_window += 1

    value = max(value, min_window)
    value = min(value, max_window)

    return value


def estimate_local_power_exponent(
    time_array,
    velocity_array,
    drag_array,
    window_points,
    min_speed,
    min_force,
    require_opposite_direction=True
):
    """
    在時間滑動視窗內擬合：
        log|F| = intercept + n * log|v|
    回傳每個中心點的 n_eff 與局部 R²。
    """

    t_arr = np.asarray(time_array, dtype=float)
    v_arr = np.asarray(velocity_array, dtype=float)
    f_arr = np.asarray(drag_array, dtype=float)

    n_eff = np.full(len(t_arr), np.nan)
    local_r2 = np.full(len(t_arr), np.nan)

    half = window_points // 2

    speed = np.abs(v_arr)
    force_mag = np.abs(f_arr)

    base_valid = (
        np.isfinite(speed)
        & np.isfinite(force_mag)
        & (speed > min_speed)
        & (force_mag > min_force)
    )

    if require_opposite_direction:
        # 理想阻力應與速度方向相反，因此 F_drag * v <= 0
        base_valid &= (f_arr * v_arr <= 0)

    for center in range(half, len(t_arr) - half):

        sl = slice(
            center - half,
            center + half + 1
        )

        valid = base_valid[sl]

        if np.count_nonzero(valid) < max(5, window_points // 3):
            continue

        local_speed = speed[sl][valid]
        local_force = force_mag[sl][valid]

        x = np.log(local_speed)
        y = np.log(local_force)

        # 速度變化太小時，斜率沒有可信度
        if np.ptp(x) < 0.05:
            continue

        slope, intercept = np.polyfit(x, y, 1)

        prediction = intercept + slope * x

        ss_res = np.sum((y - prediction) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot > 0:
            r2_value = 1.0 - ss_res / ss_tot
        else:
            r2_value = np.nan

        n_eff[center] = slope
        local_r2[center] = r2_value

    return n_eff, local_r2


def piecewise_constant_bic(
    values,
    max_segments=5,
    min_segment_points=8
):
    """
    用動態規劃尋找 piecewise-constant segmentation。
    以 BIC 在 1 ~ max_segments 中選擇最佳段數。

    回傳：
      boundaries: [0, ..., N]
      means: 每段平均值
      bic_table: 每個段數的 BIC
    """

    x = np.asarray(values, dtype=float)
    n = len(x)

    if n < max(2 * min_segment_points, 6):
        return [0, n], [float(np.mean(x))], {1: np.nan}

    max_segments = int(
        min(
            max_segments,
            max(1, n // min_segment_points)
        )
    )

    prefix = np.concatenate(
        [[0.0], np.cumsum(x)]
    )

    prefix_sq = np.concatenate(
        [[0.0], np.cumsum(x * x)]
    )

    def segment_sse(i, j):
        count = j - i

        if count <= 0:
            return np.inf

        total = prefix[j] - prefix[i]
        total_sq = prefix_sq[j] - prefix_sq[i]

        return max(
            total_sq - total * total / count,
            0.0
        )

    inf = np.inf

    dp = np.full(
        (max_segments + 1, n + 1),
        inf
    )

    back = np.full(
        (max_segments + 1, n + 1),
        -1,
        dtype=int
    )

    dp[0, 0] = 0.0

    for k in range(1, max_segments + 1):

        min_end = k * min_segment_points

        for end in range(min_end, n + 1):

            start_min = (k - 1) * min_segment_points
            start_max = end - min_segment_points

            for start in range(start_min, start_max + 1):

                previous = dp[k - 1, start]

                if not np.isfinite(previous):
                    continue

                candidate = (
                    previous
                    + segment_sse(start, end)
                )

                if candidate < dp[k, end]:

                    dp[k, end] = candidate
                    back[k, end] = start

    bic_table = {}

    for k in range(1, max_segments + 1):

        rss = dp[k, n]

        if not np.isfinite(rss):
            continue

        rss_per_point = max(
            rss / n,
            1e-12
        )

        # k 個區段平均值 + k-1 個轉變位置
        parameter_count = (
            k
            + (k - 1)
        )

        bic = (
            n * np.log(rss_per_point)
            + parameter_count * np.log(n)
        )

        bic_table[k] = float(bic)

    if not bic_table:
        return [0, n], [float(np.mean(x))], {1: np.nan}

    best_k = min(
        bic_table,
        key=bic_table.get
    )

    boundaries = [n]

    current_end = n

    for k in range(best_k, 0, -1):

        start = back[k, current_end]

        if start < 0:
            start = 0

        boundaries.append(start)
        current_end = start

    boundaries = sorted(
        set(boundaries)
    )

    if boundaries[0] != 0:
        boundaries.insert(0, 0)

    if boundaries[-1] != n:
        boundaries.append(n)

    means = []

    for i in range(len(boundaries) - 1):

        start = boundaries[i]
        end = boundaries[i + 1]

        means.append(
            float(np.mean(x[start:end]))
        )

    return boundaries, means, bic_table


law_file = st.file_uploader(
    "上傳時間－高度 CSV 進行阻力律辨識",
    type=["csv"],
    key="law_identification_csv"
)

if law_file is not None:

    try:

        law_df = pd.read_csv(law_file)

        st.subheader("1. 資料設定")

        st.dataframe(
            law_df.head(20),
            width="stretch"
        )

        law_columns = list(
            law_df.columns
        )

        if len(law_columns) < 2:

            st.error(
                "CSV 至少需要兩個欄位：時間與高度。"
            )

        else:

            col_time, col_height = st.columns(2)

            with col_time:

                law_time_column = st.selectbox(
                    "時間欄位",
                    law_columns,
                    index=0,
                    key="law_time_column"
                )

            with col_height:

                law_height_column = st.selectbox(
                    "高度欄位",
                    law_columns,
                    index=1,
                    key="law_height_column"
                )

            clean_law_df = (
                law_df[
                    [
                        law_time_column,
                        law_height_column
                    ]
                ]
                .dropna()
                .copy()
            )

            clean_law_df[
                law_time_column
            ] = pd.to_numeric(
                clean_law_df[
                    law_time_column
                ],
                errors="coerce"
            )

            clean_law_df[
                law_height_column
            ] = pd.to_numeric(
                clean_law_df[
                    law_height_column
                ],
                errors="coerce"
            )

            clean_law_df = (
                clean_law_df
                .dropna()
                .sort_values(law_time_column)
            )

            # 同一時間若有重複資料，取平均高度
            clean_law_df = (
                clean_law_df
                .groupby(
                    law_time_column,
                    as_index=False
                )[law_height_column]
                .mean()
            )

            if len(clean_law_df) < 9:

                st.error(
                    "有效資料點太少。為了平滑、微分與局部冪次分析，建議至少提供 9 點，最好有數十到數百點。"
                )

            else:

                raw_t = clean_law_df[
                    law_time_column
                ].to_numpy(dtype=float)

                raw_y = clean_law_df[
                    law_height_column
                ].to_numpy(dtype=float)

                raw_t = (
                    raw_t
                    - raw_t[0]
                )

                if np.any(
                    np.diff(raw_t) <= 0
                ):

                    st.error(
                        "時間資料必須嚴格遞增。"
                    )

                else:

                    # ------------------------------------------------
                    # 重新取樣成等時間間距，方便 Savitzky-Golay 微分
                    # ------------------------------------------------

                    grid_points_default = int(
                        np.clip(
                            len(raw_t),
                            51,
                            1500
                        )
                    )

                    grid_points = st.slider(
                        "等間距重新取樣點數",
                        min_value=51,
                        max_value=min(
                            2000,
                            max(
                                101,
                                len(raw_t) * 4
                            )
                        ),
                        value=min(
                            grid_points_default,
                            min(
                                2000,
                                max(
                                    101,
                                    len(raw_t) * 4
                                )
                            )
                        ),
                        step=10,
                        key="law_grid_points"
                    )

                    t_uniform = np.linspace(
                        raw_t[0],
                        raw_t[-1],
                        grid_points
                    )

                    y_uniform_raw = np.interp(
                        t_uniform,
                        raw_t,
                        raw_y
                    )

                    dt_uniform = (
                        t_uniform[1]
                        - t_uniform[0]
                    )

                    # ------------------------------------------------
                    # 分析設定
                    # ------------------------------------------------

                    st.subheader(
                        "2. 平滑、微分與局部冪次設定"
                    )

                    setting_c1, setting_c2 = (
                        st.columns(2)
                    )

                    with setting_c1:

                        max_smooth = (
                            grid_points
                            if grid_points % 2 == 1
                            else grid_points - 1
                        )

                        default_smooth = min(
                            31,
                            max_smooth
                        )

                        if default_smooth % 2 == 0:
                            default_smooth -= 1

                        smooth_window = st.slider(
                            "Savitzky–Golay 平滑視窗（點數）",
                            min_value=5,
                            max_value=max(
                                5,
                                max_smooth
                            ),
                            value=max(
                                5,
                                default_smooth
                            ),
                            step=2,
                            key="law_smooth_window"
                        )

                        polyorder = st.slider(
                            "平滑多項式階數",
                            min_value=2,
                            max_value=4,
                            value=3,
                            key="law_polyorder"
                        )

                        min_speed = st.number_input(
                            "分析的最小速率 |v| (m/s)",
                            min_value=0.0,
                            value=0.5,
                            step=0.1,
                            key="law_min_speed"
                        )

                        min_force = st.number_input(
                            "分析的最小阻力大小 |F| (N)",
                            min_value=0.0,
                            value=0.001,
                            format="%.6g",
                            key="law_min_force"
                        )

                    with setting_c2:

                        max_local = (
                            grid_points
                            if grid_points % 2 == 1
                            else grid_points - 1
                        )

                        default_local = min(
                            31,
                            max_local
                        )

                        if default_local % 2 == 0:
                            default_local -= 1

                        local_window = st.slider(
                            "局部 log-log 迴歸視窗（點數）",
                            min_value=7,
                            max_value=max(
                                7,
                                max_local
                            ),
                            value=max(
                                7,
                                default_local
                            ),
                            step=2,
                            key="law_local_window"
                        )

                        max_segments = st.slider(
                            "最多允許幾個阻力冪次區段",
                            min_value=1,
                            max_value=6,
                            value=4,
                            key="law_max_segments"
                        )

                        min_segment_points = st.slider(
                            "每個區段最少需要幾個局部估計點",
                            min_value=4,
                            max_value=30,
                            value=8,
                            key="law_min_segment_points"
                        )

                        require_opposite = st.checkbox(
                            "只使用『推得阻力方向與速度相反』的資料點",
                            value=True,
                            key="law_require_opposite"
                        )

                    run_law_analysis = st.button(
                        "🔍 開始阻力律辨識",
                        type="primary",
                        key="run_law_analysis"
                    )

                    if run_law_analysis:

                        smooth_window = _make_odd_window(
                            smooth_window,
                            grid_points,
                            polyorder
                        )

                        local_window = _make_odd_window(
                            local_window,
                            grid_points,
                            2
                        )

                        if smooth_window <= polyorder:

                            st.error(
                                "平滑視窗必須大於多項式階數。"
                            )

                        else:

                            # ----------------------------------------
                            # Savitzky-Golay：同時取得平滑位置、速度、加速度
                            # ----------------------------------------

                            y_smooth = savgol_filter(
                                y_uniform_raw,
                                window_length=smooth_window,
                                polyorder=polyorder,
                                deriv=0
                            )

                            velocity_est = savgol_filter(
                                y_uniform_raw,
                                window_length=smooth_window,
                                polyorder=polyorder,
                                deriv=1,
                                delta=dt_uniform
                            )

                            acceleration_est = savgol_filter(
                                y_uniform_raw,
                                window_length=smooth_window,
                                polyorder=polyorder,
                                deriv=2,
                                delta=dt_uniform
                            )

                            # 向上為正：
                            # m a = -m g + F_drag
                            # => F_drag = m(a + g)
                            if use_sr and np.any(np.abs(velocity_est) >= C_LIGHT):
                                st.error("實驗估計速度達到或超過光速，無法使用此相對論反推模型。")
                                st.stop()

                            gamma_est = (
                                lorentz_gamma(velocity_est)
                                if use_sr else np.ones_like(velocity_est)
                            )
                            # 實驗室力：dp/dt = m gamma^3 a = -mg + F_drag
                            drag_est = m * (gamma_est**3 * acceleration_est + g)

                            if use_sr and np.max(np.abs(velocity_est)) / C_LIGHT < 1e-5:
                                st.info(
                                    "本次速度下，相對論阻力反推與牛頓公式的差異遠小於"
                                    "位置二次微分的典型數值誤差；不可據此宣稱量測到相對論。"
                                )

                            speed_est = np.abs(
                                velocity_est
                            )

                            if (
                                rho > 0
                                and L > 0
                                and mu > 0
                            ):

                                re_est = (
                                    rho
                                    * speed_est
                                    * L
                                    / mu
                                )

                            else:

                                re_est = np.full(
                                    len(speed_est),
                                    np.nan
                                )

                            n_eff_est, local_r2_est = (
                                estimate_local_power_exponent(
                                    t_uniform,
                                    velocity_est,
                                    drag_est,
                                    local_window,
                                    min_speed,
                                    min_force,
                                    require_opposite
                                )
                            )

                            finite_n_mask = (
                                np.isfinite(n_eff_est)
                            )

                            finite_indices = np.where(
                                finite_n_mask
                            )[0]

                            if np.count_nonzero(
                                finite_n_mask
                            ) < max(
                                12,
                                2 * min_segment_points
                            ):

                                st.error(
                                    "目前可用的局部冪次估計點太少，無法可靠偵測轉變。"
                                    "可以嘗試增加實驗資料點、降低最小速率/阻力門檻，"
                                    "或調整平滑與局部迴歸視窗。"
                                )

                            else:

                                n_values = (
                                    n_eff_est[
                                        finite_n_mask
                                    ]
                                )

                                n_times = (
                                    t_uniform[
                                        finite_n_mask
                                    ]
                                )

                                n_speeds = (
                                    speed_est[
                                        finite_n_mask
                                    ]
                                )

                                n_re = (
                                    re_est[
                                        finite_n_mask
                                    ]
                                )

                                n_local_r2 = (
                                    local_r2_est[
                                        finite_n_mask
                                    ]
                                )

                                boundaries, segment_means, bic_table = (
                                    piecewise_constant_bic(
                                        n_values,
                                        max_segments=max_segments,
                                        min_segment_points=min_segment_points
                                    )
                                )

                                # ------------------------------------
                                # 建立分段結果
                                # ------------------------------------

                                segment_rows = []
                                change_times = []

                                for seg_i in range(
                                    len(boundaries) - 1
                                ):

                                    start = boundaries[seg_i]
                                    end = boundaries[seg_i + 1]

                                    t_start = float(
                                        n_times[start]
                                    )

                                    t_end = float(
                                        n_times[end - 1]
                                    )

                                    speed_start = float(
                                        n_speeds[start]
                                    )

                                    speed_end = float(
                                        n_speeds[end - 1]
                                    )

                                    mean_n = float(
                                        np.mean(
                                            n_values[
                                                start:end
                                            ]
                                        )
                                    )

                                    median_r2 = float(
                                        np.nanmedian(
                                            n_local_r2[
                                                start:end
                                            ]
                                        )
                                    )

                                    if np.any(
                                        np.isfinite(
                                            n_re[
                                                start:end
                                            ]
                                        )
                                    ):

                                        re_min = float(
                                            np.nanmin(
                                                n_re[
                                                    start:end
                                                ]
                                            )
                                        )

                                        re_max = float(
                                            np.nanmax(
                                                n_re[
                                                    start:end
                                                ]
                                            )
                                        )

                                        re_text = (
                                            f"{re_min:.3g} ～ {re_max:.3g}"
                                        )

                                    else:

                                        re_text = "未計算"

                                    segment_rows.append(
                                        {
                                            "區段":
                                                seg_i + 1,
                                            "時間範圍 (s)":
                                                f"{t_start:.3f} ～ {t_end:.3f}",
                                            "速率範圍 (m/s)":
                                                f"{min(speed_start, speed_end):.3f} ～ {max(speed_start, speed_end):.3f}",
                                            "Re 範圍":
                                                re_text,
                                            "平均有效冪次 n":
                                                mean_n,
                                            "局部擬合 R² 中位數":
                                                median_r2,
                                            "有效估計點數":
                                                end - start
                                        }
                                    )

                                    if (
                                        seg_i
                                        < len(boundaries) - 2
                                    ):

                                        next_start = (
                                            boundaries[
                                                seg_i + 1
                                            ]
                                        )

                                        change_times.append(
                                            float(
                                                n_times[
                                                    next_start
                                                ]
                                            )
                                        )

                                segment_df = pd.DataFrame(
                                    segment_rows
                                )

                                # ------------------------------------
                                # 結果摘要
                                # ------------------------------------

                                st.subheader(
                                    "3. 自動辨識結果"
                                )

                                st.success(
                                    f"BIC 選出的最佳模型共有 {len(segment_rows)} 個有效冪次區段。"
                                )

                                if change_times:

                                    change_text = "、".join(
                                        f"{value:.3f} s"
                                        for value in change_times
                                    )

                                    st.write(
                                        "偵測到的主要轉變時間約為："
                                        f" **{change_text}**"
                                    )

                                else:

                                    st.write(
                                        "在目前設定下，沒有偵測到需要額外分段的主要轉變。"
                                    )

                                st.dataframe(
                                    segment_df,
                                    width="stretch"
                                )

                                # ------------------------------------
                                # 位置資料與平滑
                                # ------------------------------------

                                st.subheader(
                                    "4. 原始位置資料與平滑結果"
                                )

                                fig_position = go.Figure()

                                fig_position.add_trace(
                                    go.Scatter(
                                        x=raw_t,
                                        y=raw_y,
                                        mode="markers",
                                        name="原始實驗資料"
                                    )
                                )

                                fig_position.add_trace(
                                    go.Scatter(
                                        x=t_uniform,
                                        y=y_smooth,
                                        mode="lines",
                                        name="平滑位置"
                                    )
                                )

                                fig_position.update_layout(
                                    xaxis_title="時間 (s)",
                                    yaxis_title="高度 (m)",
                                    hovermode="x unified"
                                )

                                show_plot(
                                    fig_position
                                )

                                # ------------------------------------
                                # 速度與加速度
                                # ------------------------------------

                                st.subheader(
                                    "5. 由位置資料估計出的速度"
                                )

                                fig_velocity_id = go.Figure()

                                fig_velocity_id.add_trace(
                                    go.Scatter(
                                        x=t_uniform,
                                        y=velocity_est,
                                        mode="lines",
                                        name="估計速度"
                                    )
                                )

                                fig_velocity_id.update_layout(
                                    xaxis_title="時間 (s)",
                                    yaxis_title="速度 (m/s)"
                                )

                                show_plot(
                                    fig_velocity_id
                                )

                                st.subheader(
                                    "6. 由位置資料估計出的加速度"
                                )

                                fig_acc_id = go.Figure()

                                fig_acc_id.add_trace(
                                    go.Scatter(
                                        x=t_uniform,
                                        y=acceleration_est,
                                        mode="lines",
                                        name="估計加速度"
                                    )
                                )

                                fig_acc_id.update_layout(
                                    xaxis_title="時間 (s)",
                                    yaxis_title="加速度 (m/s²)"
                                )

                                show_plot(
                                    fig_acc_id
                                )

                                # ------------------------------------
                                # 推得阻力
                                # ------------------------------------

                                st.subheader(
                                    "7. 由牛頓第二定律反推出的阻力"
                                )

                                fig_drag_id = go.Figure()

                                fig_drag_id.add_trace(
                                    go.Scatter(
                                        x=t_uniform,
                                        y=drag_est,
                                        mode="lines",
                                        name="推得阻力"
                                    )
                                )

                                fig_drag_id.add_hline(
                                    y=0,
                                    line_dash="dash"
                                )

                                fig_drag_id.update_layout(
                                    xaxis_title="時間 (s)",
                                    yaxis_title="阻力 (N)"
                                )

                                show_plot(
                                    fig_drag_id
                                )

                                # ------------------------------------
                                # n_eff vs time
                                # ------------------------------------

                                st.subheader(
                                    "8. 有效阻力冪次 n_eff 隨時間變化"
                                )

                                fig_n_time = go.Figure()

                                fig_n_time.add_trace(
                                    go.Scatter(
                                        x=n_times,
                                        y=n_values,
                                        mode="lines+markers",
                                        name="局部 n_eff"
                                    )
                                )

                                for seg_i in range(
                                    len(boundaries) - 1
                                ):

                                    start = boundaries[seg_i]
                                    end = boundaries[seg_i + 1]

                                    fig_n_time.add_trace(
                                        go.Scatter(
                                            x=[
                                                n_times[start],
                                                n_times[end - 1]
                                            ],
                                            y=[
                                                segment_means[seg_i],
                                                segment_means[seg_i]
                                            ],
                                            mode="lines",
                                            name=f"區段 {seg_i + 1} 平均"
                                        )
                                    )

                                for change_time in change_times:

                                    fig_n_time.add_vline(
                                        x=change_time,
                                        line_dash="dash"
                                    )

                                fig_n_time.update_layout(
                                    xaxis_title="時間 (s)",
                                    yaxis_title="有效冪次 n_eff",
                                    hovermode="x unified"
                                )

                                show_plot(
                                    fig_n_time
                                )

                                # ------------------------------------
                                # n_eff vs speed
                                # ------------------------------------

                                st.subheader(
                                    "9. 有效冪次與速率的關係"
                                )

                                fig_n_speed = go.Figure()

                                fig_n_speed.add_trace(
                                    go.Scatter(
                                        x=n_speeds,
                                        y=n_values,
                                        mode="markers",
                                        name="n_eff"
                                    )
                                )

                                fig_n_speed.update_layout(
                                    xaxis_title="速率 |v| (m/s)",
                                    yaxis_title="有效冪次 n_eff"
                                )

                                show_plot(
                                    fig_n_speed
                                )

                                # ------------------------------------
                                # n_eff vs Reynolds
                                # ------------------------------------

                                if np.any(
                                    np.isfinite(n_re)
                                ):

                                    st.subheader(
                                        "10. 有效冪次與 Reynolds number 的關係"
                                    )

                                    fig_n_re = go.Figure()

                                    fig_n_re.add_trace(
                                        go.Scatter(
                                            x=n_re,
                                            y=n_values,
                                            mode="markers",
                                            name="n_eff"
                                        )
                                    )

                                    fig_n_re.update_layout(
                                        xaxis_title="Reynolds number Re",
                                        yaxis_title="有效冪次 n_eff"
                                    )

                                    show_plot(
                                        fig_n_re
                                    )

                                # ------------------------------------
                                # log-log plot
                                # ------------------------------------

                                st.subheader(
                                    "11. log|F|－log|v| 關係"
                                )

                                valid_log = (
                                    np.isfinite(speed_est)
                                    & np.isfinite(drag_est)
                                    & (speed_est > min_speed)
                                    & (
                                        np.abs(drag_est)
                                        > min_force
                                    )
                                )

                                if require_opposite:
                                    valid_log &= (
                                        drag_est
                                        * velocity_est
                                        <= 0
                                    )

                                fig_loglog = go.Figure()

                                fig_loglog.add_trace(
                                    go.Scatter(
                                        x=np.log(
                                            speed_est[
                                                valid_log
                                            ]
                                        ),
                                        y=np.log(
                                            np.abs(
                                                drag_est[
                                                    valid_log
                                                ]
                                            )
                                        ),
                                        mode="markers",
                                        name="實驗推得資料"
                                    )
                                )

                                fig_loglog.update_layout(
                                    xaxis_title="ln|v|",
                                    yaxis_title="ln|F_drag|"
                                )

                                show_plot(
                                    fig_loglog
                                )

                                # ------------------------------------
                                # BIC 比較
                                # ------------------------------------

                                with st.expander(
                                    "查看不同區段數的 BIC 比較"
                                ):

                                    bic_df = pd.DataFrame(
                                        {
                                            "區段數":
                                                list(
                                                    bic_table.keys()
                                                ),
                                            "BIC":
                                                list(
                                                    bic_table.values()
                                                )
                                        }
                                    )

                                    st.dataframe(
                                        bic_df,
                                        width="stretch"
                                    )

                                    st.caption(
                                        "BIC 越小越好。它會在擬合改善與模型複雜度之間做平衡，"
                                        "避免只因增加很多區段就得到表面上更好的擬合。"
                                    )

                                # ------------------------------------
                                # 可下載分析資料
                                # ------------------------------------

                                analysis_df = pd.DataFrame(
                                    {
                                        "time":
                                            t_uniform,
                                        "height_interpolated":
                                            y_uniform_raw,
                                        "height_smoothed":
                                            y_smooth,
                                        "velocity":
                                            velocity_est,
                                        "acceleration":
                                            acceleration_est,
                                        "gamma": gamma_est,
                                        "drag_force":
                                            drag_est,
                                        "speed":
                                            speed_est,
                                        "Re":
                                            re_est,
                                        "n_eff":
                                            n_eff_est,
                                        "local_loglog_R2":
                                            local_r2_est
                                    }
                                )

                                analysis_csv = (
                                    analysis_df
                                    .to_csv(
                                        index=False
                                    )
                                    .encode("utf-8")
                                )

                                st.download_button(
                                    "⬇️ 下載阻力律辨識結果 CSV",
                                    data=analysis_csv,
                                    file_name="drag_law_identification.csv",
                                    mime="text/csv",
                                    key="download_drag_law_identification"
                                )

                                st.info(
                                    "解讀時請特別比較 n_eff 對時間、速率與 Re 的三種圖。"
                                    "若不同初始條件的多次實驗在不同時間發生轉變，"
                                    "但都在相近速率或相近 Re 發生，則更支持轉變是由運動狀態"
                                    "而非『經過幾秒』本身控制。"
                                )

    except Exception as law_error:

        st.error(
            f"阻力律辨識時發生錯誤：{law_error}"
        )



# ============================================================
# 16. 參數化可變冪次模型擬合
# ============================================================

st.divider()

st.header("🧪 參數化可變冪次模型擬合（比較用）")

st.write(
    "上傳實驗量測得到的時間－高度資料，"
    "程式會嘗試找出最符合資料的可變冪次阻力參數。"
)

st.markdown(
    """
CSV 格式例如：

```text
time,height
0.00,100.00
0.05,99.99
0.10,99.95
0.15,99.88
```
"""
)
uploaded_file = st.file_uploader(
    "上傳 CSV",
    type=["csv"]
)
if uploaded_file is not None:
    try:

        # ====================================================
        # 讀取資料
        # ====================================================

        df = pd.read_csv(uploaded_file)

        st.subheader("資料預覽")

        st.dataframe(
            df.head(20),
            width="stretch"
        )


        columns = list(df.columns)

        if len(columns) < 2:

            st.error(
                "CSV 至少需要兩個欄位，例如 time 與 height。"
            )

            st.stop()


        # ====================================================
        # 選擇資料欄位
        # ====================================================

        c1, c2 = st.columns(2)

        with c1:

            time_column = st.selectbox(
                "時間欄位",
                columns,
                index=0
            )


        with c2:

            height_column = st.selectbox(
                "高度欄位",
                columns,
                index=1
            )


        fit_v0 = st.checkbox(
            "同時擬合初速度 v₀",
            value=False
        )


        st.caption(
            "目前自動擬合 Cd₀、vc、n∞；"
            "過渡參數 p 使用左側設定值。"
        )


        # ====================================================
        # 開始擬合
        # ====================================================

        if st.button(
            "🔬 開始擬合實驗資料",
            type="primary"
        ):
            if rho <= 0 or A <= 0:

                st.error(
                "無法擬合：空氣密度 ρ 和迎風截面積 A "
                "必須大於 0，否則阻力會消失，"
                "無法辨識阻力係數與冪次參數。"
            )

                st.stop()
            # -----------------------------------------------
            # 清理資料
            # -----------------------------------------------

            if time_column == height_column:
                st.error("時間欄與高度欄必須不同。")
                st.stop()
            data = df[[time_column, height_column]].copy()
            data[time_column] = pd.to_numeric(data[time_column], errors="coerce")
            data[height_column] = pd.to_numeric(data[height_column], errors="coerce")
            data = (
                data.replace([np.inf, -np.inf], np.nan)
                .dropna()
                .sort_values(time_column)
                .groupby(time_column, as_index=False)[height_column].mean()
            )


            time_data = (
                data[time_column]
                .to_numpy(dtype=float)
            )


            height_data = (
                data[height_column]
                .to_numpy(dtype=float)
            )


            if len(time_data) < 3:

                st.error(
                    "有效資料點太少，至少需要 3 個資料點。"
                )

                st.stop()


            # 第一筆時間定義為 t = 0
            time_data = (
                time_data
                - time_data[0]
            )


            y0_fit = float(
                height_data[0]
            )


            # -----------------------------------------------
            # 如果資料很多，先抽樣以提高最佳化速度
            # -----------------------------------------------

            if len(time_data) > 250:

                indices = np.linspace(
                    0,
                    len(time_data) - 1,
                    250
                ).astype(int)


                t_fit = time_data[indices]

                y_fit = height_data[indices]

            else:

                t_fit = time_data

                y_fit = height_data


            # =================================================
            # 給定一組參數，計算模型軌跡
            # =================================================

            def simulate_fit(
                Cd_test,
                vc_test,
                n_test,
                v0_test
            ):

                def equations_fit(t, state):
                    return motion_rhs(
                        t, state, m, g,
                        lambda v: variable_drag(v, rho, Cd_test, A, vc_test, p, n_test),
                        use_sr,
                    )


                solution_fit = solve_ivp(
                    equations_fit,
                    [
                        0,
                        t_fit[-1]
                    ],
                    initial_state(y0_fit, v0_test, use_sr),
                    t_eval=t_fit,
                    rtol=1e-6,
                    atol=1e-8
                )


                if not solution_fit.success:

                    return None


                if len(solution_fit.t) != len(t_fit):

                    return None


                return solution_fit.y[0]


            # =================================================
            # 殘差函式
            # =================================================

            def residual(parameters):

                if fit_v0:

                    (
                        Cd_test,
                        vc_test,
                        n_test,
                        v0_test
                    ) = parameters

                else:

                    (
                        Cd_test,
                        vc_test,
                        n_test
                    ) = parameters


                    v0_test = v0


                prediction = simulate_fit(
                    Cd_test,
                    vc_test,
                    n_test,
                    v0_test
                )


                if prediction is None:

                    return np.full(
                        len(y_fit),
                        1e6
                    )


                return (
                    prediction
                    - y_fit
                )


            # =================================================
            # 初始猜測與參數範圍
            # =================================================

            if use_sr and abs(v0) >= C_LIGHT:
                st.error("SR 擬合模式需要 |v₀| < c。")
                st.stop()

            if fit_v0:

                x0 = [
                    max(Cd, 0.1),
                    max(vc, 1.0),
                    n_high,
                    v0
                ]


                lower = [
                    1e-4,
                    0.1,
                    1.0,
                    -200.0
                ]


                upper = [
                    5.0,
                    300.0,
                    2.0,
                    200.0
                ]

            else:

                x0 = [
                    max(Cd, 0.1),
                    max(vc, 1.0),
                    n_high
                ]


                lower = [
                    1e-4,
                    0.1,
                    1.0
                ]


                upper = [
                    5.0,
                    300.0,
                    2.0
                ]


            # =================================================
            # 最佳化
            # =================================================

            with st.spinner(
                "正在進行數值模擬與參數最佳化..."
            ):

                fit = least_squares(
                    residual,
                    x0,
                    bounds=(
                        lower,
                        upper
                    ),
                    loss="soft_l1",
                    max_nfev=100
                )


            # =================================================
            # 取得最佳參數
            # =================================================

            if fit_v0:

                (
                    Cd_fit,
                    vc_fit,
                    n_fit,
                    v0_fit
                ) = fit.x

            else:

                (
                    Cd_fit,
                    vc_fit,
                    n_fit
                ) = fit.x


                v0_fit = v0


            # =================================================
            # 使用完整資料重新模擬
            # =================================================

            def equations_final(t, state):
                return motion_rhs(
                    t, state, m, g,
                    lambda v: variable_drag(v, rho, Cd_fit, A, vc_fit, p, n_fit),
                    use_sr,
                )


            final_solution = solve_ivp(
                equations_final,
                [
                    0,
                    time_data[-1]
                ],
                initial_state(y0_fit, v0_fit, use_sr),
                t_eval=time_data,
                rtol=1e-7,
                atol=1e-9
            )


            if not final_solution.success:

                st.error(
                    "最佳參數已找到，但完整資料重新模擬失敗。"
                )

                st.stop()


            y_prediction = (
                final_solution.y[0]
            )


            # =================================================
            # 誤差分析
            # =================================================

            residuals = (
                height_data
                - y_prediction
            )


            rmse = np.sqrt(
                np.mean(
                    residuals**2
                )
            )


            ss_res = np.sum(
                residuals**2
            )


            ss_tot = np.sum(
                (
                    height_data
                    - np.mean(height_data)
                )**2
            )


            if ss_tot > 0:

                r2 = (
                    1
                    - ss_res / ss_tot
                )

            else:

                r2 = np.nan


            # =================================================
            # 顯示結果
            # =================================================

            if not fit.success:
                st.warning("最佳化尚未正常收斂：" + str(fit.message))
            st.success("✅ 擬合程序完成（請檢查殘差與參數可信度）")
            st.caption(
                "擬合框架：" + dynamics_mode + "。p 固定為側欄數值；"
                "高速下原本經驗 Cd 模型不保證有效，R²≈1 不代表唯一的物理機制。"
            )


            c1, c2, c3, c4 = st.columns(4)


            c1.metric(
                "基準阻力係數 Cd₀",
                f"{Cd_fit:.4f}"
            )


            c2.metric(
                "特徵速度 vc",
                f"{vc_fit:.3f} m/s"
            )


            c3.metric(
                "高速極限指數 n∞",
                f"{n_fit:.4f}"
            )


            c4.metric(
                "均方根誤差 RMSE",
                f"{rmse:.5g} m"
            )


            if fit_v0:

                st.metric(
                    "擬合初速度 v₀",
                    f"{v0_fit:.4f} m/s"
                )


            st.metric(
                "判定係數 R²",
                f"{r2:.6f}"
            )


            # =================================================
            # 實驗 vs 模型
            # =================================================

            st.subheader(
                "實驗資料與擬合模型比較"
            )


            fig_fit = go.Figure()


            fig_fit.add_trace(
                go.Scatter(
                    x=time_data,
                    y=height_data,
                    mode="markers",
                    name="實驗資料"
                )
            )


            fig_fit.add_trace(
                go.Scatter(
                    x=time_data,
                    y=y_prediction,
                    mode="lines",
                    name="模型擬合曲線"
                )
            )


            fig_fit.update_layout(
                title="實驗資料與模型擬合比較",
                xaxis_title="時間 (s)",
                yaxis_title="高度 (m)",
             hovermode="x unified"
            )


            show_plot(fig_fit)


            # =================================================
            # 殘差圖
            # =================================================

            st.subheader(
                "殘差分析"
            )


            fig_residual = go.Figure()


            fig_residual.add_trace(
                go.Scatter(
                    x=time_data,
                    y=residuals,
                    mode="markers+lines",
                    name="Residual"
                )
            )


            fig_residual.add_hline(
                y=0,
                line_dash="dash"
            )


            fig_residual.update_layout(
                xaxis_title="時間 (s)",
                yaxis_title="Residual (m)",
                title="Residual vs 時間"
            )


            show_plot(fig_residual)


            # =================================================
            # 下載結果
            # =================================================

            output_df = pd.DataFrame(
                {
                    "time": time_data,
                    "height_experiment": height_data,
                    "height_model": y_prediction,
                    "residual": residuals
                }
            )


            csv_output = (
                output_df
                .to_csv(index=False)
                .encode("utf-8")
            )


            st.download_button(
                "⬇️ 下載擬合結果 CSV",
                data=csv_output,
                file_name="air_drag_fit_result.csv",
                mime="text/csv"
            )


    except Exception as error:

        st.error(
            f"讀取或擬合資料時發生錯誤：{error}"
        )
