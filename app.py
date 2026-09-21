import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import solve_ivp


# ============================================================
# 1. 網頁基本設定
# ============================================================

st.set_page_config(
    page_title="空氣阻力落體運動模擬",
    page_icon="📉",
    layout="wide"
)

st.title("空氣阻力下的落體運動模擬")

st.write(
    "比較真空、線性空氣阻力與平方空氣阻力下的落體運動，"
    "並分析高度、速度、加速度與空氣阻力隨時間的變化。"
)

st.info(
    "座標方向設定：向上為正方向，因此向下運動時速度為負值。"
)


# ============================================================
# 2. Plotly 圖表設定
# ============================================================

PLOT_CONFIG = {
    "scrollZoom": True,       # 滑鼠滾輪縮放
    "displaylogo": False,
    "responsive": True
}


# ============================================================
# 3. 物體預設值
# ============================================================

OBJECT_PRESETS = {

    "自訂": None,

    "棒球（近似）": {
        "m_input": 0.145,
        "Cd_input": 0.47,
        "A_input": 0.0042
    },

    "羽毛（示意）": {
        "m_input": 0.003,
        "Cd_input": 1.20,
        "A_input": 0.010
    },

    "雨滴（直徑約 2 mm，示意）": {
        "m_input": 4.2e-6,
        "Cd_input": 0.60,
        "A_input": 3.14e-6
    }
}


# ============================================================
# 4. Session State 初始值
# ============================================================

DEFAULTS = {

    "m_input": 1.0,

    "Cd_input": 0.47,

    "A_input": 0.10,

    "y0_input": 100.0,

    "v0_input": 0.0,

    "g_input": 9.81,

    "b_input": 0.5,

    "rho_input": 1.225
}


for key, value in DEFAULTS.items():

    if key not in st.session_state:

        st.session_state[key] = value


# 記錄是否至少執行過一次模擬
if "has_run" not in st.session_state:

    st.session_state.has_run = False


# ============================================================
# 5. 套用物體預設值
# ============================================================

def apply_object_preset():

    name = st.session_state.object_preset

    preset = OBJECT_PRESETS[name]

    if preset is not None:

        for key, value in preset.items():

            st.session_state[key] = value


# ============================================================
# 6. 物理模型說明
# ============================================================

with st.expander("📘 查看物理模型與參數說明"):

    st.markdown(
        r"""
### 座標設定

本模擬定義：

- 向上為正方向
- 向下為負方向
- 地面為 $y=0$

因此重力加速度為

$$
a_g=-g
$$

---

### 真空模型

真空中只有重力作用：

$$
m\frac{dv}{dt}=-mg
$$

因此：

$$
\frac{dv}{dt}=-g
$$

位置與速度的解析解為：

$$
y(t)=y_0+v_0t-\frac{1}{2}gt^2
$$

$$
v(t)=v_0-gt
$$

---

### 線性空氣阻力

假設空氣阻力與速度成正比：

$$
F_D=-bv
$$

運動方程：

$$
m\frac{dv}{dt}=-mg-bv
$$

因此：

$$
\frac{dv}{dt}
=
-g-\frac{b}{m}v
$$

線性模型的終端速度：

$$
v_T=-\frac{mg}{b}
$$

---

### 平方空氣阻力

一般物體在較高 Reynolds number 的情況下，
常使用平方阻力模型：

$$
F_D
=
-\frac{1}{2}\rho C_D A v|v|
$$

運動方程：

$$
m\frac{dv}{dt}
=
-mg+F_D
$$

因此：

$$
\frac{dv}{dt}
=
-g-\frac{\rho C_D A}{2m}v|v|
$$

平方阻力模型的終端速度：

$$
v_T
=
-\sqrt{
\frac{2mg}
{\rho C_D A}
}
$$

---

### 主要參數

- $y_0$：初始高度，單位 m
- $v_0$：初速度，單位 m/s
- $m$：物體質量，單位 kg
- $g$：重力加速度，單位 m/s²
- $b$：線性空氣阻力係數，單位 kg/s
- $\rho$：空氣密度，單位 kg/m³
- $C_D$：阻力係數
- $A$：迎風截面積，單位 m²

---

### 速度正負號

本程式定義：

$$
v>0
$$

代表物體向上運動，

而

$$
v<0
$$

代表物體向下運動。

空氣阻力始終與物體運動方向相反。
        """
    )


# ============================================================
# 7. Sidebar：模擬設定
# ============================================================

st.sidebar.header("模擬設定")


# ------------------------------------------------------------
# 模型
# ------------------------------------------------------------

model = st.sidebar.selectbox(
    "選擇物理模型",
    [
        "真空",
        "線性空氣阻力",
        "平方空氣阻力"
    ]
)


compare_all = st.sidebar.checkbox(
    "同時比較三種模型",
    value=False
)


# ============================================================
# 8. 物體預設
# ============================================================

st.sidebar.subheader("物體")

st.sidebar.selectbox(
    "物體預設",
    list(OBJECT_PRESETS.keys()),
    key="object_preset",
    on_change=apply_object_preset
)

st.sidebar.caption(
    "物體參數僅作為近似或示意值，選擇後仍可自行修改。"
)


# ============================================================
# 9. 圖表縮放設定
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
        "顯示起始時間 (s)",
        min_value=0.0,
        value=0.0,
        step=1.0
    )

    custom_t_max = st.sidebar.number_input(
        "顯示結束時間 (s)",
        min_value=0.1,
        value=10.0,
        step=1.0
    )


st.sidebar.caption(
    "圖表也支援滑鼠滾輪縮放、拖曳與 Plotly 工具列。"
)


# ============================================================
# 10. 輸入參數表單
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
        "v₀ > 0：向上拋；v₀ < 0：向下拋"
    )


    # --------------------------------------------------------
    # 基本參數
    # --------------------------------------------------------

    st.subheader("基本物理參數")


    m = st.number_input(
        "質量 m (kg)",
        min_value=1e-8,
        format="%.6g",
        key="m_input"
    )


    g = st.number_input(
        "重力加速度 g (m/s²)",
        min_value=0.001,
        format="%.4f",
        key="g_input"
    )


    # --------------------------------------------------------
    # 線性阻力
    # --------------------------------------------------------

    if model == "線性空氣阻力" or compare_all:

        st.subheader("線性阻力參數")


        b = st.number_input(
            "線性阻力係數 b (kg/s)",
            min_value=0.0,
            format="%.6g",
            key="b_input"
        )

    else:

        b = st.session_state.b_input


    # --------------------------------------------------------
    # 平方阻力
    # --------------------------------------------------------

    if model == "平方空氣阻力" or compare_all:

        st.subheader("平方阻力參數")


        rho = st.number_input(
            "空氣密度 ρ (kg/m³)",
            min_value=0.0,
            format="%.6g",
            key="rho_input"
        )


        Cd = st.number_input(
            "阻力係數 Cd",
            min_value=0.0,
            format="%.6g",
            key="Cd_input"
        )


        A = st.number_input(
            "截面積 A (m²)",
            min_value=0.0,
            format="%.6g",
            key="A_input"
        )

    else:

        rho = st.session_state.rho_input

        Cd = st.session_state.Cd_input

        A = st.session_state.A_input


    submitted = st.form_submit_button(
        "🚀 開始模擬",
        type="primary"
    )


# 按過開始後就記住
if submitted:

    st.session_state.has_run = True


# ============================================================
# 11. 真空模型
# ============================================================

def solve_vacuum(
    y0,
    v0,
    g
):


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


    a = np.full_like(
        t,
        -g
    )


    F_drag = np.zeros_like(t)


    return {

        "t": t,

        "y": y,

        "v": v,

        "a": a,

        "F_drag": F_drag,

        "t_hit": t_hit,

        "v_hit": v[-1],

        "v_terminal": None
    }


# ============================================================
# 12. 線性阻力模型
# ============================================================

def solve_linear(
    y0,
    v0,
    m,
    g,
    b
):


    def equations(t, state):

        y = state[0]

        v = state[1]


        dydt = v


        dvdt = (
            -g
            -
            (b / m) * v
        )


        return [
            dydt,
            dvdt
        ]


    def hit_ground(t, state):

        return state[0]


    hit_ground.direction = -1

    hit_ground.terminal = True


    solution = solve_ivp(
        equations,
        [0, 1000],
        [y0, v0],
        events=hit_ground,
        max_step=0.05,
        rtol=1e-8,
        atol=1e-10
    )


    t = solution.t

    y = solution.y[0]

    v = solution.y[1]


    F_drag = -b * v


    a = (
        -g
        +
        F_drag / m
    )


    if len(solution.t_events[0]) > 0:

        t_hit = solution.t_events[0][0]

        v_hit = solution.y_events[0][0][1]

    else:

        t_hit = None

        v_hit = None


    if b > 0:

        v_terminal = (
            -m * g / b
        )

    else:

        v_terminal = None


    return {

        "t": t,

        "y": y,

        "v": v,

        "a": a,

        "F_drag": F_drag,

        "t_hit": t_hit,

        "v_hit": v_hit,

        "v_terminal": v_terminal
    }


# ============================================================
# 13. 平方阻力模型
# ============================================================

def solve_quadratic(
    y0,
    v0,
    m,
    g,
    rho,
    Cd,
    A
):


    def equations(t, state):

        y = state[0]

        v = state[1]


        F_drag = (
            -0.5
            *
            rho
            *
            Cd
            *
            A
            *
            v
            *
            abs(v)
        )


        dydt = v


        dvdt = (
            -g
            +
            F_drag / m
        )


        return [
            dydt,
            dvdt
        ]


    def hit_ground(t, state):

        return state[0]


    hit_ground.direction = -1

    hit_ground.terminal = True


    solution = solve_ivp(
        equations,
        [0, 1000],
        [y0, v0],
        events=hit_ground,
        max_step=0.05,
        rtol=1e-8,
        atol=1e-10
    )


    t = solution.t

    y = solution.y[0]

    v = solution.y[1]


    F_drag = (
        -0.5
        *
        rho
        *
        Cd
        *
        A
        *
        v
        *
        np.abs(v)
    )


    a = (
        -g
        +
        F_drag / m
    )


    if len(solution.t_events[0]) > 0:

        t_hit = solution.t_events[0][0]

        v_hit = solution.y_events[0][0][1]

    else:

        t_hit = None

        v_hit = None


    if (
        rho > 0
        and Cd > 0
        and A > 0
    ):

        v_terminal = -np.sqrt(
            2 * m * g
            /
            (
                rho
                *
                Cd
                *
                A
            )
        )

    else:

        v_terminal = None


    return {

        "t": t,

        "y": y,

        "v": v,

        "a": a,

        "F_drag": F_drag,

        "t_hit": t_hit,

        "v_hit": v_hit,

        "v_terminal": v_terminal
    }


# ============================================================
# 14. 圖表時間軸縮放函式
# ============================================================

def apply_zoom(
    fig,
    zoom_mode,
    all_times,
    custom_t_min=0.0,
    custom_t_max=10.0
):


    # --------------------------------------------------------
    # 完整範圍
    # --------------------------------------------------------

    if zoom_mode == "完整時間範圍":

        fig.update_xaxes(
            autorange=True
        )


    # --------------------------------------------------------
    # 聚焦區間
    # --------------------------------------------------------

    elif zoom_mode == "聚焦主要區間":

        valid_times = [
            time_array
            for time_array in all_times
            if (
                time_array is not None
                and len(time_array) > 0
            )
        ]


        if valid_times:

            end_times = [
                time_array[-1]
                for time_array in valid_times
            ]


            # 最短模擬時間
            focus_time = min(end_times)


            # 多保留 20 %
            focus_time = focus_time * 1.2


            fig.update_xaxes(
                range=[
                    0,
                    focus_time
                ]
            )


    # --------------------------------------------------------
    # 自訂
    # --------------------------------------------------------

    elif zoom_mode == "自訂時間範圍":

        if custom_t_max > custom_t_min:

            fig.update_xaxes(
                range=[
                    custom_t_min,
                    custom_t_max
                ]
            )


    return fig


# ============================================================
# 15. 建立單一圖表
# ============================================================

def single_plot(
    x,
    y,
    title,
    y_title,
    name
):


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name
        )
    )


    fig.update_layout(

        title=title,

        xaxis_title="Time (s)",

        yaxis_title=y_title,

        hovermode="x unified",

        dragmode="zoom"
    )


    return fig


# ============================================================
# 16. 顯示 Plotly 圖
# ============================================================

def show_plot(fig):

    st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOT_CONFIG
    )


# ============================================================
# 17. 模擬開始
# ============================================================

if st.session_state.has_run:


    # ========================================================
    # 防呆提醒
    # ========================================================

    if (
        model == "平方空氣阻力"
        or compare_all
    ):


        if rho == 0:

            st.warning(
                "⚠️ 目前空氣密度 ρ = 0，因此平方空氣阻力為 0。"
                "Quadratic Drag 將與 Vacuum 完全相同。"
            )


        if Cd == 0:

            st.warning(
                "⚠️ 目前 Cd = 0，因此平方空氣阻力為 0。"
                "Quadratic Drag 將與 Vacuum 完全相同。"
            )


        if A == 0:

            st.warning(
                "⚠️ 目前截面積 A = 0，因此平方空氣阻力為 0。"
                "Quadratic Drag 將與 Vacuum 完全相同。"
            )


    # --------------------------------------------------------
    # 線性阻力提示
    # --------------------------------------------------------

    if (
        model == "線性空氣阻力"
        or compare_all
    ):


        if b == 0:

            st.warning(
                "⚠️ 目前 b = 0，因此線性空氣阻力為 0。"
                "Linear Drag 將與 Vacuum 相同。"
            )


        elif b > 0:

            terminal_speed_linear = (
                m * g / b
            )


            if terminal_speed_linear < 5:

                st.warning(
                    f"⚠️ 目前線性阻力較強，"
                    f"終端速率只有約 "
                    f"{terminal_speed_linear:.2f} m/s。"
                    "因此從較高的位置落下時，"
                    "落地時間可能非常長。"
                )


    # ========================================================
    # 18. 單一模型模式
    # ========================================================

    if not compare_all:


        # ----------------------------------------------------
        # 求解
        # ----------------------------------------------------

        if model == "真空":

            result = solve_vacuum(
                y0,
                v0,
                g
            )


        elif model == "線性空氣阻力":

            result = solve_linear(
                y0,
                v0,
                m,
                g,
                b
            )


        else:

            result = solve_quadratic(
                y0,
                v0,
                m,
                g,
                rho,
                Cd,
                A
            )


        st.header(
            f"模擬結果：{model}"
        )


        # ====================================================
        # 數值結果
        # ====================================================

        col1, col2, col3 = st.columns(3)


        if result["t_hit"] is not None:

            col1.metric(
                "落地時間",
                f'{result["t_hit"]:.4f} s'
            )


            col2.metric(
                "落地速度",
                f'{result["v_hit"]:.4f} m/s'
            )


        else:

            col1.metric(
                "落地時間",
                "模擬時間內未落地"
            )


            col2.metric(
                "落地速度",
                "-"
            )


        if result["v_terminal"] is not None:

            col3.metric(
                "理論終端速度",
                f'{result["v_terminal"]:.4f} m/s'
            )

        else:

            col3.metric(
                "理論終端速度",
                "無"
            )


        # ====================================================
        # 高度圖
        # ====================================================

        st.subheader("高度－時間")


        fig = single_plot(
            result["t"],
            result["y"],
            "Height vs Time",
            "Height (m)",
            model
        )


        fig = apply_zoom(
            fig,
            zoom_mode,
            [result["t"]],
            custom_t_min,
            custom_t_max
        )


        show_plot(fig)


        # ====================================================
        # 速度圖
        # ====================================================

        st.subheader("速度－時間")


        fig = single_plot(
            result["t"],
            result["v"],
            "Velocity vs Time",
            "Velocity (m/s)",
            model
        )


        if result["v_terminal"] is not None:

            fig.add_hline(
                y=result["v_terminal"],
                line_dash="dash",
                annotation_text="Terminal velocity"
            )


        fig = apply_zoom(
            fig,
            zoom_mode,
            [result["t"]],
            custom_t_min,
            custom_t_max
        )


        show_plot(fig)


        # ====================================================
        # 加速度圖
        # ====================================================

        st.subheader("加速度－時間")


        fig = single_plot(
            result["t"],
            result["a"],
            "Acceleration vs Time",
            "Acceleration (m/s²)",
            model
        )


        fig = apply_zoom(
            fig,
            zoom_mode,
            [result["t"]],
            custom_t_min,
            custom_t_max
        )


        show_plot(fig)


        # ====================================================
        # 阻力圖
        # ====================================================

        st.subheader("空氣阻力－時間")


        fig = single_plot(
            result["t"],
            result["F_drag"],
            "Drag Force vs Time",
            "Drag Force (N)",
            model
        )


        fig = apply_zoom(
            fig,
            zoom_mode,
            [result["t"]],
            custom_t_min,
            custom_t_max
        )


        show_plot(fig)


    # ========================================================
    # 19. 三模型比較
    # ========================================================

    else:


        vacuum = solve_vacuum(
            y0,
            v0,
            g
        )


        linear = solve_linear(
            y0,
            v0,
            m,
            g,
            b
        )


        quadratic = solve_quadratic(
            y0,
            v0,
            m,
            g,
            rho,
            Cd,
            A
        )


        st.header(
            "真空、線性阻力、平方阻力比較"
        )


        # ====================================================
        # 數值結果
        # ====================================================

        col1, col2, col3 = st.columns(3)


        # ----------------------------------------------------
        # 真空
        # ----------------------------------------------------

        with col1:

            st.subheader("真空")


            st.metric(
                "落地時間",
                f'{vacuum["t_hit"]:.4f} s'
            )


            st.metric(
                "落地速度",
                f'{vacuum["v_hit"]:.4f} m/s'
            )


        # ----------------------------------------------------
        # 線性
        # ----------------------------------------------------

        with col2:

            st.subheader("線性阻力")


            if linear["t_hit"] is not None:

                st.metric(
                    "落地時間",
                    f'{linear["t_hit"]:.4f} s'
                )


                st.metric(
                    "落地速度",
                    f'{linear["v_hit"]:.4f} m/s'
                )


            else:

                st.metric(
                    "落地時間",
                    "模擬時間內未落地"
                )


                st.metric(
                    "落地速度",
                    "-"
                )


            if linear["v_terminal"] is not None:

                st.metric(
                    "終端速度",
                    f'{linear["v_terminal"]:.4f} m/s'
                )


        # ----------------------------------------------------
        # 平方
        # ----------------------------------------------------

        with col3:

            st.subheader("平方阻力")


            if quadratic["t_hit"] is not None:

                st.metric(
                    "落地時間",
                    f'{quadratic["t_hit"]:.4f} s'
                )


                st.metric(
                    "落地速度",
                    f'{quadratic["v_hit"]:.4f} m/s'
                )


            else:

                st.metric(
                    "落地時間",
                    "模擬時間內未落地"
                )


                st.metric(
                    "落地速度",
                    "-"
                )


            if quadratic["v_terminal"] is not None:

                st.metric(
                    "終端速度",
                    f'{quadratic["v_terminal"]:.4f} m/s'
                )


        # ====================================================
        # 共用時間陣列
        # ====================================================

        comparison_times = [

            vacuum["t"],

            linear["t"],

            quadratic["t"]
        ]


        # ====================================================
        # 高度比較
        # ====================================================

        st.subheader("高度比較")


        fig_height = go.Figure()


        fig_height.add_trace(
            go.Scatter(
                x=vacuum["t"],
                y=vacuum["y"],
                mode="lines",
                name="Vacuum"
            )
        )


        fig_height.add_trace(
            go.Scatter(
                x=linear["t"],
                y=linear["y"],
                mode="lines",
                name="Linear Drag"
            )
        )


        fig_height.add_trace(
            go.Scatter(
                x=quadratic["t"],
                y=quadratic["y"],
                mode="lines",
                name="Quadratic Drag"
            )
        )


        fig_height.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Height (m)",
            hovermode="x unified",
            dragmode="zoom"
        )


        fig_height = apply_zoom(
            fig_height,
            zoom_mode,
            comparison_times,
            custom_t_min,
            custom_t_max
        )


        show_plot(fig_height)


        # ====================================================
        # 速度比較
        # ====================================================

        st.subheader("速度比較")


        fig_velocity = go.Figure()


        fig_velocity.add_trace(
            go.Scatter(
                x=vacuum["t"],
                y=vacuum["v"],
                mode="lines",
                name="Vacuum"
            )
        )


        fig_velocity.add_trace(
            go.Scatter(
                x=linear["t"],
                y=linear["v"],
                mode="lines",
                name="Linear Drag"
            )
        )


        fig_velocity.add_trace(
            go.Scatter(
                x=quadratic["t"],
                y=quadratic["v"],
                mode="lines",
                name="Quadratic Drag"
            )
        )


        fig_velocity.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Velocity (m/s)",
            hovermode="x unified",
            dragmode="zoom"
        )


        fig_velocity = apply_zoom(
            fig_velocity,
            zoom_mode,
            comparison_times,
            custom_t_min,
            custom_t_max
        )


        show_plot(fig_velocity)


        # ====================================================
        # 加速度比較
        # ====================================================

        st.subheader("加速度比較")


        fig_acceleration = go.Figure()


        fig_acceleration.add_trace(
            go.Scatter(
                x=vacuum["t"],
                y=vacuum["a"],
                mode="lines",
                name="Vacuum"
            )
        )


        fig_acceleration.add_trace(
            go.Scatter(
                x=linear["t"],
                y=linear["a"],
                mode="lines",
                name="Linear Drag"
            )
        )


        fig_acceleration.add_trace(
            go.Scatter(
                x=quadratic["t"],
                y=quadratic["a"],
                mode="lines",
                name="Quadratic Drag"
            )
        )


        fig_acceleration.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Acceleration (m/s²)",
            hovermode="x unified",
            dragmode="zoom"
        )


        fig_acceleration = apply_zoom(
            fig_acceleration,
            zoom_mode,
            comparison_times,
            custom_t_min,
            custom_t_max
        )


        show_plot(fig_acceleration)


        # ====================================================
        # 空氣阻力比較
        # ====================================================

        st.subheader("空氣阻力比較")


        fig_drag = go.Figure()


        fig_drag.add_trace(
            go.Scatter(
                x=vacuum["t"],
                y=vacuum["F_drag"],
                mode="lines",
                name="Vacuum"
            )
        )


        fig_drag.add_trace(
            go.Scatter(
                x=linear["t"],
                y=linear["F_drag"],
                mode="lines",
                name="Linear Drag"
            )
        )


        fig_drag.add_trace(
            go.Scatter(
                x=quadratic["t"],
                y=quadratic["F_drag"],
                mode="lines",
                name="Quadratic Drag"
            )
        )


        fig_drag.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="Drag Force (N)",
            hovermode="x unified",
            dragmode="zoom"
        )


        fig_drag = apply_zoom(
            fig_drag,
            zoom_mode,
            comparison_times,
            custom_t_min,
            custom_t_max
        )


        show_plot(fig_drag)


# ============================================================
# 20. 尚未開始模擬
# ============================================================

else:

    st.subheader(
        "設定左側參數後，按下「🚀 開始模擬」"
    )

    st.write(
        "你可以選擇物體、物理模型以及初始條件，"
        "再觀察高度、速度、加速度與空氣阻力的變化。"
    )