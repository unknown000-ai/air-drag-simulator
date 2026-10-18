"""自由落體研究平台：基礎模擬、資料辨識與狹義相對論動量比較。

本版本由先前 air_drag_app_test_fixed 系列整理而來。請使用：
python -m streamlit run air_drag_project_clear.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares
from scipy.signal import savgol_filter

光速 = 299_792_458.0
模型名稱 = {"vacuum": "真空（無空氣阻力）", "linear": "線性阻力", "quadratic": "平方阻力", "variable": "可變冪次（進階候選模型）"}
圖表設定 = {"displaylogo": False, "scrollZoom": True, "responsive": True}


# ────────────────────────── 純物理計算 ──────────────────────────
def 洛倫茲因子(v):
    v = np.asarray(v, dtype=float)
    if np.any(np.abs(v) >= 光速):
        raise ValueError("相對論速度必須小於光速。")
    return 1.0 / np.sqrt(1.0 - (v / 光速) ** 2)


def 動量換速度(u):
    # u = gamma*v；使用 hypot 可避免高速計算溢位或數值速度超光速
    u = np.asarray(u, dtype=float)
    return u / np.hypot(1.0, u / 光速)


def 加速度修正比例(v):
    """SR 與牛頓在『同一瞬時速度、同一實驗室淨力』下的加速度比值。"""
    b2 = (np.asarray(v, dtype=float) / 光速) ** 2
    return (1 - b2) ** 1.5


def 阻力(v, kind, par):
    """空氣相對實驗室靜止時的 *低速* 經驗阻力，向上為正。"""
    v = np.asarray(v)
    s = np.abs(v)
    if kind == "vacuum":
        out = np.zeros_like(v, dtype=float)
    elif kind == "linear":
        out = -par["b"] * v
    elif kind == "quadratic":
        out = -0.5 * par["rho"] * par["cd"] * par["area"] * v * s
    elif kind == "variable":
        z = np.power(s / par["vc"], par["shape"])
        correction = np.power(1 + z, -(2 - par["n_high"]) / par["shape"])
        out = -0.5 * par["rho"] * par["cd"] * par["area"] * v * s * correction
    else:
        raise ValueError("未知阻力模型")
    return float(out) if out.ndim == 0 else out


def 理論有效冪次(speed, par):
    z = (np.asarray(speed) / par["vc"]) ** par["shape"]
    return 2 - (2 - par["n_high"]) * z / (1 + z)


def 終端速度(m, g, kind, par):
    if kind == "vacuum":
        return None
    if kind == "linear":
        return -m * g / par["b"] if par["b"] > 0 else None
    if kind == "quadratic":
        k = 0.5 * par["rho"] * par["cd"] * par["area"]
        return -np.sqrt(m * g / k) if k > 0 else None
    if par["rho"] * par["cd"] * par["area"] <= 0:
        return None
    def f(s):
        return -阻力(-s, kind, par) - m * g
    high = 1.0
    for _ in range(90):
        if f(high) > 0:
            return -brentq(f, 0.0, high)
        high *= 2
    return None


def 求解運動(y0, v0, m, g, kind, par, 相對論=False, t_limit=120.0,
         points=400, rtol=2e-9, atol=1e-10):
    """牛頓積分 v 或 SR 積分 u=gamma*v，重力採用固定外力 -mg。"""
    if not (np.isfinite(y0) and np.isfinite(v0) and y0 > 0 and m > 0 and g > 0 and t_limit > 0):
        raise ValueError("高度、質量、重力、模擬時間必須是有效正數。")
    if 相對論 and abs(v0) >= 光速:
        raise ValueError("相對論模型的初速度必須小於光速。")
    if kind != "vacuum" and abs(v0) > 1500:
        raise ValueError("一般空氣阻力經驗式不可直接用在超過 1500 m/s 的初速度。請改用真空的相對論理想測試頁，或建立專門的可壓縮流體模型。")
    if 相對論:
        u0 = float(洛倫茲因子(v0) * v0)
    else:
        u0 = v0

    def eq(t, state):
        velocity = float(動量換速度(state[1])) if 相對論 else state[1]
        return (velocity, -g + 阻力(velocity, kind, par) / m)

    def landed(t, state):
        return state[0]
    landed.terminal = True
    landed.direction = -1

    solution = solve_ivp(eq, [0.0, t_limit], [y0, u0], method="RK45",
                         dense_output=True, events=landed, rtol=rtol, atol=atol)
    if not solution.success:
        raise RuntimeError("微分方程求解失敗：" + solution.message)
    hit = float(solution.t_events[0][0]) if len(solution.t_events[0]) else None
    end = hit if hit is not None else float(t_limit)
    t = np.linspace(0, end, points)
    yy, second = solution.sol(t)
    vv = 動量換速度(second) if 相對論 else second
    drag = 阻力(vv, kind, par)
    a = (-g + drag / m) * (加速度修正比例(vv) if 相對論 else 1.0)
    return {"t": t, "y": yy, "v": vv, "a": a, "fd": drag,
            "hit": hit, "v_hit": float(vv[-1]) if hit is not None else None,
            "terminal": 終端速度(m, g, kind, par), "max_beta": float(np.max(np.abs(vv)) / 光速)}


def 擬合軌跡(t, y0, v0, m, g, kind, par, 相對論=False, rtol=2e-7, atol=1e-9):
    """同資料時間座標的預測位置，供最小平方法使用。"""
    t = np.asarray(t, dtype=float)
    if len(t) < 2 or (abs(v0) >= 光速 and 相對論) or (kind != "vacuum" and abs(v0) > 1500):
        return None
    u0 = float(洛倫茲因子(v0) * v0) if 相對論 else v0
    def eq(_t, state):
        v = float(動量換速度(state[1])) if 相對論 else state[1]
        return (v, -g + 阻力(v, kind, par) / m)
    try:
        result = solve_ivp(eq, (0.0, float(t[-1])), [y0, u0],
                           t_eval=t, rtol=rtol, atol=atol)
        if not result.success or result.y.shape[1] != len(t):
            return None
        return result.y[0]
    except (ValueError, OverflowError, FloatingPointError):
        return None


# ────────────────────────── 資料分析 ──────────────────────────
def 整理資料(df, time_col, height_col):
    data = pd.DataFrame({"time": pd.to_numeric(df[time_col], errors="coerce"),
                         "height": pd.to_numeric(df[height_col], errors="coerce")})
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    data = data.groupby("time", as_index=False)["height"].mean().sort_values("time")
    if len(data) < 9:
        raise ValueError("有效時間－高度資料至少需要 9 筆。")
    t = data["time"].to_numpy(dtype=float)
    if t[-1] <= t[0]:
        raise ValueError("時間範圍需要大於零。")
    data["time"] = t - t[0]
    return data


def odd_window(requested, available, polyorder):
    x = min(int(requested), available if available % 2 else available - 1)
    x = max(x, polyorder + 2)
    if x % 2 == 0:
        x += 1
    return min(x, available if available % 2 else available - 1)


def 局部冪次(t, v, f, window, min_speed, min_force):
    speed = np.abs(v)
    valid = (np.isfinite(v) & np.isfinite(f) & (speed > min_speed)
             & (np.abs(f) > min_force) & (v * f <= 0))
    half = window // 2
    output = np.full(len(t), np.nan)
    quality = np.full(len(t), np.nan)
    for k in range(half, len(t) - half):
        idx = np.arange(k - half, k + half + 1)
        idx = idx[valid[idx]]
        if len(idx) < max(5, window // 3):
            continue
        x = np.log(speed[idx])
        y = np.log(np.abs(f[idx]))
        if np.ptp(x) < 0.05:
            continue
        slope, bias = np.polyfit(x, y, 1)
        ss_total = np.sum((y - y.mean()) ** 2)
        output[k] = slope
        quality[k] = 1 - np.sum((y - (slope * x + bias)) ** 2) / ss_total if ss_total > 0 else np.nan
    return output, quality


def 辨識阻力律(data, m, g, count=420, smooth=31, poly=3, local=31,
            min_speed=0.5, min_force=0.001, 相對論=False):
    tr = data["time"].to_numpy(dtype=float)
    yr = data["height"].to_numpy(dtype=float)
    tt = np.linspace(tr[0], tr[-1], count)
    yi = np.interp(tt, tr, yr)
    smooth = odd_window(smooth, count, poly)
    local = odd_window(local, count, 2)
    dt = tt[1] - tt[0]
    yy = savgol_filter(yi, smooth, poly, deriv=0)
    vel = savgol_filter(yi, smooth, poly, deriv=1, delta=dt)
    acc = savgol_filter(yi, smooth, poly, deriv=2, delta=dt)
    if 相對論:
        gamma = 洛倫茲因子(vel)
        fd = m * (gamma ** 3 * acc + g)
    else:
        fd = m * (acc + g)
    n_eff, r2 = 局部冪次(tt, vel, fd, local, min_speed, min_force)
    return pd.DataFrame({"time": tt, "height": yy, "velocity": vel,
                         "acceleration": acc, "drag_force": fd,
                         "speed": np.abs(vel), "n_eff": n_eff, "local_R2": r2})


def 分段BIC(x, limit=4, min_length=12):
    """供進階診斷：用分段常數描述連續變化，不是宣稱存在物理相變。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * min_length:
        return pd.DataFrame(), []
    limit = min(limit, n // min_length)
    sums = np.r_[0, np.cumsum(x)]
    sq = np.r_[0, np.cumsum(x * x)]
    dp = np.full((limit + 1, n + 1), np.inf)
    back = np.full((limit + 1, n + 1), -1, dtype=int)
    dp[0, 0] = 0
    for k in range(1, limit + 1):
        for end in range(k * min_length, n + 1):
            starts = np.arange((k - 1) * min_length, end - min_length + 1)
            count = end - starts
            ss = (sq[end] - sq[starts]) - (sums[end] - sums[starts]) ** 2 / count
            scores = dp[k - 1, starts] + np.maximum(ss, 0)
            j = np.argmin(scores)
            dp[k, end] = scores[j]
            back[k, end] = starts[j]
    bics = [n * np.log(max(dp[k, n] / n, 1e-12)) + (2 * k - 1) * np.log(n)
            for k in range(1, limit + 1)]
    best_k = int(np.argmin(bics)) + 1
    ends = [n]
    end = n
    for k in range(best_k, 0, -1):
        end = int(back[k, end])
        ends.append(end)
    ends = sorted(ends)
    return pd.DataFrame({"區段數": list(range(1, limit + 1)), "BIC": bics}), [
        (ends[j], ends[j + 1], float(np.mean(x[ends[j]: ends[j + 1]])))
        for j in range(len(ends) - 1)
    ]


def 擬合模型(data, m, g, rho, area, v0, shape=4.0, fit_v0=False,
           kinds=("linear", "quadratic"), 相對論=False):
    time = data["time"].to_numpy(dtype=float)
    y = data["height"].to_numpy(dtype=float)
    sample = np.unique(np.linspace(0, len(time) - 1, min(len(time), 170)).astype(int))
    t_sub, y_sub = time[sample], y[sample]
    initial_y = float(y[0])
    results = []
    for kind in kinds:
        if kind == "linear":
            init = [0.3]
            lo, hi = [1e-7], [100.0]
            labels = ["b (kg/s)"]
        elif kind == "quadratic":
            init = [0.47]
            lo, hi = [1e-6], [10.0]
            labels = ["Cd"]
        else:
            init = [0.47, 10.0, 1.5]
            lo, hi = [1e-6, 0.1, 1.0], [10.0, 300.0, 2.0]
            labels = ["Cd₀", "vc (m/s)", "n∞"]
        if fit_v0:
            init.append(v0)
            lo.append(-min(0.99 * 光速, 3e8))
            hi.append(min(0.99 * 光速, 3e8))
            labels.append("v₀ (m/s)")

        def prepare(params):
            par = {"rho": rho, "area": area, "shape": shape, "cd": 0.47,
                   "b": 0.3, "vc": 10.0, "n_high": 1.5}
            if kind == "linear":
                par["b"] = params[0]
            elif kind == "quadratic":
                par["cd"] = params[0]
            else:
                par.update({"cd": params[0], "vc": params[1], "n_high": params[2]})
            chosen_v0 = params[-1] if fit_v0 else v0
            return par, chosen_v0

        def residual(params):
            par, vzero = prepare(params)
            pred = 擬合軌跡(t_sub, initial_y, vzero, m, g, kind, par, 相對論)
            return pred - y_sub if pred is not None else np.full(len(t_sub), 1e8)
        try:
            optimization = least_squares(residual, init, bounds=(lo, hi),
                                         max_nfev=130, loss="linear",
                                         xtol=1e-10, ftol=1e-10, gtol=1e-10)
            par, vzero = prepare(optimization.x)
            predicted = 擬合軌跡(time, initial_y, vzero, m, g, kind, par,
                              相對論, rtol=1e-8, atol=1e-10)
            if predicted is None or not optimization.success:
                results.append({"模型": 模型名稱[kind], "狀態": "未收斂"})
                continue
            error = y - predicted
            rss = float(np.sum(error ** 2))
            rmse = float(np.sqrt(rss / len(y)))
            k = len(optimization.x)
            # 僅為常態獨立等變異殘差假設下的比較指標，非物理定律判定
            bic = len(y) * np.log(max(rss / len(y), 1e-24)) + k * np.log(len(y))
            values = dict(zip(labels, optimization.x))
            results.append({"模型": 模型名稱[kind], "狀態": "成功", "RMSE (m)": rmse,
                            "BIC": bic, "參數": values, "預測": predicted, "殘差": error})
        except (ValueError, RuntimeError, FloatingPointError) as err:
            results.append({"模型": 模型名稱[kind], "狀態": f"失敗：{err}"})
    return results


# ────────────────────────── 圖表 ──────────────────────────
def 共同圖表(results, key, y_label, title):
    fig = go.Figure()
    for kind, result in results.items():
        fig.add_trace(go.Scatter(x=result["t"], y=result[key], mode="lines", name=模型名稱[kind]))
    fig.update_layout(title=title, xaxis_title="時間（秒）", yaxis_title=y_label,
                      hovermode="x unified", legend_title="模型")
    return fig


def 畫圖(fig):
    st.plotly_chart(fig, use_container_width=True, config=圖表設定)


def 顯示數值(value, unit, digits=3):
    return f"{value:,.{digits}f} {unit}" if value is not None and np.isfinite(value) else "未於模擬期間落地／不適用"


def 基本參數():
    return dict(rho=1.225, cd=0.47, area=0.10, b=0.5, vc=10.0, shape=4.0, n_high=1.5)


# ────────────────────────── 網頁 ──────────────────────────
st.set_page_config(page_title="自由落體研究平台", page_icon="📉", layout="wide")
st.title("自由落體研究平台")
st.write("從運動模擬、數據辨識到相對論動量比較，分別回答三個不同的物理問題。")
st.caption("全程以向上為正方向；向下速度為負。圖表為數值模擬，是否代表真實實驗須依模型適用範圍判斷。")
頁面模擬, 頁面資料, 頁面相對論 = st.tabs(["① 基本運動模擬", "② 數據與模型辨識", "③ 狹義相對論動量與固定外力比較"])

with 頁面模擬:
    st.subheader("研究問題：空氣阻力會如何改變物體的落地時間與速度？")
    st.write("先從真空、線性阻力、平方阻力比較開始。速度－時間圖最適合觀察終端速度；高度－時間圖則用來比較落地時間。")
    with st.form("normal_inputs"):
        a, b, c = st.columns(3)
        with a:
            h0 = st.number_input("初始高度（公尺）", min_value=0.001, value=100.0, key="basic_y0")
            v0 = st.number_input("初速度（公尺／秒；向上為正）", value=0.0, key="basic_v0")
            mass = st.number_input("質量（公斤）", min_value=1e-7, value=1.0, format="%.6g", key="basic_mass")
        with b:
            g = st.number_input("重力加速度（公尺／秒²）", min_value=0.001, value=9.81)
            rho = st.number_input("空氣密度（公斤／公尺³）", min_value=0.0, value=1.225)
            area = st.number_input("迎風截面積（平方公尺）", min_value=0.0, value=0.1, format="%.5g")
        with c:
            cd = st.number_input("平方阻力係數 Cd", min_value=0.0, value=0.47)
            linear_b = st.number_input("線性阻力係數 b（公斤／秒）", min_value=0.0, value=0.5)
            tmax = st.number_input("最長模擬時間（秒）", min_value=1.0, max_value=5000.0, value=60.0)
        with st.expander("進階候選模型：可變冪次（不是普適空氣阻力定律）"):
            show_variable = st.checkbox("將可變冪次加入比較", value=False)
            vv1, vv2, vv3 = st.columns(3)
            with vv1:
                vc = st.number_input("轉變特徵速度 vc（公尺／秒）", min_value=0.01, value=10.0)
            with vv2:
                shape = st.number_input("過渡陡峭參數 p", min_value=0.1, value=4.0)
            with vv3:
                nh = st.number_input("高速極限冪次 n∞", min_value=1.0, max_value=2.0, value=1.5)
            st.caption("此公式只是讓阻力冪次由 2 平滑接近 n∞ 的候選經驗模型，需靠數據檢驗；不是相對論推導出的定律。")
        start_sim = st.form_submit_button("開始／更新模擬", type="primary")
    settings = dict(y0=h0, v0=v0, m=mass, g=g, tmax=tmax,
                    kinds=["vacuum", "linear", "quadratic"] + (["variable"] if show_variable else []),
                    par=dict(rho=rho, cd=cd, area=area, b=linear_b, vc=vc, shape=shape, n_high=nh))
    if start_sim or "sim_settings" not in st.session_state:
        st.session_state["sim_settings"] = settings
    else:
        if settings != st.session_state["sim_settings"]:
            st.info("參數已改動；按「開始／更新模擬」才會套用新設定。")
    try:
        ss = st.session_state["sim_settings"]
        simulated = {k: 求解運動(ss["y0"], ss["v0"], ss["m"], ss["g"], k, ss["par"],
                           t_limit=ss["tmax"]) for k in ss["kinds"]}
        st.subheader("主要結果")
        summary = pd.DataFrame([{"模型": 模型名稱[k], "落地時間（秒）": round(r["hit"], 5) if r["hit"] is not None else None,
                                 "落地速度（公尺／秒）": round(r["v_hit"], 5) if r["v_hit"] is not None else None,
                                 "終端速度（公尺／秒）": round(r["terminal"], 5) if r["terminal"] is not None else None}
                                for k, r in simulated.items()])
        st.dataframe(summary, hide_index=True, use_container_width=True)
        st.caption("終端速度是長時間運動下的理論極限，不代表物體落地前一定已達到。若沒有落地，表示超過設定的模擬時間。")
        left, right = st.columns(2)
        with left:
            畫圖(共同圖表(simulated, "v", "速度（公尺／秒）", "重點圖一：速度如何改變？"))
            st.caption("觀察有阻力的速度曲線是否趨於水平；真空中速度則會持續隨時間改變。")
        with right:
            畫圖(共同圖表(simulated, "y", "高度（公尺）", "重點圖二：何時落地？"))
            st.caption("比較各模型曲線何時到達高度 0，判斷空氣阻力如何影響落地時間。")
        with st.expander("次要圖：加速度與阻力（需要時再看）"):
            畫圖(共同圖表(simulated, "a", "加速度（公尺／秒²）", "加速度－時間"))
            畫圖(共同圖表(simulated, "fd", "阻力（牛頓）", "阻力－時間"))
        if ss["y0"] > 100_000:
            st.warning("起始高度超過 100 公里：固定重力加速度及固定空氣密度的近地面近似可能不再適用。")
        if max(r["max_beta"] for r in simulated.values()) > 1e-6:
            st.warning("速度已不屬於普通落體的低速範圍；本頁為牛頓模型且使用低速空氣阻力經驗式，請勿當作高速空氣動力學預測。")
    except (ValueError, RuntimeError, OverflowError) as e:
        st.error("模擬未能完成：" + str(e))
    with st.expander("本頁使用的物理方程"):
        st.latex(r"m\frac{dv}{dt}=-mg+F_D,\quad \frac{dy}{dt}=v")
        st.latex(r"F_{D,\,\mathrm{linear}}=-bv")
        st.latex(r"F_{D,\,\mathrm{quadratic}}=-\frac12\rho C_D A v|v|")
        st.write("線性阻力適用於特定黏滯流動近似；平方阻力適用於其他常見流動區間，是否適用需考慮物體形狀、雷諾數與流體狀態。")

with 頁面資料:
    st.subheader("研究問題：只知道物體的位置紀錄，能否反推出空氣阻力？")
    st.write("這裡有兩種互補的方法：先不指定阻力公式，估計有效冪次；或事先指定候選模型，比較整體運動軌跡擬合效果。")
    with st.expander("沒有資料？產生一份已知答案的示範 CSV"):
        demo_par = dict(rho=1.225, cd=0.47, area=0.1, b=0.5, vc=10, shape=4, n_high=1.5)
        demo = 求解運動(150, 0, 1, 9.81, "variable", demo_par, t_limit=25, points=420)
        example_df = pd.DataFrame({"time": demo["t"], "height": demo["y"]})
        st.download_button("下載示範資料（無雜訊，可變冪次）", example_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="落體示範資料.csv", mime="text/csv")
        st.caption("示範真值：質量 1 kg、初速 0、空氣密度 1.225、面積 0.1、Cd₀=0.47、vc=10、p=4、n∞=1.5。這是測試數據，不是真實實驗。")
    uploaded = st.file_uploader("上傳時間－高度 CSV", type=["csv"], key="uploaded_data")
    if uploaded is None:
        st.info("先上傳 CSV（至少 time、height 兩欄），即可開始兩種資料分析。")
    else:
        try:
            raw = pd.read_csv(uploaded)
            if len(raw.columns) < 2:
                raise ValueError("CSV 至少要有時間和高度兩欄。")
            cp1, cp2 = st.columns(2)
            with cp1:
                tcol = st.selectbox("時間欄位（秒）", raw.columns, index=0)
            with cp2:
                ycol = st.selectbox("高度欄位（公尺）", raw.columns, index=1)
            if tcol == ycol:
                raise ValueError("時間欄和高度欄不可相同。")
            cleaned = 整理資料(raw, tcol, ycol)
            st.success(f"資料已讀取：{len(cleaned)} 筆有效時間點（重複時間已合併）。")
            st.dataframe(cleaned.head(8), hide_index=True, use_container_width=True)
            st.markdown("**兩種分析共用的物理參數**（請務必設定為資料產生或實驗時的條件）")
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                dm = st.number_input("質量（公斤）", min_value=1e-8, value=1.0, key="data_m")
                dv0 = st.number_input("已知初速度（公尺／秒）", value=0.0, key="data_v0")
            with p2:
                dg = st.number_input("重力加速度（公尺／秒²）", min_value=0.0001, value=9.81, key="data_g")
            with p3:
                drho = st.number_input("空氣密度（公斤／公尺³）", min_value=0.0, value=1.225, key="data_rho")
            with p4:
                darea = st.number_input("截面積（平方公尺）", min_value=0.0, value=0.1, key="data_a")
            with st.expander("進階設定：動量方程"):
                data_sr = st.checkbox("使用狹義相對論動量修正（僅理想化分析）", value=False)
                st.caption("一般落體數據請保持關閉。修正假設固定重力外力；阻力仍是空氣靜止參考系的低速經驗式，不可直接外推至接近光速。")
            method_id, method_fit = st.tabs(["方法甲：不預設阻力冪次", "方法乙：候選模型參數擬合"])
            with method_id:
                st.write("由高度對時間的變化估計速度、加速度，利用運動方程反推出阻力，再估計局部冪次。**這種方法容易放大測量雜訊。**")
                with st.expander("微分與篩選設定"):
                    q1, q2, q3 = st.columns(3)
                    with q1:
                        smooth = st.slider("平滑視窗點數", 7, 151, 31, step=2)
                        poly = st.slider("局部多項式階數", 2, 4, 3)
                    with q2:
                        local = st.slider("局部冪次視窗", 7, 151, 31, step=2)
                        count = st.slider("重取樣數量", 101, 1201, int(np.clip(len(cleaned), 101, 801)), step=1)
                    with q3:
                        min_speed = st.number_input("最小速率（公尺／秒）", min_value=0.0, value=0.5)
                        min_force = st.number_input("最小阻力（牛頓）", min_value=0.0, value=0.001, format="%.6g")
                if st.button("開始阻力律辨識", type="primary"):
                    try:
                        result_id = 辨識阻力律(cleaned, dm, dg, count=count, smooth=smooth, poly=poly,
                                              local=local, min_speed=min_speed, min_force=min_force,
                                              相對論=data_sr)
                        st.session_state["law_result"] = result_id
                        st.session_state["law_signature"] = (uploaded.name, len(cleaned), dm, dg, data_sr)
                    except (ValueError, RuntimeError) as e:
                        st.error(str(e))
                if "law_result" in st.session_state:
                    if st.session_state.get("law_signature") != (uploaded.name, len(cleaned), dm, dg, data_sr):
                        st.warning("資料或物理參數已改變，請重新按「開始阻力律辨識」。")
                    else:
                        rr = st.session_state["law_result"]
                        finite = rr.dropna(subset=["n_eff"])
                        if len(finite) < 12:
                            st.warning("有效冪次點數不足。可嘗試降低篩選門檻、改變平滑視窗或增加數據。")
                        else:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=finite["speed"], y=finite["n_eff"],
                                                     mode="markers", name="由資料估計的冪次"))
                            fig.update_layout(title="重點圖：有效阻力冪次隨速率變化",
                                              xaxis_title="速率（公尺／秒）", yaxis_title="有效冪次 n")
                            畫圖(fig)
                            st.caption("n 接近 1 表示局部近似線性阻力；接近 2 表示局部近似平方阻力。低速區的二次微分誤差可能很大；異常凸起不代表新定律。")
                        with st.expander("診斷圖：估計速度、加速度、反推阻力"):
                            for key, name in [("velocity", "速度"), ("acceleration", "加速度"), ("drag_force", "反推阻力")]:
                                fig = go.Figure(go.Scatter(x=rr["time"], y=rr[key], mode="lines"))
                                fig.update_layout(title=name + "－時間", xaxis_title="時間（秒）", yaxis_title=name)
                                畫圖(fig)
                        with st.expander("進階診斷：分段常數 BIC（不是物理相變）"):
                            count_segments = st.slider("最多區段數", 1, 6, 4)
                            if len(finite) >= 24:
                                bic_table, segments = 分段BIC(finite["n_eff"], limit=count_segments)
                                st.dataframe(bic_table, hide_index=True, use_container_width=True)
                                for s, e, n_value in segments:
                                    st.write(f"估計速率 {finite['speed'].iloc[s]:.2f} ～ {finite['speed'].iloc[e-1]:.2f} 公尺／秒：平均冪次 {n_value:.3f}")
                                st.caption("BIC 只是在分段近似與複雜度間做比較，不能證明真實阻力律存在這些分段。")
                        st.download_button("下載辨識結果 CSV", rr.to_csv(index=False).encode("utf-8-sig"),
                                           file_name="阻力律辨識結果.csv", mime="text/csv")
            with method_fit:
                st.write("先假設一種阻力形式，解微分方程得到預測高度，再調整係數，使預測與 CSV 最接近。")
                fitv0 = st.checkbox("初速度未知，讓程式一起擬合", value=False)
                advanced_fit = st.checkbox("另外擬合可變冪次候選模型（較慢）", value=False)
                dshape = st.number_input("可變冪次模型過渡參數 p（固定，不參與擬合）", 0.1, 20.0, 4.0) if advanced_fit else 4.0
                if st.button("開始候選模型擬合", type="primary"):
                    if drho <= 0 or darea <= 0:
                        st.warning("平方與可變冪次模型需要正的空氣密度與截面積，請先修改參數。")
                    else:
                        with st.spinner("正在數值積分與擬合，可能需要數秒至數十秒……"):
                            fits = 擬合模型(cleaned, dm, dg, drho, darea, dv0,
                                          shape=dshape, fit_v0=fitv0,
                                          kinds=("linear", "quadratic", "variable") if advanced_fit else ("linear", "quadratic"),
                                          相對論=data_sr)
                        st.session_state["fit_result_v2"] = fits
                        st.session_state["fit_signature_v2"] = (uploaded.name, len(cleaned), dm, dg, drho, darea, dv0, data_sr, fitv0, advanced_fit, dshape)
                if "fit_result_v2" in st.session_state:
                    sign = (uploaded.name, len(cleaned), dm, dg, drho, darea, dv0, data_sr, fitv0, advanced_fit, dshape)
                    if st.session_state.get("fit_signature_v2") != sign:
                        st.warning("資料或參數已改變，請重新按「開始候選模型擬合」。")
                    else:
                        fits = st.session_state["fit_result_v2"]
                        view = pd.DataFrame([{"候選模型": f["模型"], "計算狀態": f["狀態"],
                                              "RMSE（公尺）": f.get("RMSE (m)"), "BIC": f.get("BIC")}
                                             for f in fits])
                        st.dataframe(view, hide_index=True, use_container_width=True)
                        st.caption("RMSE 越小代表位置誤差越小；BIC 越小通常較佳，但依賴獨立同分布殘差等統計假設，不能單獨用來證明物理機制。")
                        ok = [f for f in fits if f["狀態"] == "成功"]
                        if ok:
                            best = min(ok, key=lambda z: z["BIC"])
                            st.write("**目前最佳候選：** " + best["模型"])
                            st.write("**擬合參數：** " + "；".join(f"{k} = {v:.6g}" for k, v in best["參數"].items()))
                            t = cleaned["time"].to_numpy()
                            yy = cleaned["height"].to_numpy()
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=t, y=yy, mode="markers", name="上傳的位置數據", marker={"size": 4}))
                            for f in ok:
                                fig.add_trace(go.Scatter(x=t, y=f["預測"], mode="lines", name=f["模型"]))
                            fig.update_layout(title="重點圖：測量高度與模型預測比較", xaxis_title="時間（秒）", yaxis_title="高度（公尺）")
                            畫圖(fig)
                            fig = go.Figure(go.Scatter(x=t, y=best["殘差"], mode="markers", marker={"size": 4}, name="殘差"))
                            fig.add_hline(y=0, line_dash="dash")
                            fig.update_layout(title="重點圖：最佳候選的殘差－時間", xaxis_title="時間（秒）", yaxis_title="實測減預測（公尺）")
                            畫圖(fig)
                            st.caption("如果殘差具有連續規律或波動，即使 R² 很高，也值得調查數值誤差、資料來源及模型適用範圍。")
                            with st.expander("查看所有候選的擬合參數"):
                                for f in ok:
                                    st.write("**" + f["模型"] + "**：" + "；".join(f"{k}={v:.6g}" for k, v in f["參數"].items()))
                            csvfit = pd.DataFrame({"time": t, "height_measured": yy})
                            for f in ok:
                                csvfit[f["模型"] + "_predicted"] = f["預測"]
                                csvfit[f["模型"] + "_residual"] = f["殘差"]
                            st.download_button("下載擬合結果 CSV", csvfit.to_csv(index=False).encode("utf-8-sig"),
                                               file_name="候選模型擬合結果.csv", mime="text/csv")
        except (ValueError, KeyError, pd.errors.ParserError, UnicodeError) as e:
            st.error("CSV 無法分析：" + str(e))

with 頁面相對論:
    st.subheader("研究問題：相對論動量與牛頓動量，對同一外力的運動預測差多少？")
    st.write("這一頁主要研究**動量方程**，不把原本的低速空氣阻力式當成接近光速的真實阻力定律。")
    c1, c2 = st.columns(2)
    with c1:
        beta = st.slider("初速度占光速的比例 |v₀|/c（向上）", min_value=0.0, max_value=0.99, value=0.8, step=0.01)
        lab_g = st.number_input("固定外力參數", min_value=0.001, value=9.81, key="sr_g")
    with c2:
        duration = st.slider("比較時間（秒）", min_value=10, max_value=1000, value=200, step=10)
        st.caption("這裡假設真空且外力固定為 -mg；高速度和大高度下，不可視為真實地球重力落體。")
    gamma = float(洛倫茲因子(beta * 光速))
    ratio = float(加速度修正比例(beta * 光速))
    z1, z2, z3 = st.columns(3)
    z1.metric("初始洛倫茲因子 γ", f"{gamma:.4f}")
    z2.metric("同淨力下相對論／牛頓加速度", f"{ratio:.6f}")
    z3.metric("固定外力下的相對論座標加速度", f"{-lab_g*ratio:.6g} m/s²")
    st.write("**重點圖：速度越接近光速，固定淨力造成的座標加速度修正有多大？**")
    speeds = np.linspace(0, 0.999, 500)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=speeds, y=(1 - speeds ** 2) ** 1.5,
                             mode="lines", name="相對論／牛頓加速度"))
    fig.add_trace(go.Scatter(x=[beta], y=[ratio], mode="markers", marker={"size": 12}, name="目前速度"))
    fig.update_layout(xaxis_title="速度占光速的比例 |v|/c", yaxis_title="加速度比值 1/γ³", yaxis_range=[0, 1.03])
    畫圖(fig)
    st.latex(r"p=\gamma mv,\quad \frac{dp}{dt}=F,\quad a_{\mathrm{SR}}=\frac{F}{m\gamma^3}")
    with st.expander("比較理想化真空情況下的位移差、速度差"):
        # 兩條軌跡均用同一實驗室時間比較；設定足夠高的起始位置避免碰地事件
        vstart = beta * 光速
        height = max(1e7, vstart * duration * 2 + 0.5 * lab_g * duration ** 2 + 10)
        basic_par = 基本參數()
        newton = 求解運動(height, vstart, 1, lab_g, "vacuum", basic_par,
                         相對論=False, t_limit=duration, points=300)
        sr = 求解運動(height, vstart, 1, lab_g, "vacuum", basic_par,
                    相對論=True, t_limit=duration, points=300)
        t = newton["t"]
        if beta < 0.001:
            st.info("這個速度區間的相對論差異接近或小於一般雙精度數值相減的解析度，應使用解析近似檢查，不宜解讀差異圖中的微小波動。")
        else:
            for key, desc, unit in [("y", "高度差（相對論－牛頓）", "公尺"), ("v", "速度差（相對論－牛頓）", "公尺／秒")]:
                fig = go.Figure(go.Scatter(x=t, y=sr[key] - newton[key], mode="lines"))
                fig.update_layout(title=desc, xaxis_title="時間（秒）", yaxis_title=desc + "（" + unit + "）")
                畫圖(fig)
        st.caption("因初始速度向上且高度極大，此例比較的是理想化固定外力下的座標運動，不是可在地球表面實作的高速落體實驗。")
    with st.expander("進階：相同低速經驗阻力式下，比較兩種動量方程"):
        st.warning("此比較只用於測試動量修正的數學效果。原本線性、平方或可變冪次阻力係數沒有被證實可外推到接近光速。")
        drag_kind = st.selectbox("阻力模型", ["linear", "quadratic", "variable"], format_func=lambda k: 模型名稱[k])
        par = 基本參數()
        par["rho"] = st.number_input("空氣密度（公斤／公尺³）", 0.0, 50.0, 1.225, key="sr_rho")
        par["area"] = st.number_input("截面積（平方公尺）", 0.0, 10.0, 0.1, key="sr_area")
        with st.expander("其他低速經驗係數"):
            par["cd"] = st.number_input("阻力係數 Cd", min_value=0.0, value=0.47, key="sr_cd")
            par["b"] = st.number_input("線性係數 b", min_value=0.0, value=0.5, key="sr_b")
            par["vc"] = st.number_input("特徵速度 vc", min_value=0.01, value=10.0, key="sr_vc")
            par["shape"] = st.number_input("過渡參數 p", min_value=0.1, value=4.0, key="sr_p")
            par["n_high"] = st.number_input("高速冪次 n∞", min_value=1.0, max_value=2.0, value=1.5, key="sr_n")
        test_v = st.number_input("模型測試初速度（公尺／秒；向上為正）", value=20.0, key="sr_testv")
        if st.button("比較阻力模型下的兩種動量方程"):
            if abs(test_v) >= 光速:
                st.error("速度必須小於光速。")
            else:
                try:
                    height = max(5000.0, abs(test_v) * 30 + 1e5)
                    sims = {"牛頓": 求解運動(height, test_v, 1.0, lab_g, drag_kind, par, False, 30.0),
                            "相對論": 求解運動(height, test_v, 1.0, lab_g, drag_kind, par, True, 30.0)}
                    fig = go.Figure()
                    for label, r in sims.items():
                        fig.add_trace(go.Scatter(x=r["t"], y=r["a"], mode="lines", name=label))
                    fig.update_layout(title="加速度－時間比較", xaxis_title="時間（秒）", yaxis_title="加速度（公尺／秒²）")
                    畫圖(fig)
                    st.write(f"起始加速度比值：{float(加速度修正比例(test_v)):.12g}；在普通落體速度下，此修正極小。")
                except (ValueError, RuntimeError) as e:
                    st.error("無法完成比較：" + str(e))
    st.info("結論：普通落體速度的相對論修正遠小於常見測量不確定度；高速理想化比較可用來研究動量方程，但不能取代完整重力與高速流體模型。")
