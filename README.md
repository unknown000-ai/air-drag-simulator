# Air Drag Simulator

自由落體與空氣阻力數值模擬網頁。

## Online Demo

https://nknu-air-drag-simulator.streamlit.app/

## Features

- 真空自由落體
- 線性空氣阻力
- 平方空氣阻力
- 三種模型比較
- 自訂初始高度與初速度
- 自訂質量、空氣密度、阻力係數與截面積
- 終端速度計算
- 高度、速度、加速度與空氣阻力圖
- Plotly 互動式縮放
- 棒球、羽毛、雨滴等物體預設值

## Physics Models

### Vacuum

\[
\frac{dv}{dt}=-g
\]

### Linear Drag

\[
F_D=-bv
\]

### Quadratic Drag

\[
F_D=-\frac{1}{2}\rho C_D A v|v|
\]

## Numerical Method

The simulation uses SciPy `solve_ivp` to solve the equations of motion numerically.

## Technologies

- Python
- Streamlit
- SciPy
- NumPy
- Plotly
