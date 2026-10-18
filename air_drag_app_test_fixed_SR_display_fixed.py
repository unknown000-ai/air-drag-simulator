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
