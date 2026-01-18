"""
Streamlit应用模块
负责创建用户界面和交互逻辑
"""

from src.utils.helpers import setup_logging
from src.visualization.chart_generator import ChartGenerator
from src.strategies.trading_strategy import TradingStrategy
from src.models.hmm_regime_detector import HMMRegimeDetector
from src.features.feature_engineer import FeatureEngineer
from src.data.data_fetcher import DataFetcher
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


class StreamlitApp:
    """Streamlit应用类"""

    def __init__(self):
        self.setup_page_config()
        self.chart_gen = ChartGenerator()
        self.logger = setup_logging()

    def setup_page_config(self):
        """设置页面配置"""
        st.set_page_config(
            page_title="光大银行市场状态识别系统",
            page_icon="🏦",
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def create_sidebar(self):
        """创建侧边栏控件"""
        st.sidebar.title("参数设置")

        # 股票选择
        stock_options = {
            "光大银行(601818)": "sh.601818",
            "工商银行(601398)": "sh.601398",
            "建设银行(601939)": "sh.601939",
            "农业银行(601288)": "sh.601288",
        }

        selected_stock = st.sidebar.selectbox(
            "选择股票", options=list(stock_options.keys()), index=0
        )
        stock_code = stock_options[selected_stock]

        # 日期范围选择
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 3)  # 默认3年数据

        date_range = st.sidebar.date_input(
            "选择日期范围",
            value=(start_date, end_date),
            min_value=end_date - timedelta(days=365 * 10),  # 最多10年
            max_value=end_date,
        )

        # HMM参数设置
        st.sidebar.subheader("HMM参数")
        n_states = st.sidebar.slider(
            "状态数量",
            min_value=2,
            max_value=5,
            value=3,
            help="隐马尔可夫模型的状态数量",
        )

        use_smoothing = st.sidebar.checkbox(
            "启用状态平滑", value=True, help="使用Viterbi算法进行状态平滑"
        )

        # 分析按钮
        analyze_clicked = st.sidebar.button(
            "开始分析", type="primary", use_container_width=True
        )

        return {
            "stock_code": stock_code,
            "stock_name": selected_stock,
            "start_date": date_range[0] if len(date_range) > 0 else start_date,
            "end_date": date_range[1] if len(date_range) > 1 else end_date,
            "n_states": n_states,
            "use_smoothing": use_smoothing,
            "analyze_clicked": analyze_clicked,
        }

    def display_header(self, stock_name: str):
        """显示页面头部"""
        st.title(f"🏦 {stock_name} 市场状态识别系统")
        st.markdown(
            """
        基于隐马尔可夫模型(HMM)的市场状态识别系统，通过技术指标和宏观数据
        识别股票市场的牛市、熊市和中性状态，并生成相应的交易策略。
        """
        )

    def display_loading_indicator(self, message: str):
        """显示加载指示器"""
        with st.spinner(message):
            time.sleep(0.5)  # 模拟加载过程

    def display_data_summary(self, df: pd.DataFrame, stock_name: str):
        """显示数据摘要"""
        st.subheader("📊 数据摘要")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "数据期间",
                f"{df.index.min().strftime('%Y-%m-%d')} 至 {df.index.max().strftime('%Y-%m-%d')}",
            )

        with col2:
            st.metric("数据天数", len(df))

        with col3:
            price_change = (
                (df["close"].iloc[-1] - df["close"].iloc[0]) / df["close"].iloc[0]
            ) * 100
            st.metric("期间涨跌幅", f"{price_change:.2f}%")

        with col4:
            avg_volume = df["volume"].mean()
            st.metric("平均成交量", f"{avg_volume:,.0f}")

        # 显示价格图表
        price_chart = self.chart_gen.create_price_chart(df, f"{stock_name} 价格走势")
        st.plotly_chart(price_chart, use_container_width=True)

    def display_feature_analysis(self, features_df: pd.DataFrame):
        """显示特征分析"""
        st.subheader("🔍 特征分析")

        # 显示特征统计
        col1, col2, col3, col4 = st.columns(4)

        feature_stats = features_df.describe()

        with col1:
            if "VOL" in features_df.columns:
                st.metric("波动率均值", f"{feature_stats.loc['mean', 'VOL']:.4f}")

        with col2:
            if "SPREAD" in features_df.columns:
                st.metric("股债利差均值", f"{feature_stats.loc['mean', 'SPREAD']:.4f}")

        with col3:
            if "EBS" in features_df.columns:
                st.metric("EBS均值", f"{feature_stats.loc['mean', 'EBS']:.4f}")

        with col4:
            if "BUFFETT" in features_df.columns:
                st.metric(
                    "巴菲特指数均值", f"{feature_stats.loc['mean', 'BUFFETT']:.4f}"
                )

        # 显示特征分析图表
        feature_chart = self.chart_gen.create_feature_analysis_chart(features_df)
        st.plotly_chart(feature_chart, use_container_width=True)

    def display_market_regime(
        self, df: pd.DataFrame, regime_data: pd.Series, model_info: dict
    ):
        """显示市场状态识别结果"""
        st.subheader("📈 市场状态识别")

        # 显示模型信息
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("模型收敛", "是" if model_info.get("converged", False) else "否")

        with col2:
            st.metric("训练迭代次数", model_info.get("n_iter", 0))

        with col3:
            st.metric(
                "最终对数似然", f"{model_info.get('final_log_likelihood', 0):.2f}"
            )

        # 显示状态识别图表
        regime_chart = self.chart_gen.create_market_regime_chart(df, regime_data)
        st.plotly_chart(regime_chart, use_container_width=True)

        # 显示状态统计
        regime_counts = regime_data.value_counts().to_dict()
        regime_stats_chart = self.chart_gen.create_regime_statistics_chart(
            regime_counts
        )
        st.plotly_chart(regime_stats_chart, use_container_width=True)

    def display_strategy_performance(self, strategy_results: dict):
        """显示策略表现"""
        st.subheader("💹 策略表现")

        if not strategy_results:
            st.warning("无策略结果可显示")
            return

        # 显示关键指标
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("年化收益率", f"{strategy_results.get('cagr', 0)*100:.2f}%")

        with col2:
            st.metric("夏普比率", f"{strategy_results.get('sharpe', 0):.2f}")

        with col3:
            st.metric("最大回撤", f"{strategy_results.get('max_drawdown', 0)*100:.2f}%")

        with col4:
            st.metric("胜率", f"{strategy_results.get('win_rate', 0)*100:.2f}%")

        # 显示策略表现图表
        if "returns" in strategy_results:
            perf_chart = self.chart_gen.create_strategy_performance_chart(
                strategy_results["returns"]
            )
            st.plotly_chart(perf_chart, use_container_width=True)

    def display_error_message(self, error_message: str):
        """显示错误信息"""
        st.error(f"❌ 错误: {error_message}")
        st.info(
            """
        可能的解决方案：
        - 检查网络连接
        - 确认数据源可用性  
        - 调整日期范围
        - 重启应用
        """
        )

    def run_analysis(self, params: dict):
        """运行分析流程"""
        try:
            # 1. 获取数据
            self.display_loading_indicator("正在获取股票数据...")
            data_fetcher = DataFetcher()
            stock_data = data_fetcher.get_cebbank_data(
                params["stock_code"], params["start_date"], params["end_date"]
            )

            if stock_data is None or stock_data.empty:
                raise ValueError("无法获取股票数据")

            # 2. 获取宏观数据
            self.display_loading_indicator("正在获取宏观数据...")
            ebs_data = data_fetcher.get_ebs_data(
                params["start_date"], params["end_date"]
            )
            buffett_data = data_fetcher.get_buffett_index(
                params["start_date"], params["end_date"]
            )

            # 3. 特征工程
            self.display_loading_indicator("正在进行特征工程...")
            feature_engineer = FeatureEngineer()
            feature_result = feature_engineer.engineer_features(
                stock_data, ebs_data, buffett_data
            )

            if not feature_engineer.validate_features():
                raise ValueError("特征数据验证失败")

            # 4. HMM模型训练
            self.display_loading_indicator("正在训练HMM模型...")
            hmm_detector = HMMRegimeDetector(n_states=params["n_states"])
            regime_result = hmm_detector.detect_regimes(feature_result["features_df"])

            # 5. 生成交易策略
            self.display_loading_indicator("正在生成交易策略...")
            strategy = TradingStrategy()
            strategy_results = strategy.generate_signals(
                stock_data, regime_result["regime_series"]
            )

            return {
                "stock_data": stock_data,
                "feature_result": feature_result,
                "regime_result": regime_result,
                "strategy_results": strategy_results,
            }

        except Exception as e:
            self.logger.error(f"分析过程中出现错误: {str(e)}")
            raise

    def run(self):
        """运行Streamlit应用"""
        # 显示页面头部
        st.markdown(
            """
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # 创建侧边栏
        params = self.create_sidebar()

        # 显示主内容
        self.display_header(params["stock_name"])

        # 检查是否点击分析按钮
        if params["analyze_clicked"]:
            try:
                # 运行分析
                results = self.run_analysis(params)

                # 显示结果
                self.display_data_summary(results["stock_data"], params["stock_name"])
                self.display_feature_analysis(results["feature_result"]["features_df"])
                self.display_market_regime(
                    results["stock_data"],
                    results["regime_result"]["regime_series"],
                    results["regime_result"]["model_info"],
                )
                self.display_strategy_performance(results["strategy_results"])

                st.success("✅ 分析完成！")

            except Exception as e:
                self.display_error_message(str(e))
        else:
            # 显示欢迎信息
            st.info("👈 请在左侧设置参数，然后点击'开始分析'按钮")

            # 显示使用说明
            with st.expander("📖 使用说明"):
                st.markdown(
                    """
                ### 系统功能说明
                
                1. **数据获取**: 从baostock获取股票数据，从akshare获取宏观数据
                2. **特征工程**: 计算技术指标和标准化处理
                3. **状态识别**: 使用HMM模型识别市场状态
                4. **策略生成**: 基于市场状态生成交易信号
                5. **可视化**: 多维度展示分析结果
                
                ### 参数说明
                
                - **状态数量**: HMM模型的状态数量（2-5）
                - **状态平滑**: 使用Viterbi算法进行状态平滑，提高识别准确性
                - **日期范围**: 建议选择3年以上数据以获得更稳定的结果
                """
                )


def main():
    """主函数"""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
