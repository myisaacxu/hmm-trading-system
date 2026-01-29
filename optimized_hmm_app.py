#!/usr/bin/env python3
"""
光大银行市场状态识别系统 - 优化版

基于隐马尔可夫模型(HMM)的银行股市场状态识别与交易策略应用。
系统使用baostock获取光大银行历史数据，结合akshare获取的宏观数据，
实现完整的市场状态识别与交易策略回测功能，并添加参数优化和模型应用功能。

使用方法:
    streamlit run optimized_hmm_app.py

作者: AI Assistant
版本: 2.0.0
"""

import os
import sys
import warnings
from pathlib import Path

# 导入日志系统
from src.utils.logger import global_logger

# 添加当前目录到Python路径，确保模块导入正常
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 过滤警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import baostock as bs
    import akshare as ak
    from hmmlearn import hmm
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import joblib
    import json
    from datetime import datetime, timedelta
    from typing import Optional, Dict, Tuple, List

    # 导入项目模块
    try:
        from src.data.data_fetcher import DataFetcher
        from src.features.feature_engineer import FeatureEngineer
        from src.models.market_state_analyzer import MarketStateAnalyzer
        from src.models.model_manager import ModelManager
        from src.models.hmm_optimizer import HMMOptimizer
        from src.models.hmm_regime_detector import HMMRegimeDetector

        MODULE_IMPORT_SUCCESS = True
    except ImportError as e:
        st.warning(f"模块导入警告: {e}")
        st.info("正在使用备用模式运行...")
        MODULE_IMPORT_SUCCESS = False

except ImportError as e:
    print(f"错误: 缺少必要的依赖包 - {e}")
    print("\n请安装以下依赖包:")
    print("pip install streamlit pandas numpy baostock akshare hmmlearn plotly joblib")
    sys.exit(1)


def check_dependencies():
    """检查依赖包是否已安装"""
    required_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "baostock",
        "akshare",
        "hmmlearn",
        "plotly",
        "joblib",
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    return missing_packages


def setup_environment():
    """设置运行环境"""
    # 创建必要的目录
    models_dir = current_dir / "models"
    models_dir.mkdir(exist_ok=True)
    global_logger.info(f"已创建或确认模型目录: {models_dir}")

    logs_dir = current_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    global_logger.info(f"已创建或确认日志目录: {logs_dir}")


def get_ebs_data():
    """获取股债利差数据"""
    try:
        ebs_df = ak.stock_ebs_lg()

        if ebs_df.empty:
            st.error("获取的股债利差数据为空")
            return None

        ebs_df = ebs_df.rename(columns={"日期": "date", "股债利差": "ebs_indicator"})
        ebs_df = ebs_df[["date", "ebs_indicator"]]
        ebs_df["ebs_indicator"] = ebs_df["ebs_indicator"] * 100

        ebs_df["date"] = pd.to_datetime(ebs_df["date"])
        ebs_df = ebs_df.set_index("date").sort_index()

        return ebs_df["ebs_indicator"]

    except Exception as e:
        st.error(f"获取股债利差数据失败: {e}")
        return None


def get_buffett_index():
    """获取巴菲特指数数据"""
    try:
        # 优先使用akshare的直接接口获取巴菲特指数
        buffett_df = ak.stock_buffett_index_lg()

        if not buffett_df.empty:
            # 重命名和格式化
            buffett_df = buffett_df.rename(columns={"日期": "date"})

            # 计算巴菲特指数：总市值/GDP * 100
            if "总市值" in buffett_df.columns and "GDP" in buffett_df.columns:
                buffett_df["buffett_index"] = (
                    buffett_df["总市值"] / buffett_df["GDP"]
                ) * 100
            else:
                # 如果列名不同，尝试其他可能的列名组合
                if "market_cap" in buffett_df.columns and "gdp" in buffett_df.columns:
                    buffett_df["buffett_index"] = (
                        buffett_df["market_cap"] / buffett_df["gdp"]
                    ) * 100
                else:
                    # 使用默认的巴菲特指数列
                    buffett_df["buffett_index"] = (
                        buffett_df.iloc[:, 1] if len(buffett_df.columns) > 1 else 100
                    )

            # 转换日期格式
            buffett_df["date"] = pd.to_datetime(buffett_df["date"])

            # 设置日期为索引并排序
            buffett_df.set_index("date", inplace=True)
            buffett_df = buffett_df.sort_index()

            return buffett_df["buffett_index"]
        else:
            # 如果直接接口失败，回退到组合计算方式
            return _calculate_buffett_index_fallback()

    except Exception as e:
        st.warning(f"获取巴菲特指数数据失败，尝试备用方法: {e}")
        # 使用备用计算方式
        return _calculate_buffett_index_fallback()


def _calculate_buffett_index_fallback():
    """备用方法：手动计算巴菲特指数"""
    try:
        # 使用akshare获取GDP和市值数据
        gdp_df = ak.macro_china_gdp()
        market_cap_df = ak.macro_china_stock_market_cap()

        if gdp_df.empty or market_cap_df.empty:
            st.warning("无法获取完整的巴菲特指数数据")
            return None

        # 处理市值数据
        if not market_cap_df.empty:
            latest_market_cap = market_cap_df.iloc[0]
            shanghai_market_cap = latest_market_cap.get("市价总值-上海", 0)
            shenzhen_market_cap = latest_market_cap.get("市价总值-深圳", 0)

            if pd.notna(shanghai_market_cap) and pd.notna(shenzhen_market_cap):
                total_market_cap = shanghai_market_cap + shenzhen_market_cap
            else:
                total_market_cap = None
        else:
            total_market_cap = None

        # 获取最新的GDP数据
        latest_gdp = gdp_df.iloc[0] if not gdp_df.empty else None

        if latest_gdp is not None and "value" in latest_gdp:
            current_gdp = latest_gdp["value"]

            # 计算巴菲特指数 = 总市值 / GDP
            buffett_index_value = (
                (total_market_cap / current_gdp) * 100
                if current_gdp > 0 and total_market_cap is not None
                else 100
            )

            # 创建时间序列数据，使用日频率以匹配股票数据
            dates = pd.date_range(start="2010-01-01", end=datetime.now(), freq="D")
            buffett_values = np.full(len(dates), buffett_index_value)

            return pd.Series(buffett_values, index=dates, name="buffett_index")
        else:
            # GDP数据获取失败，使用模拟数据
            dates = pd.date_range(start="2010-01-01", end=datetime.now(), freq="D")
            buffett_values = np.linspace(80, 120, len(dates))

            return pd.Series(buffett_values, index=dates, name="buffett_index")

    except Exception as e:
        st.warning(f"备用方法也失败，使用默认值: {e}")
        # 返回默认数据，使用更合理的范围80-120而不是固定的150
        dates = pd.date_range(start="2010-01-01", end=datetime.now(), freq="D")
        buffett_values = np.linspace(80, 120, len(dates))

        return pd.Series(buffett_values, index=dates, name="buffett_index")


def get_cebbank_data(start_date: str, end_date: str = None) -> Optional[pd.DataFrame]:
    """获取光大银行股票数据"""
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    try:
        # 登录baostock
        bs.login()

        # 查询光大银行(601818)数据
        rs = bs.query_history_k_data_plus(
            "sh.601818",
            "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )

        data_list = []
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())

        result = pd.DataFrame(data_list, columns=rs.fields)

        if result.empty:
            st.error("获取的光大银行数据为空")
            return None

        # 数据类型转换
        numeric_cols = [
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "turn",
            "pctChg",
        ]
        for col in numeric_cols:
            result[col] = pd.to_numeric(result[col], errors="coerce")

        result["date"] = pd.to_datetime(result["date"])
        result = result.set_index("date").sort_index()

        # 登出baostock
        bs.logout()

        return result

    except Exception as e:
        st.error(f"获取光大银行数据失败: {e}")
        try:
            bs.logout()
        except Exception:
            pass
        return None


def calculate_technical_indicators(
    price_series: pd.Series,
    vol_window: int = 30,
    ma_short: int = 20,
    ma_long: int = 100,
) -> pd.DataFrame:
    """计算技术指标"""
    price_series = price_series[~price_series.index.duplicated(keep="first")]
    price_series = price_series.sort_index().ffill().bfill().fillna(0.0)

    # 计算对数收益率和波动率
    lr = np.log(price_series).diff().fillna(0.0)
    vol = lr.rolling(vol_window, min_periods=1).std().fillna(0.0)

    # 计算均线和趋势指标
    ma_short = price_series.rolling(ma_short).mean().bfill()
    ma_long = price_series.rolling(ma_long).mean().bfill()
    spread = ((ma_short - ma_long) / ma_long).fillna(0.0)

    tech_df = pd.DataFrame(
        {
            "PX": price_series,
            "log_ret": lr,
            "VOL": vol,
            "MA_SHORT": ma_short,
            "MA_LONG": ma_long,
            "SPREAD": spread,
        }
    )

    return tech_df


def align_macro_data(
    tech_df: pd.DataFrame,
    ebs_data: Optional[pd.Series] = None,
    buffett_data: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """对齐宏观数据"""
    features_df = tech_df.copy()

    # 对齐股债利差数据
    if ebs_data is not None:
        ebs_aligned = ebs_data.reindex(features_df.index, method="ffill")
        features_df["EBS"] = ebs_aligned.fillna(0.0)
    else:
        features_df["EBS"] = 0.0

    # 对齐巴菲特指数数据
    if buffett_data is not None:
        buffett_aligned = buffett_data.reindex(features_df.index, method="ffill")
        features_df["BUFFETT"] = buffett_aligned.fillna(0.0)
    else:
        features_df["BUFFETT"] = 0.0

    return features_df


def create_feature_matrix(
    features_df: pd.DataFrame, use_standardization: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """创建特征矩阵"""
    feature_cols = ["log_ret", "VOL", "SPREAD", "EBS", "BUFFETT"]

    if not all(col in features_df.columns for col in feature_cols):
        st.error("特征数据不完整")
        return np.array([]), np.array([])

    X = features_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Z-分数标准化
    Xz = X.copy()
    if use_standardization and len(X) > 1:
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-12
        Xz = (X - X_mean) / X_std

    return X, Xz


def generate_trading_signals(
    price_data: pd.DataFrame, labels: np.ndarray, state_names: List[str] = None
) -> pd.DataFrame:
    """生成交易信号"""
    if state_names is None:
        state_names = [f"State_{i}" for i in range(len(np.unique(labels)))]

    signals = price_data.copy()
    signals["regime"] = [state_names[i] for i in labels]

    # 生成交易信号
    signals["position"] = 0
    signals.loc[signals["regime"].str.contains("Bull"), "position"] = 1
    signals.loc[signals["regime"].str.contains("Bear"), "position"] = -1

    # 计算策略收益率
    signals["log_ret"] = np.log(signals["close"]).diff().fillna(0.0)
    signals["strat_ret"] = signals["position"] * signals["log_ret"]

    return signals


def calculate_performance_metrics(signals: pd.DataFrame) -> Dict:
    """计算性能指标"""
    if signals.empty:
        return {}

    # 买入持有策略
    bh_cum_ret = np.exp(signals["log_ret"].cumsum())
    bh_final_ret = bh_cum_ret.iloc[-1] - 1

    # HMM策略
    strat_cum_ret = np.exp(signals["strat_ret"].cumsum())
    strat_final_ret = strat_cum_ret.iloc[-1] - 1

    # 年化收益率
    days = len(signals)
    years = days / 252

    bh_cagr = (1 + bh_final_ret) ** (1 / years) - 1 if years > 0 else 0
    strat_cagr = (1 + strat_final_ret) ** (1 / years) - 1 if years > 0 else 0

    # 夏普比率（简化计算）
    bh_sharpe = (
        bh_cagr / (signals["log_ret"].std() * np.sqrt(252))
        if signals["log_ret"].std() > 0
        else 0
    )
    strat_sharpe = (
        strat_cagr / (signals["strat_ret"].std() * np.sqrt(252))
        if signals["strat_ret"].std() > 0
        else 0
    )

    # 最大回撤
    bh_roll_max = bh_cum_ret.cummax()
    bh_drawdown = (bh_cum_ret - bh_roll_max) / bh_roll_max
    bh_mdd = bh_drawdown.min() if not bh_drawdown.empty else 0

    strat_roll_max = strat_cum_ret.cummax()
    strat_drawdown = (strat_cum_ret - strat_roll_max) / strat_roll_max
    strat_mdd = strat_drawdown.min() if not strat_drawdown.empty else 0

    return {
        "buy_hold": {
            "cagr": bh_cagr,
            "sharpe": bh_sharpe,
            "final_return": bh_final_ret,
            "mdd": bh_mdd,
        },
        "hmm_strategy": {
            "cagr": strat_cagr,
            "sharpe": strat_sharpe,
            "final_return": strat_final_ret,
            "mdd": strat_mdd,
        },
    }


def create_price_chart(price_data: pd.DataFrame, labels: np.ndarray) -> go.Figure:
    """创建价格走势图"""
    fig = go.Figure()

    # 价格走势
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data["close"],
            mode="lines",
            name="光大银行收盘价",
            line=dict(color="blue", width=2),
        )
    )

    # 添加状态区域
    unique_states = np.unique(labels)
    colors = ["green", "red", "gray", "orange", "purple"]

    for i, state in enumerate(unique_states):
        mask = labels == state
        state_dates = price_data.index[mask]

        if len(state_dates) > 0:
            fig.add_vrect(
                x0=state_dates[0],
                x1=state_dates[-1],
                fillcolor=colors[i % len(colors)],
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text=f"State {state}",
                annotation_position="top left",
            )

    fig.update_layout(
        title="光大银行价格走势与市场状态识别",
        xaxis_title="日期",
        yaxis_title="价格(元)",
        height=400,
    )

    return fig


def main():
    """主函数"""
    # 记录应用启动
    global_logger.info("光大银行市场状态识别系统(优化版)启动")

    # 设置页面配置
    st.set_page_config(
        page_title="光大银行市场状态识别 - 优化版",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # 应用标题
    st.title("🏦 光大银行市场状态识别系统 - 优化版")
    st.markdown(
        "基于隐马尔可夫模型(HMM)的银行股市场状态识别与交易策略，支持参数优化和模型应用"
    )

    # 侧边栏配置参数
    st.sidebar.header("策略参数配置")

    # 基本参数
    start_date = st.sidebar.date_input("开始日期", pd.to_datetime("2010-01-01"))
    n_states = st.sidebar.slider("状态数量", 2, 6, 3)
    min_len = st.sidebar.slider("最小状态持续时间", 5, 30, 15)
    stickiness = st.sidebar.slider("状态粘性", 1.0, 20.0, 8.0)

    # 银行股专用参数
    st.sidebar.markdown("---")
    st.sidebar.header("银行股专用参数")
    vol_window = st.sidebar.slider("波动率计算窗口", 10, 60, 30)
    ma_short = st.sidebar.slider("短期均线窗口", 10, 50, 20)
    ma_long = st.sidebar.slider("长期均线窗口", 50, 200, 100)

    # 参数优化配置
    st.sidebar.markdown("---")
    st.sidebar.header("参数优化配置")
    enable_optimization = st.sidebar.checkbox("启用参数优化", value=False)

    if enable_optimization:
        optimization_method = st.sidebar.selectbox(
            "优化方法", ["网格搜索", "特征选择", "平滑参数优化"]
        )
        n_splits = st.sidebar.slider("交叉验证折数", 2, 5, 3)

    # 模型管理配置
    st.sidebar.markdown("---")
    st.sidebar.header("模型管理")
    use_saved_model = st.sidebar.checkbox("使用保存的模型", value=False)

    # 初始化模型管理器
    model_manager = ModelManager()
    optimizer = HMMOptimizer()

    end = None

    global_logger.info(
        f"策略参数配置完成: 开始日期={start_date}, 状态数量={n_states}, 最小状态持续时间={min_len}, 状态粘性={stickiness}"
    )

    # 显示进度
    with st.spinner("正在获取数据并计算..."):
        # 下载光大银行数据
        global_logger.info(f"开始获取光大银行数据，日期范围: {start_date} 到 {end}")
        df_cebbank = get_cebbank_data(start_date.strftime("%Y-%m-%d"), end)

        if df_cebbank is None or df_cebbank.empty:
            st.error("无法获取光大银行数据，请检查网络连接或日期范围")
            global_logger.error("无法获取光大银行数据")
            st.stop()

        global_logger.info(f"成功获取光大银行数据，数据长度: {len(df_cebbank)}")

        px_series = df_cebbank["close"].rename("PX")

        # 检查并处理重复索引
        px_series = px_series[~px_series.index.duplicated(keep="first")]

        # 计算对数收益率和波动率特征（针对银行股优化）
        global_logger.info("开始计算技术指标")
        lr = np.log(px_series).diff().fillna(0.0)  # 对数收益率
        vol = lr.rolling(vol_window, min_periods=1).std().fillna(0.0)  # 可调波动率窗口
        ma_short_series = px_series.rolling(ma_short).mean().bfill()
        ma_long_series = px_series.rolling(ma_long).mean().bfill()
        spread = ((ma_short_series - ma_long_series) / ma_long_series).fillna(
            0.0
        )  # 可调趋势指标
        global_logger.info("技术指标计算完成")

        # 获取股债利差数据
        global_logger.info("开始获取股债利差数据")
        ebs_data = get_ebs_data()
        if ebs_data is not None:
            # 检查并处理重复索引
            ebs_data = ebs_data[~ebs_data.index.duplicated(keep="first")]
            ebs_data = ebs_data.reindex(px_series.index, method="ffill")  # 对齐日期索引
            global_logger.info("成功获取并对齐股债利差数据")
        else:
            # 如果获取失败，创建空的股债利差列
            ebs_data = pd.Series(0.0, index=px_series.index, name="ebs_indicator")
            global_logger.warning("无法获取股债利差数据，使用默认值")

        # 获取巴菲特指数数据
        global_logger.info("开始获取巴菲特指数数据")
        buffett_data = get_buffett_index()
        if buffett_data is not None:
            # 检查并处理重复索引
            buffett_data = buffett_data[~buffett_data.index.duplicated(keep="first")]
            # 确保数据范围与股票数据匹配
            if len(buffett_data) > 0:
                # 获取与股票数据日期范围重叠的部分
                buffett_in_range = buffett_data[
                    (buffett_data.index >= px_series.index.min())
                    & (buffett_data.index <= px_series.index.max())
                ]

                if len(buffett_in_range) > 0:
                    # 使用前向填充对齐到股票数据索引
                    buffett_aligned = buffett_in_range.reindex(
                        px_series.index, method="ffill"
                    )
                    buffett_data = buffett_aligned.fillna(method="bfill").fillna(
                        buffett_in_range.mean() if not buffett_in_range.empty else 100.0
                    )
                else:
                    # 如果数据范围不重叠，使用最近的值
                    buffett_data = pd.Series(
                        buffett_data.iloc[-1] if not buffett_data.empty else 100.0,
                        index=px_series.index,
                        name="buffett_index",
                    )
            else:
                buffett_data = pd.Series(
                    100.0, index=px_series.index, name="buffett_index"
                )
            global_logger.info("成功获取并对齐巴菲特指数数据")
        else:
            # 如果获取失败，使用合理的默认值而不是0.0
            buffett_data = pd.Series(100.0, index=px_series.index, name="buffett_index")
            global_logger.warning("无法获取巴菲特指数数据，使用默认值")

        # 创建特征数据框
        df = pd.DataFrame(
            {
                "PX": px_series,
                "VOL": vol,
                "SPREAD": spread,
                "EBS": (
                    ebs_data["ebs_indicator"]
                    if isinstance(ebs_data, pd.DataFrame)
                    else ebs_data
                ),
                "BUFFETT": (
                    buffett_data["buffett_index"]
                    if isinstance(buffett_data, pd.DataFrame)
                    else buffett_data
                ),
            }
        ).dropna()

        # 确保日期索引对齐
        df = df.sort_index()
        global_logger.info(f"特征数据框创建完成，数据长度: {len(df)}")

        # 设计矩阵（行对应 df.index）
        X = np.column_stack(
            [
                np.log(df["PX"]).diff().fillna(0.0).values,  # 对数收益率
                df["VOL"].values,  # 波动率
                df["SPREAD"].values,  # 趋势指标
                df["EBS"].values,  # 股债利差
                df["BUFFETT"].values,  # 巴菲特指数
            ]
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 可选：Z-分数标准化，提高状态区分度
        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-12
        Xz = (X - X_mean) / X_std

        # 核心逻辑 — 使用导入的HMMRegimeDetector类

        # 执行参数优化
        best_params = {"n_states": n_states, "covariance_type": "diag", "n_iter": 100}

        if enable_optimization:
            st.subheader("参数优化结果")

            if optimization_method == "网格搜索":
                # 定义参数网格
                param_grid = {
                    "n_states": [2, 3, 4],
                    "covariance_type": ["diag", "full"],
                    "n_iter": [100, 200],
                }

                # 执行网格搜索
                with st.spinner("正在执行网格搜索优化..."):
                    optimization_result = optimizer.optimize_parameters(
                        Xz, param_grid, n_splits=n_splits
                    )

                best_params = optimization_result["best_params"]
                best_score = optimization_result["best_score"]

                st.success(f"网格搜索优化完成！最佳参数: {best_params}")
                st.info(f"最佳得分: {best_score:.4f}")

                # 显示交叉验证结果
                with st.expander("查看交叉验证结果"):
                    cv_results = optimization_result["cv_results"]
                    cv_df = pd.DataFrame(cv_results)
                    st.dataframe(cv_df)

            elif optimization_method == "特征选择":
                # 定义特征列表
                # 先计算log_ret并添加到df中
                df_with_log_ret = df.copy()
                df_with_log_ret["log_ret"] = (
                    np.log(df_with_log_ret["PX"]).diff().fillna(0.0)
                )

                feature_columns = ["log_ret", "VOL", "SPREAD", "EBS", "BUFFETT"]

                # 执行特征选择
                with st.spinner("正在执行特征选择..."):
                    feature_result = optimizer.optimize_features(
                        df_with_log_ret, feature_columns, n_states=n_states
                    )

                best_features = feature_result["best_features"]
                best_score = feature_result["best_score"]
                feature_importance = feature_result["feature_importance"]

                st.success(f"特征选择完成！最佳特征组合: {best_features}")
                st.info(f"最佳得分: {best_score:.4f}")

                # 显示特征重要性
                with st.expander("查看特征重要性"):
                    importance_df = pd.DataFrame(
                        list(feature_importance.items()), columns=["特征", "重要性"]
                    ).sort_values(by="重要性", ascending=False)
                    st.dataframe(importance_df)

                    # 可视化特征重要性
                    fig = px.bar(
                        importance_df, x="特征", y="重要性", title="特征重要性排序"
                    )
                    st.plotly_chart(fig)

                # 使用最佳特征重新创建特征矩阵
                if best_features:
                    # 重新计算最佳特征的特征矩阵
                    feature_map = {
                        "log_ret": np.log(df["PX"]).diff().fillna(0.0).values,
                        "VOL": df["VOL"].values,
                        "SPREAD": df["SPREAD"].values,
                        "EBS": df["EBS"].values,
                        "BUFFETT": df["BUFFETT"].values,
                    }

                    X_best = np.column_stack(
                        [feature_map[feature] for feature in best_features]
                    )
                    X_best = np.nan_to_num(X_best, nan=0.0, posinf=0.0, neginf=0.0)

                    # 标准化
                    X_best_mean = X_best.mean(axis=0, keepdims=True)
                    X_best_std = X_best.std(axis=0, keepdims=True) + 1e-12
                    Xz = (X_best - X_best_mean) / X_best_std

            elif optimization_method == "平滑参数优化":
                # 执行平滑参数优化
                with st.spinner("正在执行平滑参数优化..."):
                    smoothing_result = optimizer.optimize_smoothing_parameters(
                        Xz, n_states=n_states
                    )

                best_smoothing_params = smoothing_result["best_params"]
                best_score = smoothing_result["best_score"]

                st.success(f"平滑参数优化完成！最佳参数: {best_smoothing_params}")
                st.info(f"最佳得分: {best_score:.4f}")

                # 更新平滑参数
                min_len = best_smoothing_params["min_duration"]
                stickiness = best_smoothing_params["sticky_strength"]

        # 训练模型或加载保存的模型
        if use_saved_model:
            # 列出保存的模型
            saved_models = model_manager.list_saved_models()

            if saved_models:
                st.subheader("选择保存的模型")

                # 创建模型选择下拉菜单
                model_options = [
                    f"{model['name']} (夏普: {model['sharpe']:.2f}, 年化: {model['cagr']:.2f})"
                    for model in saved_models
                ]
                selected_model_idx = st.selectbox(
                    "选择模型",
                    range(len(model_options)),
                    format_func=lambda x: model_options[x],
                )

                selected_model = saved_models[selected_model_idx]
                model_name = selected_model["name"]

                # 加载模型
                with st.spinner(f"正在加载模型: {model_name}..."):
                    loaded_model, loaded_metrics = model_manager.load_model(model_name)

                if loaded_model is not None:
                    st.success(f"模型加载成功！模型名称: {model_name}")
                    if loaded_metrics:
                        sharpe = loaded_metrics.get("sharpe", 0.0)
                        cagr = loaded_metrics.get("cagr", 0.0)
                        st.info(f"模型性能: 夏普比率={sharpe:.2f}, 年化收益={cagr:.2f}")

                    # 使用加载的模型预测
                    states, proba = loaded_model.predict(Xz)
                else:
                    st.error("模型加载失败，请选择其他模型")
                    # 回退到默认模型
                    detector = HMMRegimeDetector(
                        n_states=best_params.get("n_states", 3),
                        covariance_type=best_params.get("covariance_type", "diag"),
                        n_iter=best_params.get("n_iter", 100),
                    )
                    detector.fit(Xz)
                    states, proba = detector.predict(
                        Xz, min_len=min_len, sticky_strength=stickiness
                    )
            else:
                st.warning("没有找到保存的模型，使用默认参数训练新模型")
                # 回退到默认模型
                detector = HMMRegimeDetector(
                    n_states=best_params.get("n_states", 3),
                    covariance_type=best_params.get("covariance_type", "diag"),
                    n_iter=best_params.get("n_iter", 100),
                )
                detector.fit(Xz)
                states, proba = detector.predict(
                    Xz, min_len=min_len, sticky_strength=stickiness
                )
        else:
            # 执行和结果（训练、标记、绘图、简单回测）
            # 在标准化特征上训练 HMM
            global_logger.info(
                f"开始训练HMM模型，状态数量: {best_params.get('n_states', 3)}"
            )
            detector = HMMRegimeDetector(
                n_states=best_params.get("n_states", 3),
                covariance_type=best_params.get("covariance_type", "diag"),
                n_iter=best_params.get("n_iter", 100),
            )
            detector.fit(Xz)
            global_logger.info("HMM模型训练完成")

            # 使用粘性和最小持续时间平滑预测状态
            global_logger.info(
                f"开始预测市场状态，最小持续时间: {min_len}, 状态粘性: {stickiness}"
            )
            states, proba = detector.predict(
                Xz, min_len=min_len, sticky_strength=stickiness
            )
            global_logger.info(
                f"市场状态预测完成，共识别 {len(np.unique(states))} 种状态"
            )

        # 组装输出数据框
        out = df.copy()
        out["log_ret"] = np.log(df["PX"]).diff().fillna(0.0)
        out["state"] = states

        # 按平均收益率排序状态并映射到市场状态
        state_means = (
            out.groupby("state")["log_ret"].mean().sort_values(ascending=False)
        )
        ranked = state_means.index.tolist()
        labels = {ranked[0]: "Bull", ranked[-1]: "Bear"}
        for s in set(range(best_params.get("n_states", 3))) - set(labels):
            labels[s] = "Neutral"
        out["regime"] = out["state"].map(labels)

        # 记录状态映射
        global_logger.info(f"状态映射完成: {labels}")

        # 简单状态交易回测（仅用于直观理解，非执行级别）
        # 牛市做多，熊市做空，中性市场观望；第二天执行交易（移位持仓）
        global_logger.info("开始生成交易信号")
        out["position"] = 0
        out.loc[out["regime"] == "Bull", "position"] = 1
        out.loc[out["regime"] == "Bear", "position"] = -1

        # 应用次日执行
        out["position"] = out["position"].shift(1).fillna(0)
        global_logger.info("交易信号生成完成")

        # 策略对数收益率和累积增长（对数/指数用于数值稳定性）
        out["strat_lr"] = out["position"] * out["log_ret"]
        cum = np.exp(out[["log_ret", "strat_lr"]].cumsum())
        cum.columns = ["BuyHold", "HMM_Strategy"]

        # 简单指标计算
        def sharpe(x, periods=252):
            """计算夏普比率

            Args:
                x: 收益率序列
                periods: 年化因子，默认252（日度数据）

            Returns:
                夏普比率值
            """
            mu, sd = x.mean(), x.std()
            return (mu / sd) * np.sqrt(periods) if sd > 0 else np.nan

        def max_drawdown(series):
            """计算最大回撤

            Args:
                series: 资产价格或净值序列

            Returns:
                最大回撤值
            """
            rollmax = series.cummax()
            dd = series / rollmax - 1.0
            return dd.min()

        bh_cagr = cum["BuyHold"].iloc[-1] ** (252 / len(out)) - 1
        st_cagr = cum["HMM_Strategy"].iloc[-1] ** (252 / len(out)) - 1
        bh_sharp = sharpe(out["log_ret"])
        st_sharp = sharpe(out["strat_lr"])
        bh_mdd = max_drawdown(cum["BuyHold"])
        st_mdd = max_drawdown(cum["HMM_Strategy"])

        # 记录策略表现
        global_logger.info(
            f"策略表现计算完成: \
            买入持有年化收益: {bh_cagr:.2%}, \
            HMM策略年化收益: {st_cagr:.2%}, \
            买入持有夏普比率: {bh_sharp:.2f}, \
            HMM策略夏普比率: {st_sharp:.2f}"
        )

        # 显示关键指标
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("当前市场状态", out["regime"].iloc[-1])
        col2.metric("光大银行价格", f"{out['PX'].iloc[-1]:.2f}")
        col3.metric("股债利差", f"{out['EBS'].iloc[-1]:.2f}%")
        col4.metric("巴菲特指数", f"{out['BUFFETT'].iloc[-1]:.2f}")

        # 记录当前市场状态
        global_logger.log_market_state(
            timestamp=out.index[-1],
            state=out["regime"].iloc[-1],
            confidence=proba[-1, states[-1]] if len(proba) > 0 else None,
        )

        # 创建选项卡显示不同图表
        tab1, tab2, tab3, tab4 = st.tabs(
            ["光大银行走势", "指标分析", "策略表现", "状态统计"]
        )

        with tab1:
            # 光大银行走势与市场状态
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("光大银行价格", "市场状态"),
            )

            # 添加价格线
            fig.add_trace(
                go.Scatter(
                    x=out.index,
                    y=out["PX"],
                    name="光大银行",
                    line=dict(color="#1f77b4"),
                ),
                row=1,
                col=1,
            )

            # 添加市场状态背景
            colors = {"Bull": "#2ca02c", "Bear": "#d62728", "Neutral": "#ff7f0e"}
            prev_regime = None
            start_idx = out.index[0]

            for i, (date, regime) in enumerate(out["regime"].items()):
                if prev_regime is None:
                    prev_regime = regime
                    continue

                if regime != prev_regime:
                    # 添加矩形区域
                    fig.add_vrect(
                        x0=start_idx,
                        x1=date,
                        fillcolor=colors[prev_regime],
                        opacity=0.2,
                        line_width=0,
                        row=1,
                        col=1,
                    )
                    fig.add_vrect(
                        x0=start_idx,
                        x1=date,
                        fillcolor=colors[prev_regime],
                        opacity=0.2,
                        line_width=0,
                        row=2,
                        col=1,
                    )
                    start_idx = date
                    prev_regime = regime

            # 添加最后一个区域
            fig.add_vrect(
                x0=start_idx,
                x1=out.index[-1],
                fillcolor=colors[prev_regime],
                opacity=0.2,
                line_width=0,
                row=1,
                col=1,
            )
            fig.add_vrect(
                x0=start_idx,
                x1=out.index[-1],
                fillcolor=colors[prev_regime],
                opacity=0.2,
                line_width=0,
                row=2,
                col=1,
            )

            # 添加状态标签
            for regime in ["Bull", "Bear", "Neutral"]:
                regime_data = out[out["regime"] == regime]
                if not regime_data.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=regime_data.index,
                            y=[regime] * len(regime_data),
                            mode="markers",
                            name=regime,
                            marker=dict(color=colors[regime], size=5),
                        ),
                        row=2,
                        col=1,
                    )

            fig.update_layout(
                height=600, showlegend=True, title_text="光大银行走势与市场状态识别"
            )
            fig.update_yaxes(title_text="价格", row=1, col=1)
            fig.update_yaxes(title_text="市场状态", row=2, col=1)
            fig.update_xaxes(title_text="日期", row=2, col=1)

            st.plotly_chart(fig, width="stretch")

        with tab2:
            # 指标分析
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=("股债利差", "巴菲特指数", "波动率", "趋势指标"),
            )

            # 股债利差
            fig.add_trace(
                go.Scatter(
                    x=out.index,
                    y=out["EBS"],
                    name="股债利差",
                    line=dict(color="#17becf"),
                ),
                row=1,
                col=1,
            )

            # 巴菲特指数
            fig.add_trace(
                go.Scatter(
                    x=out.index,
                    y=out["BUFFETT"],
                    name="巴菲特指数",
                    line=dict(color="#e377c2"),
                ),
                row=1,
                col=2,
            )

            # 波动率
            fig.add_trace(
                go.Scatter(
                    x=out.index, y=out["VOL"], name="波动率", line=dict(color="#7f7f7f")
                ),
                row=2,
                col=1,
            )

            # 趋势指标
            fig.add_trace(
                go.Scatter(
                    x=out.index,
                    y=out["SPREAD"],
                    name="趋势指标",
                    line=dict(color="#bcbd22"),
                ),
                row=2,
                col=2,
            )

            fig.update_layout(height=600, showlegend=True, title_text="市场指标分析")
            st.plotly_chart(fig, width="stretch")

        with tab3:
            # 策略表现对比
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=cum.index,
                    y=cum["BuyHold"],
                    name="买入持有",
                    line=dict(color="#1f77b4"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=cum.index,
                    y=cum["HMM_Strategy"],
                    name="HMM策略",
                    line=dict(color="#ff7f0e"),
                )
            )

            fig.update_layout(
                title="策略表现对比",
                xaxis_title="日期",
                yaxis_title="累计收益",
                hovermode="x unified",
                height=500,
            )

            st.plotly_chart(fig, width="stretch")

            # 显示性能指标
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("买入持有年化收益", f"{bh_cagr:.2%}")
            col2.metric("HMM策略年化收益", f"{st_cagr:.2%}")
            col3.metric("买入持有夏普比率", f"{bh_sharp:.2f}")
            col4.metric("HMM策略夏普比率", f"{st_sharp:.2f}")

            col5, col6 = st.columns(2)
            col5.metric("买入持有最大回撤", f"{bh_mdd:.2%}")
            col6.metric("HMM策略最大回撤", f"{st_mdd:.2%}")

            # 显示最近交易信号
            st.subheader("最近交易信号")
            recent_signals = out[["PX", "regime", "position"]].tail(10)
            st.dataframe(
                recent_signals.style.map(
                    lambda x: (
                        "background-color: #2ca02c"
                        if x == "Bull"
                        else (
                            "background-color: #d62728"
                            if x == "Bear"
                            else "background-color: #ff7f0e"
                        )
                    ),
                    subset=["regime"],
                )
            )

        with tab4:
            # 状态统计
            regime_counts = out["regime"].value_counts()
            regime_returns = out.groupby("regime")["log_ret"].mean()
            regime_volatility = out.groupby("regime")["log_ret"].std()

            global_logger.info(f"市场状态分布: {dict(regime_counts)}")
            global_logger.info(f"各状态平均收益: {dict(regime_returns)}")
            global_logger.info(f"各状态波动率: {dict(regime_volatility)}")

            col1, col2 = st.columns(2)

            with col1:
                # 修复饼图问题：将Series转换为列表
                fig = px.pie(
                    values=regime_counts.values.tolist(),
                    names=regime_counts.index.tolist(),
                    title="市场状态分布",
                    color=regime_counts.index.tolist(),
                    color_discrete_map={
                        "Bull": "#2ca02c",
                        "Bear": "#d62728",
                        "Neutral": "#ff7f0e",
                    },
                )
                st.plotly_chart(fig, width="stretch")

            with col2:
                fig = go.Figure(
                    data=[
                        go.Bar(
                            name="平均收益",
                            x=regime_returns.index.tolist(),
                            y=regime_returns.values.tolist(),
                            marker_color=["#2ca02c", "#d62728", "#ff7f0e"],
                        ),
                        go.Bar(
                            name="波动率",
                            x=regime_volatility.index.tolist(),
                            y=regime_volatility.values.tolist(),
                            marker_color=["#1f77b4", "#1f77b4", "#1f77b4"],
                        ),
                    ]
                )
                fig.update_layout(title="各状态收益与波动率", barmode="group")
                st.plotly_chart(fig, width="stretch")

            # 显示状态转换矩阵
            st.subheader("状态转换矩阵")
            # 计算状态转换
            transitions = []
            prev_state = None
            for state in out["regime"]:
                if prev_state is not None and prev_state != state:
                    transitions.append((prev_state, state))
                prev_state = state

            global_logger.info(f"共检测到 {len(transitions)} 次状态转换")

            # 创建转换矩阵
            if transitions:
                transition_df = pd.DataFrame(transitions, columns=["From", "To"])
                transition_matrix = pd.crosstab(
                    transition_df["From"], transition_df["To"], normalize="index"
                )
                st.dataframe(transition_matrix.style.background_gradient(cmap="Blues"))
            else:
                st.info("没有检测到状态转换")

        # 模型保存功能
        st.subheader("模型管理")

        # 生成性能指标
        performance_metrics = {
            "cagr": st_cagr,
            "sharpe": st_sharp,
            "mdd": st_mdd,
            "params": {
                **best_params,
                "min_duration": min_len,
                "sticky_strength": stickiness,
            },
        }

        # 比较当前模型与已保存模型
        comparison_result = model_manager.compare_models(performance_metrics)

        if comparison_result["should_save"]:
            st.success("当前模型性能优于已保存模型，建议保存！")
        else:
            st.info(f"当前模型夏普比率: {comparison_result['current_sharpe']:.2f}")
            st.info(f"最佳模型夏普比率: {comparison_result['best_model_sharpe']:.2f}")
            st.info(f"改进幅度: {comparison_result['improvement']:.2f}")

        # 保存模型按钮
        if st.button("保存当前模型"):
            model_name = model_manager.save_model(detector, performance_metrics)
            st.success(f"模型保存成功！模型名称: {model_name}")

        # 显示已保存的模型
        with st.expander("查看已保存的模型"):
            saved_models = model_manager.list_saved_models()
            if saved_models:
                saved_models_df = pd.DataFrame(saved_models)
                st.dataframe(saved_models_df)
            else:
                st.info("暂无保存的模型")

        # 显示原始数据
        if st.sidebar.checkbox("显示原始数据"):
            st.subheader("原始数据")
            st.dataframe(out)

        # 添加说明
        st.sidebar.markdown("---")
        st.sidebar.info(
            """
        **策略说明**:
        - 使用隐马尔可夫模型识别光大银行市场状态
        - 牛市做多，熊市做空，中性观望
        - 基于光大银行股价、股债利差、巴菲特指数等多因子
        - 数据来源：baostock（股票数据）+ akshare（宏观数据）
        - 支持参数优化、特征选择和平滑参数优化
        - 支持模型保存、加载和比较
        """
        )

        # 记录应用运行完成
        global_logger.info("光大银行市场状态识别系统(优化版)运行完成")


if __name__ == "__main__":
    # 检查依赖
    missing_packages = check_dependencies()

    if missing_packages:
        print("错误: 缺少以下依赖包:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装:")
        print("pip install", " ".join(missing_packages))
        sys.exit(1)

    # 设置环境
    setup_environment()

    # 运行主程序
    main()
