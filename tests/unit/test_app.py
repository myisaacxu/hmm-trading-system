#!/usr/bin/env python3
"""
app.py 单元测试
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

# 导入要测试的函数
from app import (
    check_dependencies,
    setup_environment,
    get_ebs_data,
    get_buffett_index,
    _calculate_buffett_index_fallback,
    get_cebbank_data,
    calculate_technical_indicators,
)


class TestAppFunctions(unittest.TestCase):
    """测试app.py中的函数"""

    def test_check_dependencies(self):
        """测试检查依赖包函数"""
        # 测试所有依赖包都已安装的情况
        missing_packages = check_dependencies()
        # 由于我们在测试环境中运行，所有依赖包应该都已安装
        # 但为了测试的健壮性，我们只检查返回值是否为列表
        self.assertIsInstance(missing_packages, list)

    @patch("app.st")
    def test_get_ebs_data_success(self, mock_st):
        """测试成功获取股债利差数据"""
        with patch("app.ak.stock_ebs_lg") as mock_stock_ebs_lg:
            # 模拟返回非空数据
            mock_df = pd.DataFrame(
                {"日期": ["2023-01-01", "2023-01-02"], "股债利差": [0.5, 0.6]}
            )
            mock_stock_ebs_lg.return_value = mock_df

            result = get_ebs_data()

            # 检查返回值类型
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(len(result), 2)

    @patch("app.st")
    def test_get_ebs_data_empty(self, mock_st):
        """测试获取股债利差数据为空的情况"""
        with patch("app.ak.stock_ebs_lg") as mock_stock_ebs_lg:
            # 模拟返回空数据
            mock_stock_ebs_lg.return_value = pd.DataFrame()

            result = get_ebs_data()

            # 检查返回值
            self.assertIsNone(result)
            mock_st.error.assert_called_once()

    @patch("app.st")
    def test_get_ebs_data_exception(self, mock_st):
        """测试获取股债利差数据发生异常的情况"""
        with patch("app.ak.stock_ebs_lg") as mock_stock_ebs_lg:
            # 模拟发生异常
            mock_stock_ebs_lg.side_effect = Exception("Test error")

            result = get_ebs_data()

            # 检查返回值
            self.assertIsNone(result)
            mock_st.error.assert_called_once()

    @patch("app.st")
    def test_get_buffett_index_success(self, mock_st):
        """测试成功获取巴菲特指数数据"""
        with patch("app.ak.stock_buffett_index_lg") as mock_stock_buffett_index_lg:
            # 模拟返回非空数据
            mock_df = pd.DataFrame(
                {
                    "日期": ["2023-01-01"],
                    "总市值": [100000000000000],
                    "GDP": [10000000000000],
                }
            )
            mock_stock_buffett_index_lg.return_value = mock_df

            result = get_buffett_index()

            # 检查返回值类型
            self.assertIsInstance(result, pd.Series)

    @patch("app.st")
    def test_get_buffett_index_empty(self, mock_st):
        """测试获取巴菲特指数数据为空的情况"""
        with patch("app.ak.stock_buffett_index_lg") as mock_stock_buffett_index_lg:
            # 模拟返回空数据
            mock_stock_buffett_index_lg.return_value = pd.DataFrame()

        with patch("app._calculate_buffett_index_fallback") as mock_fallback:
            # 模拟备用方法返回值
            mock_fallback.return_value = pd.Series(
                [100], index=pd.date_range("2023-01-01", periods=1)
            )

            result = get_buffett_index()

            # 检查返回值类型
            self.assertIsInstance(result, pd.Series)

    @patch("app.st")
    def test_get_buffett_index_exception(self, mock_st):
        """测试获取巴菲特指数数据发生异常的情况"""
        with patch("app.ak.stock_buffett_index_lg") as mock_stock_buffett_index_lg:
            # 模拟发生异常
            mock_stock_buffett_index_lg.side_effect = Exception("Test error")

        with patch("app._calculate_buffett_index_fallback") as mock_fallback:
            # 模拟备用方法返回值
            mock_fallback.return_value = pd.Series(
                [100], index=pd.date_range("2023-01-01", periods=1)
            )

            result = get_buffett_index()

            # 检查返回值类型
            self.assertIsInstance(result, pd.Series)

    @patch("app.st")
    @patch("app.bs")
    def test_get_cebbank_data_success(self, mock_bs, mock_st):
        """测试成功获取光大银行数据"""
        # 模拟bs.login返回值
        mock_bs.login.return_value = None

        # 模拟查询结果
        mock_rs = MagicMock()
        mock_rs.error_code = "0"
        mock_rs.fields = [
            "date",
            "code",
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

        # 模拟next方法和get_row_data方法
        def mock_next():
            """模拟查询结果的next方法
            
            Returns:
                bool: 指示是否还有下一行数据
            """
            if not hasattr(mock_next, "called"):
                mock_next.called = 0
            mock_next.called += 1
            return mock_next.called <= 2

        mock_rs.next = mock_next

        def mock_get_row_data():
            """模拟查询结果的get_row_data方法
            
            Returns:
                list: 包含股票数据的列表
            """
            if mock_next.called == 1:
                return [
                    "2023-01-01",
                    "sh.601818",
                    "10.0",
                    "10.5",
                    "9.5",
                    "10.2",
                    "10.0",
                    "1000000",
                    "10200000",
                    "1.0",
                    "2.0",
                ]
            else:
                return [
                    "2023-01-02",
                    "sh.601818",
                    "10.2",
                    "10.8",
                    "10.0",
                    "10.5",
                    "10.2",
                    "1200000",
                    "12600000",
                    "1.2",
                    "2.9",
                ]

        mock_rs.get_row_data = mock_get_row_data

        mock_bs.query_history_k_data_plus.return_value = mock_rs
        mock_bs.logout.return_value = None

        result = get_cebbank_data("2023-01-01", "2023-01-02")

        # 检查返回值类型
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    @patch("app.st")
    @patch("app.bs")
    def test_get_cebbank_data_empty(self, mock_bs, mock_st):
        """测试获取光大银行数据为空的情况"""
        # 模拟bs.login返回值
        mock_bs.login.return_value = None

        # 模拟查询结果
        mock_rs = MagicMock()
        mock_rs.error_code = "0"
        mock_rs.fields = [
            "date",
            "code",
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

        # 模拟next方法返回False
        mock_rs.next.return_value = False
        mock_rs.get_row_data.return_value = []

        mock_bs.query_history_k_data_plus.return_value = mock_rs
        mock_bs.logout.return_value = None

        result = get_cebbank_data("2023-01-01", "2023-01-02")

        # 检查返回值
        self.assertIsNone(result)
        mock_st.error.assert_called_once()

    @patch("app.st")
    @patch("app.bs")
    def test_get_cebbank_data_exception(self, mock_bs, mock_st):
        """测试获取光大银行数据发生异常的情况"""
        # 模拟bs.login返回值
        mock_bs.login.return_value = None

        # 模拟查询发生异常
        mock_bs.query_history_k_data_plus.side_effect = Exception("Test error")
        mock_bs.logout.return_value = None

        result = get_cebbank_data("2023-01-01", "2023-01-02")

        # 检查返回值
        self.assertIsNone(result)
        mock_st.error.assert_called_once()

    def test_calculate_technical_indicators(self):
        """测试计算技术指标函数"""
        # 创建测试数据
        dates = pd.date_range("2023-01-01", periods=30)
        prices = pd.Series(np.random.randn(30) + 10, index=dates)

        result = calculate_technical_indicators(prices)

        # 检查返回值类型
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 30)


if __name__ == "__main__":
    unittest.main()
