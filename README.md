# 光大银行市场状态识别系统

基于隐马尔可夫模型(HMM)的银行股市场状态识别与交易策略应用。系统使用baostock获取光大银行历史数据，结合akshare获取的宏观数据，实现完整的市场状态识别与交易策略回测功能。

## 功能特性

- **数据获取**：使用baostock获取光大银行(601818)历史股票数据
- **宏观指标**：集成akshare获取的股债利差、巴菲特指数等宏观数据
- **状态识别**：基于HMM模型识别市场状态（牛市、熊市、中性）
- **交易策略**：根据市场状态生成多空交易信号
- **参数优化**：支持网格搜索和贝叶斯优化自动调参
- **特征选择**：自动评估和选择最优特征组合
- **性能评估**：多维度策略性能评估和风险指标计算
- **交互界面**：完整的Streamlit界面，支持参数配置和可视化展示
- **模型管理**：支持模型保存、加载和性能比较

## 技术架构

### 核心模块

1. **数据获取模块**：baostock股票数据 + akshare宏观数据
2. **特征工程模块**：技术指标计算 + 宏观指标整合
3. **HMM模型模块**：市场状态识别与平滑处理
4. **参数优化模块**：网格搜索 + 贝叶斯优化自动调参
5. **特征选择模块**：特征重要性评估 + 最优特征组合选择
6. **性能评估模块**：多维度策略评估 + 风险指标计算
7. **交易策略模块**：多空信号生成与收益计算
8. **界面交互模块**：Streamlit界面与可视化展示
9. **模型管理模块**：模型持久化与性能跟踪

### 技术栈

- **前端框架**：Streamlit (Python)
- **数据处理**：pandas, numpy
- **机器学习**：hmmlearn (HMM模型)
- **优化算法**：scikit-optimize (贝叶斯优化)
- **数据源**：baostock (股票数据)、akshare (宏观数据)
- **可视化**：plotly, matplotlib
- **模型持久化**：joblib, json
- **并行处理**：tqdm (进度显示)

## 安装与运行

### 环境要求

- Python 3.8+
- 以下Python包：
  ```
  pip install baostock akshare hmmlearn pandas numpy matplotlib streamlit plotly joblib scikit-optimize tqdm
  ```

### 运行应用

```bash
streamlit run app.py
```

应用将在 `http://localhost:8501` 启动。

### 运行优化版应用

```bash
streamlit run optimized_hmm_app.py
```

优化版应用包含更多高级功能和性能优化。

## 使用说明

### 参数配置

在侧边栏可以配置以下参数：

- **选择资产**：当前仅支持光大银行(601818)
- **开始日期**：数据获取的开始时间
- **状态数量**：HMM模型识别的状态数量（2-6）
- **最小状态持续时间**：状态平滑处理的最小天数（5-30）
- **状态粘性**：状态转移的粘性强度（1.0-20.0）
- **银行股专用参数**：波动率窗口、短期/长期均线窗口

### 参数优化

系统提供两种参数优化方法：

#### 1. 网格搜索优化
- **参数范围**：可配置状态数量、协方差类型、迭代次数等
- **交叉验证**：使用时间序列交叉验证避免过拟合
- **评估指标**：基于模型对数似然值评估

#### 2. 贝叶斯优化
- **智能搜索**：基于高斯过程的贝叶斯优化
- **自动调参**：自动探索最优参数组合
- **效率提升**：相比网格搜索更高效，收敛更快

#### 3. 特征选择
- **特征评估**：评估每个特征的重要性
- **自动选择**：选择最优特征组合
- **性能提升**：减少噪音特征，提高模型精度

### 优化结果

优化完成后，系统会提供：
- **最佳参数组合**：详细的参数配置
- **优化前后对比**：性能提升幅度
- **特征重要性**：各特征的贡献度
- **优化报告**：完整的优化过程分析

### 使用方法

1. **手动配置参数**：通过侧边栏直接调整参数
2. **网格搜索**：使用`HMMRegimeDetector.optimize_parameters()`方法
3. **贝叶斯优化**：使用`HMMRegimeDetector.optimize_with_bayesian()`方法
4. **特征选择**：使用`HMMRegimeDetector.optimize_features()`方法

示例代码：

```python
# 使用贝叶斯优化获取最佳参数
detector = HMMRegimeDetector()
opt_result = detector.optimize_with_bayesian(X, n_calls=50)
best_params = opt_result['best_params']

# 使用最佳参数重新训练模型
detector = HMMRegimeDetector(
    n_states=best_params['n_states'],
    covariance_type=best_params['covariance_type'],
    n_iter=best_params['n_iter']
)
detector.fit(X)

# 预测市场状态
states, proba = detector.predict(
    X,
    min_len=best_params['min_duration'],
    sticky_strength=best_params['sticky_strength']
)
```

### 功能界面

应用包含以下主要界面：

1. **光大银行走势**：价格走势与市场状态可视化
2. **指标分析**：股债利差、巴菲特指数、波动率等指标分析
3. **策略表现**：买入持有与HMM策略对比
4. **状态统计**：市场状态分布与转换矩阵

### 模型管理

- **自动保存**：开启后可自动保存表现更好的模型
- **模型比较**：与历史最佳模型进行性能对比
- **模型列表**：查看所有已保存模型的性能指标

## 数据处理流程

1. **数据获取**：从baostock获取光大银行数据，从akshare获取宏观数据
2. **特征计算**：计算对数收益率、波动率、趋势指标等
3. **数据对齐**：将宏观数据与股票数据日期对齐
4. **特征标准化**：Z-分数标准化提高状态区分度
5. **HMM训练**：训练隐马尔可夫模型识别市场状态
6. **状态平滑**：应用最小持续时间和粘性平滑
7. **策略生成**：根据市场状态生成交易信号
8. **性能评估**：计算年化收益、夏普比率、最大回撤等指标

## 算法原理

### HMM市场状态识别

使用高斯隐马尔可夫模型(Gaussian HMM)识别市场状态：

- **观测变量**：对数收益率、波动率、趋势指标、股债利差、巴菲特指数
- **隐藏状态**：市场状态（牛市/熊市/中性）
- **状态平滑**：最小持续时间约束和状态粘性优化

### 交易策略

- **牛市状态**：做多（持仓100%）
- **熊市状态**：做空（持仓-100%）
- **中性状态**：观望（持仓0%）
- **执行时机**：次日开盘执行交易信号

## 性能指标

- **年化收益率(CAGR)**：策略的年化复合增长率
- **夏普比率(Sharpe Ratio)**：风险调整后的收益指标
- **最大回撤(Max Drawdown)**：策略的最大亏损幅度
- **状态分布**：各市场状态的持续时间分布
- **转换矩阵**：市场状态之间的转换概率

## 文件结构

```
光大银行市场状态识别/
├── app.py                              # 主程序文件
├── optimized_hmm_app.py                # 优化版HMM应用
├── find_missing_docstrings.py          # 文档字符串检查脚本
├── performance_test.py                 # 性能测试脚本
├── README.md                           # 项目说明文档
├── setup.py                            # 项目安装配置
├── requirements.txt                    # 依赖包列表
├── pytest.ini                          # pytest配置文件
├── mypy.ini                            # mypy类型检查配置
├── CODE_QUALITY_GUIDE.md               # 代码质量指南
├── auto_examples_python/               # 自动示例代码
├── cache/                              # 缓存目录
├── logs/                               # 日志目录
├── models/                             # 模型保存目录
│   ├── *.joblib                        # HMM模型文件
│   └── *_metrics.json                  # 模型性能指标
├── src/                                # 源代码目录
│   ├── __init__.py                     # 包初始化文件
│   ├── config/                         # 配置模块
│   │   ├── __init__.py
│   │   └── config.py                   # 配置文件
│   ├── data/                           # 数据模块
│   │   ├── __init__.py
│   │   ├── data_fetcher.py             # 数据获取器
│   │   └── data_processor.py           # 数据处理器
│   ├── features/                       # 特征模块
│   │   ├── __init__.py
│   │   ├── feature_engineer.py         # 特征工程
│   │   └── technical_indicators.py     # 技术指标计算
│   ├── models/                         # 模型模块
│   │   ├── __init__.py
│   │   ├── hmm_regime_detector.py      # HMM模型核心实现
│   │   ├── hmm_optimizer.py            # 参数优化模块
│   │   ├── bayesian_optimizer.py       # 贝叶斯优化实现
│   │   ├── cross_validation.py         # 交叉验证实现
│   │   ├── performance_evaluator.py    # 性能评估模块
│   │   ├── market_state_analyzer.py    # 市场状态分析器
│   │   ├── model_manager.py            # 模型管理器
│   │   ├── model_selector.py           # 模型选择器
│   │   └── optimizer_runner.py         # 优化运行器
│   ├── strategies/                     # 策略模块
│   │   ├── __init__.py
│   │   ├── trading_strategy.py         # 交易策略
│   │   └── performance_metrics.py      # 性能指标
│   ├── utils/                          # 工具模块
│   │   ├── __init__.py
│   │   ├── helpers.py                  # 辅助函数
│   │   └── logger.py                   # 日志工具
│   └── visualization/                  # 可视化模块
│       ├── __init__.py
│       ├── chart_generator.py          # 图表生成器
│       └── streamlit_app.py            # Streamlit应用
└── tests/                              # 测试目录
    ├── __init__.py
    ├── conftest.py                     # 测试配置
    ├── integration/                    # 集成测试
    │   ├── __init__.py
    │   ├── test_data_integration.py
    │   ├── test_end_to_end_workflow.py
    │   └── test_system_robustness.py
    └── unit/                           # 单元测试
        ├── __init__.py
        ├── test_app.py
        ├── test_config.py
        ├── test_cross_validation.py
        ├── test_data_fetcher.py
        ├── test_feature_engineer.py
        ├── test_hmm_optimizer.py
        ├── test_hmm_regime_detector.py
        ├── test_market_state_analyzer.py
        ├── test_model_selector.py
        ├── test_optimizer_runner.py
        ├── test_performance_evaluator.py
        ├── test_technical_indicators.py
        ├── test_trading_strategy.py
        ├── test_visualization.py
        └── 其他测试文件
```

## 注意事项

1. **数据源稳定性**：baostock和akshare的数据获取依赖于网络连接
2. **模型参数**：不同参数设置会影响状态识别效果，建议通过回测优化
3. **历史数据**：策略表现基于历史数据，不代表未来收益
4. **风险提示**：投资有风险，决策需谨慎

## 开发计划

### 已完成

- ✅ 优化HMM模型参数选择
- ✅ 添加特征选择功能
- ✅ 实现自动参数调优系统
- ✅ 完善性能评估体系
- ✅ 建立完整的测试套件

### 未来计划

- [ ] 支持更多银行股分析
- [ ] 添加更多技术指标
- [ ] 增加实时数据更新
- [ ] 添加风险控制模块
- [ ] 实现模型集成学习
- [ ] 开发API接口
- [ ] 支持更多资产类别分析
- [ ] 增加量化因子库
- [ ] 开发策略回测平台
- [ ] 实现自动交易接口

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题或建议，请通过GitHub Issues反馈。