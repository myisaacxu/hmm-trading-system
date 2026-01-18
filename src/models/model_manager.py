"""
模型管理模块 - 负责模型保存、加载和比较
"""

import os
import json
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class ModelManager:
    """模型管理器类 - 负责模型的保存、加载、删除和比较"""

    def __init__(self, models_dir: str = "models"):
        """
        初始化模型管理器

        Args:
            models_dir: 模型保存目录
        """
        if not isinstance(models_dir, str) or not models_dir:
            raise ValueError("models_dir必须是有效的字符串路径")

        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

    def save_model(
        self, model, performance_metrics: Dict, model_name: Optional[str] = None
    ) -> str:
        """
        保存模型和性能指标

        Args:
            model: 要保存的模型对象
            performance_metrics: 性能指标字典
            model_name: 模型名称，如果为None则自动生成

        Returns:
            保存的模型名称
        """
        if model is None:
            raise ValueError("model不能为空")

        if not isinstance(performance_metrics, dict):
            raise ValueError("performance_metrics必须是字典类型")

        # 自动生成模型名称
        if model_name is None:
            model_name = f"hmm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 清理模型名称
        model_name = model_name.strip()
        if not model_name:
            raise ValueError("model_name不能为空")

        # 确保目录存在
        os.makedirs(self.models_dir, exist_ok=True)

        # 构建文件路径
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        metrics_path = os.path.join(self.models_dir, f"{model_name}_metrics.json")

        # 保存模型
        joblib.dump(model, model_path)

        # 保存性能指标，添加创建日期
        metrics_with_date = performance_metrics.copy()
        metrics_with_date["created_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_with_date, f, indent=2, ensure_ascii=False)

        return model_name

    def load_model(self, model_name: str) -> Tuple[Optional[object], Optional[Dict]]:
        """
        加载模型和性能指标

        Args:
            model_name: 模型名称

        Returns:
            模型对象和性能指标字典
        """
        if not isinstance(model_name, str) or not model_name:
            return None, None

        # 构建文件路径
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        metrics_path = os.path.join(self.models_dir, f"{model_name}_metrics.json")

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            return None, None

        try:
            # 加载模型
            model = joblib.load(model_path)

            # 加载性能指标
            metrics = {}
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)

            return model, metrics
        except Exception:
            # 处理加载错误
            return None, None

    def list_saved_models(self) -> List[Dict]:
        """
        列出所有保存的模型

        Returns:
            模型信息列表，按夏普比率降序排序
        """
        if not os.path.exists(self.models_dir):
            return []

        models = []

        try:
            for file in os.listdir(self.models_dir):
                if file.endswith(".joblib"):
                    model_name = file.replace(".joblib", "")
                    metrics_file = os.path.join(
                        self.models_dir, f"{model_name}_metrics.json"
                    )

                    if os.path.exists(metrics_file):
                        try:
                            with open(metrics_file, "r", encoding="utf-8") as f:
                                metrics = json.load(f)

                            models.append(
                                {
                                    "name": model_name,
                                    "cagr": metrics.get("cagr", 0.0),
                                    "sharpe": metrics.get("sharpe", 0.0),
                                    "mdd": metrics.get("mdd", 0.0),
                                    "date": metrics.get("created_date", ""),
                                    "params": metrics.get("params", {}),
                                }
                            )
                        except (json.JSONDecodeError, KeyError, TypeError):
                            # 跳过损坏的指标文件
                            continue
        except OSError:
            # 处理目录访问错误
            return []

        # 按夏普比率降序排序
        models.sort(key=lambda x: x["sharpe"], reverse=True)
        return models

    def delete_model(self, model_name: str) -> bool:
        """
        删除模型文件

        Args:
            model_name: 模型名称

        Returns:
            是否成功删除
        """
        if not isinstance(model_name, str) or not model_name:
            return False

        # 构建文件路径
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        metrics_path = os.path.join(self.models_dir, f"{model_name}_metrics.json")

        try:
            # 删除模型文件
            if os.path.exists(model_path):
                os.remove(model_path)
            # 删除指标文件
            if os.path.exists(metrics_path):
                os.remove(metrics_path)
            return True
        except OSError:
            # 处理删除错误
            return False

    def compare_models(self, current_performance: Dict, threshold: float = 0.1) -> Dict:
        """
        比较当前模型与已保存模型

        Args:
            current_performance: 当前模型性能指标
            threshold: 保存阈值，默认0.1

        Returns:
            比较结果字典
        """
        if not isinstance(current_performance, dict):
            raise ValueError("current_performance必须是字典类型")

        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold必须是数值类型")

        saved_models = self.list_saved_models()
        current_sharpe = current_performance.get("sharpe", 0.0)

        if not saved_models:
            return {
                "has_better_model": False,
                "best_model_sharpe": 0.0,
                "current_sharpe": current_sharpe,
                "improvement": float("inf"),
                "should_save": True,
            }

        # 获取最佳模型（按夏普比率排序）
        best_model = saved_models[0]
        best_sharpe = best_model["sharpe"]
        improvement = current_sharpe - best_sharpe

        return {
            "has_better_model": current_sharpe > best_sharpe + threshold,
            "best_model_sharpe": best_sharpe,
            "current_sharpe": current_sharpe,
            "improvement": improvement,
            "should_save": improvement > threshold,
        }

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """
        获取模型详细信息

        Args:
            model_name: 模型名称

        Returns:
            模型详细信息字典
        """
        if not isinstance(model_name, str) or not model_name:
            return None

        model, metrics = self.load_model(model_name)

        if model is None:
            return None

        return {
            "name": model_name,
            "model_type": type(model).__name__,
            "metrics": metrics,
            "file_size": self._get_file_size(model_name),
            "created_date": self._get_creation_date(model_name),
        }

    def _get_file_size(self, model_name: str) -> int:
        """获取模型文件大小"""
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        if os.path.exists(model_path):
            return os.path.getsize(model_path)
        return 0

    def _get_creation_date(self, model_name: str) -> str:
        """获取模型创建日期"""
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        if os.path.exists(model_path):
            try:
                timestamp = os.path.getctime(model_path)
                return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            except OSError:
                return ""
        return ""
