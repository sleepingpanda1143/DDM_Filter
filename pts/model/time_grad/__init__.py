try:
    from .estimator import TimeGradEstimator
except ImportError:
    from .time_grad_estimator import TimeGradEstimator

try:
    from .module import TimeGradModel
except ImportError:
    TimeGradModel = None  # type: ignore[misc, assignment]

try:
    from .lightning_module import TimeGradLightningModule
except ImportError:
    TimeGradLightningModule = None  # type: ignore[misc, assignment]

__all__ = ["TimeGradEstimator"]
if TimeGradModel is not None:
    __all__.append("TimeGradModel")
if TimeGradLightningModule is not None:
    __all__.append("TimeGradLightningModule")
