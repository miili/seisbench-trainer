# Partial type stubs for optuna-integration (it ships no type information).
# Only the API used by seisbench-trainer is declared; signatures follow 5.0.
import optuna
from lightning.pytorch.callbacks import Callback

class PyTorchLightningPruningCallback(Callback):
    def __init__(self, trial: optuna.Trial, monitor: str) -> None: ...
