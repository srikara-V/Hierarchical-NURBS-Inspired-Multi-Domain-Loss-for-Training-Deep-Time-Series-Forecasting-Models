import importlib
import torch

from utils.device import resolve_device


def _load_model_class(model_name: str):
    """Import only the requested model module (avoids loading all optional deps)."""
    try:
        module = importlib.import_module(f"model.{model_name}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Unknown or unavailable model '{model_name}'") from exc
    if not hasattr(module, "Model"):
        raise ValueError(f"Model module 'model.{model_name}' has no Model class")
    return module.Model


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        Model = _load_model_class(self.args.model)
        return Model(self.args).float()

    def _acquire_device(self):
        if hasattr(self.args, "device"):
            device = self.args.device
        else:
            device = resolve_device(
                use_gpu=getattr(self.args, "use_gpu", True),
                gpu=getattr(self.args, "gpu", 0),
            )
        print(f"Use device: {device}")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
