import torch
from pytorch_lightning import Callback


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        for k, v in trainer.logged_metrics.items():
            if k not in self.metrics.keys():
                self.metrics[k] = [self._convert(v)]
            else:
                self.metrics[k].append(self._convert(v))

    def _convert(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().detach().numpy()
        return x
