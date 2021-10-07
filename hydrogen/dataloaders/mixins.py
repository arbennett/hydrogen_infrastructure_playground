

class ScalerMixin:

    def __init__(self, *args, **kwargs):
        pass

    def fit_scalers(self):
        pass

    def apply_scalers(self):
        pass

    def save_scalers(self):
        pass

    def load_scalers(self):
        pass

    @property
    def needs_fitting(self):
        pass


class CNNDataAugmentorMixin:

    def augment(self, batch):
        pass
