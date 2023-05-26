from .datamodule_base import BaseDataModule
from gvt.datasets import VCRCountDataset, VCRMultiClassDataset

class VCRCountDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VCRCountDataset

    @property
    def dataset_name(self):
        return "vcr_count"


class VCRMultiClassDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VCRMultiClassDataset

    @property
    def dataset_name(self):
        return "vcr_multiclass"