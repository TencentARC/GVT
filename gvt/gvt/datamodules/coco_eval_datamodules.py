from gvt.datasets import COCOMultiClassDataset, COCOCountDataset
from .datamodule_base import BaseDataModule


class COCOMultiClassDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return COCOMultiClassDataset

    @property
    def dataset_name(self):
        return "coco_multiclass"


class COCOCountDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return COCOCountDataset

    @property
    def dataset_name(self):
        return "coco_count"
