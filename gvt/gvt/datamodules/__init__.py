from .vqav2_datamodule import VQAv2DataModule
from .coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from .coco_eval_datamodules import COCOMultiClassDataModule, COCOCountDataModule
from .vcr_eval_datamodules import VCRCountDataModule, VCRMultiClassDataModule


_datamodules = {
    "coco": CocoCaptionKarpathyDataModule,
    "vqa": VQAv2DataModule,
    "coco_multiclass": COCOMultiClassDataModule,
    "coco_count": COCOCountDataModule,
    "vcr_count": VCRCountDataModule,
    "vcr_multiclass": VCRMultiClassDataModule,
}
