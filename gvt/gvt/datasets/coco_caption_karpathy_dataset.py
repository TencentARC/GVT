from .base_dataset import BaseDataset


class CocoCaptionKarpathyDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        names = ["coco_caption_karpathy_val"]
        super().__init__(*args, **kwargs, names=names, text_column_name="caption")


    def __getitem__(self, index):
        suite = self.get_suite(index)

        if "test" in self.split or "val" in self.split:
            _index, _ = self.index_mapper[index]
            iid = self.table["image_id"][_index].as_py()
            iid = int(iid.split(".")[0].split("_")[-1])
            suite.update({"iid": iid})

        return suite

    @property
    def dataset_name(self):
        return "coco"
