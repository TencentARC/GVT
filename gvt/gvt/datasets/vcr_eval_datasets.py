from .base_dataset import BaseDataset

class VCRMultiClassDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        names = ["vcr_mci"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="caption",
            remove_duplicate=False,
        )


    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text_in  = self.table["caption"][index][0].as_py()
        text_out = self.table["answer"][index][0].as_py()
        image_id = self.table["image_id"][index][0].as_py()
        n_obj_exist = self.table["n_obj_exist"][index][0].as_py()

        return {
            "image": image_tensor,
            "text_in": text_in,
            "text_out": text_out,
            "image_id": image_id,
            "n_obj_exist": n_obj_exist
        }
    

class VCRCountDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        names = ["vcr_oc"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="caption",
            remove_duplicate=False,
        )


    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text_in  = self.table["caption"][index][0].as_py()
        text_out = self.table["answer"][index][0].as_py()
        image_id = self.table["image_id"][index][0].as_py()
        n_obj_exist = self.table["n_obj_exist"][index][0].as_py()

        return {
            "image": image_tensor,
            "text_in": text_in,
            "text_out": text_out,
            "image_id": image_id,
            "n_obj_exist": n_obj_exist
        }

