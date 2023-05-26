from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()

        self._config = _config

        self.data_dir = _config["data_root"]
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size
        self.image_size = _config["image_size"]

        self.setup_flag = False


    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")


    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")


    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            split="train",
            image_size=self.image_size,
            config=self._config
        )


    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            split="val",
            image_size=self.image_size,
            config=self._config
        )

        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.dataset_cls_no_false(
                self.data_dir,
                split="val",
                image_size=self.image_size,
                config=self._config
            )


    def make_no_false_val_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            self.data_dir,
            split="val",
            image_size=self.image_size,
            image_only=image_only,
        )


    def make_no_false_test_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            self.data_dir,
            split="test",
            image_size=self.image_size,
            image_only=image_only,
        )


    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            split="test",
            image_size=self.image_size,
            config=self._config
        )


    def setup(self):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.setup_flag = True


    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader


    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader


    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader
