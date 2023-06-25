import io
import os
import random

import torch
from PIL import Image
import pyarrow as pa

from gvt.datasets.transforms import get_transforms


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        image_size: int,
        names: list,
        text_column_name: str = "",
        remove_duplicate=False,
        config=None
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        super().__init__()

        self.transforms = get_transforms(size=image_size)
        self.text_column_name = text_column_name
        self.names = names

        self.data_dir = data_dir
        self.config = config

        self.text_in = "What does the image describe?"

        if len(names) != 0: 
            tables = [
                pa.ipc.RecordBatchFileReader(
                    pa.memory_map(f"{data_dir}/{name}.arrow", "r")
                ).read_all()
                for name in names
                if os.path.isfile(f"{data_dir}/{name}.arrow")
            ]

            self.table_names = list()
            for i, name in enumerate(names):
                self.table_names += [name] * len(tables[i])

            self.table = pa.concat_tables(tables, promote=True)
            if text_column_name != "":
                self.text_column_name = text_column_name
                self.all_texts = self.table[text_column_name].to_pandas().tolist()
                self.all_texts = (
                    [list(set(texts)) for texts in self.all_texts]
                    if remove_duplicate
                    else self.all_texts
                )
            else:
                self.all_texts = list()
        else:
            self.all_texts = list()

        self.index_mapper = dict()

        if text_column_name != "" and text_column_name != "boxes" :
            j = 0
            for i, texts in enumerate(self.all_texts):
                for _j in range(len(texts)):
                    self.index_mapper[j] = (i, _j)
                    j += 1
        else:
            for i in range(len(self.table)):
                self.index_mapper[i] = (i, None)


    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]


    def __len__(self):
        return len(self.index_mapper)


    def get_raw_image(self, index, image_key="image"):
        index, _ = self.index_mapper[index]
        image_bytes = io.BytesIO(self.table[image_key][index].as_py())
        image_bytes.seek(0)
        return Image.open(image_bytes).convert("RGB")


    def get_image(self, index, image_key="image"):
        image = self.get_raw_image(index, image_key=image_key)
        image_tensor = [image]
        for tr in self.transforms:
            image_tensor[0] = tr(image_tensor[0])
        return {
            "image": image_tensor,
            "img_index": self.index_mapper[index][0],
            "cap_index": self.index_mapper[index][1],
            "raw_index": index,
        }


    def get_false_image(self, rep, image_key="image"):
        random_index = random.randint(0, len(self.index_mapper) - 1)
        image = self.get_raw_image(random_index, image_key=image_key)
        image_tensor = [tr(image) for tr in self.transforms]
        return {f"false_image_{rep}": image_tensor}


    def get_text(self, raw_index):
        index, caption_index = self.index_mapper[raw_index]

        text = self.all_texts[index][caption_index]

        # by default, using image-text pair
        text_in = "what does the image describe?"
        text_out = text
        return {
            "text_in":  text_in,
            "text_out": text_out,
            "img_index": index,
            "cap_index": caption_index,
            "raw_index": raw_index,
        }

   
    def get_suite(self, index):
        result = None

        while result is None:
            if True:
                ret = dict()
                image = self.get_image(index)
                ret.update(image)

                # if not self.image_only:
                txt = self.get_text(index)
                # ret.update({"replica": True if txt["cap_index"] > 0 else False})
                ret.update(txt)

                result = True

        return ret


    def get_text_suite(self, index):
        result = None
        while result is None:
            ret = dict()
            txt = self.get_text(index)
            ret.update({"replica": True if txt["cap_index"] > 0 else False})
            ret.update(txt)
            result = True
        return ret


    def collate(self, batch):
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        img_keys = [k for k in list(dict_batch.keys()) if "image" in k]
        img_keys = [k for k in img_keys if 'id' not in k]

        for img_key in img_keys:
            new_imgs = [tmp_img[0] for tmp_img in dict_batch[img_key]]
            batch_new_imgs = torch.stack(new_imgs, dim=0)
            dict_batch[img_key] = [batch_new_imgs]

        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d for d in dict_batch[txt_key]] for txt_key in txt_keys]

            for _, txt_key in enumerate(txt_keys):
                texts = [d for d in dict_batch[txt_key]]
                dict_batch[txt_key] = texts

        return dict_batch
