import json
import os
import os.path as osp
import numpy as np
import pandas as pd
import pyarrow as pa
from tqdm import tqdm
from collections import defaultdict


IMG_PATHS = {
    "coco":  "/path/to/coco/val2017",
    "vcr" :  "/path/to/vcr"
}
SAVE_PATH = "/path/to/save/data"


def generate_arrow_from_anno(tasks, task_name):
    batches = []

    for task in tqdm(tasks):
        if 'coco' in task_name:
            img_path = "{}/{:012}.jpg".format(IMG_PATHS['coco'], task['image_id'])
        else:
            img_path = osp.join(IMG_PATHS['vcr'], 'vcr1images', task['image_id'])
        
        with open(img_path, "rb") as fp:
            binary = fp.read()
        
        batches.append([
            binary,
            [task["text_in"]],
            [task["text_out"]],
            [task["image_id"]],
            [task["n_obj_exist"]]
        ])


    dataframe = pd.DataFrame(
        batches, columns=["image", "caption", "answer", "image_id", "n_obj_exist"]
    )
    table = pa.Table.from_pandas(dataframe)
    
    save_name = f"{SAVE_PATH}/{task_name}.arrow"
    with pa.OSFile(save_name, "wb") as sink:  
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)
    print("saved file:", save_name)
    


if __name__ == '__main__':
    annotations = ["coco_mci.json", "coco_oc.json", "vcr_mci.json", "vcr_oc.json"]
    for anno in annotations:
        print("generating for annotation file:", anno)
        tasks = json.load(open(osp.join("GVTBench", anno)))
        task_name = anno[:-5]
        generate_arrow_from_anno(tasks, task_name)