# GVT

## Installation
```
pip install -r requirements.txt
```

install apex
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

install pycocotools
```
pip install git+https://github.com/philferriere/cocoapi.git
```


## Dataset Preparation


### OC and MCI
The evaluation used [MS-COCO](https://cocodataset.org/) and [VisualCommenseReasoning](https://visualcommonsense.com/) datasets, please download them.

Then, we convert them into `.arrow` format using the following script
1. Change the path in [write_task_from_anno.py](./gvt/utils/write_task_from_anno.py) to where you stored the datasets (line11 - line15).
2. Run `python gvt/utils/write_task_from_anno.py` to generate the `.arrow` files. 

### VQA
Download VQAv2 datasets from [link](). 

Change the file path (`VQA_PATH` and `SAVE_PATH`) in `gvt/utils/write_vqa_open_ended.py`

Generate corrsponding `.arrow` file using `python gvt/utils/write_vqa_open_ended.py`.

Copy evaluation annoation:

`cp v2_OpenEnded_mscoco_val2014_questions ./eval_gt/`

`cp v2_mscoco_val2014_annotations.json ./eval_gt/`

### Captioning
Download MSCOCO datasets from [link]().

Change the file path (COCO_PATH and SAVE_PATH) in `gvt/utils/write_coco_karpathy.py`

Generate corrsponding `.arrow` file using `python gvt/utils/write_coco_karpathy.py`.

Copy evaluation annotation:

`cp coco_karpathy_val_gt.json ./eval_gt`


## Checkpoint
Download the weights from [here](https://drive.google.com/file/d/14ficAR-WL8M0-rZaAz5bSh2DnzgaSpqS/view?usp=share_link)

### Prepraing Vicuna
Our model used Vicuna-7b, please follow corresponding [instructions](https://github.com/lm-sys/FastChat) to prepare the weights


## Evaluation
Create directory to save generation results: `mkdir pred_results output`

Change the paths `data_root`, `vicuna_path` and `load_path` in the scripts to where you stored the checkpoint and dataset.


You may evaluate corrsponding task with different task names
```
bash scripts/eval.sh
```

