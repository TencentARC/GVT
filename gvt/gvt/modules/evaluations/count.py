import os
import re
import json
import numpy as np
import torch.distributed as dist
from collections import defaultdict, OrderedDict

str2d = {
    "none": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nive": 9
}


def save_result(result, result_dir, filename, remove_duplicate=""):
    import json

    result_file = os.path.join(
        result_dir, "%s_rank%d.json" % (filename, dist.get_rank())
    )
    final_result_file = os.path.join(result_dir, "%s.json" % filename)

    json.dump(result, open(result_file, "w"))


    result = []
    for rank in range(dist.get_world_size()):
        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, rank)
        )
        res = json.load(open(result_file, "r"))
        result += res

    if remove_duplicate:
        result_new = []
        id_list = []
        for res in result:
            if res[remove_duplicate] not in id_list:
                id_list.append(res[remove_duplicate])
                result_new.append(res)
        result = result_new

    json.dump(result, open(final_result_file, "w"))
    print("result file saved to %s" % final_result_file)

    return final_result_file


def extract_digit(string):
    for k, v in str2d.items():
        string = string.replace(k, str(v))

    digit = re.findall("\d+", string)
    if digit and len(digit):
        digit = int(digit[0])
    else:
        digit = 0
    return digit


def _report_metrics(result_file):
    count, correct, mae_sum, mse_sum = 0, 0, 0, 0

    pred_gt = json.load(open(result_file))

    result_bin = defaultdict(list)
    for entry in pred_gt:
        pred = extract_digit(entry["pred"])
        gt = entry["gt"]

        pred = int(pred)
        gt = int(gt)

        count += 1
        if int(pred) == int(gt):
            correct += 1

        mae_sum += np.abs(int(gt) - int(pred))
        mse_sum += np.power(int(gt) - int(pred), 2) 

        result_bin[gt//3].append([pred, gt])


    return {
        "accuracy": correct / count, 
        "mae": mae_sum / count,
        "rmse": np.sqrt(mse_sum / count),
    }, result_bin


def stat(result):
    count = len(result)
    mae_sum = 0
    mse_sum = 0
    correct = 0
    for pred, gt in result:
        pred = int(pred)
        gt = int(gt)
        
        mae_sum += np.abs(int(gt) - int(pred))
        mse_sum += np.power(int(gt) - int(pred), 2)
        if pred == gt: 
            correct += 1

    return{
        "acc":  "{:.4f}".format(1.0 * correct / count),
        "mae":  "{:.4f}".format(mae_sum / count),
        "rmse": "{:.4f}".format(np.sqrt(mse_sum / count))
    }


def eval(outputs, model_name, split="val"):
    pred_results = []
    for out in outputs:
        for pred, gt, image_id, n_obj_exist in zip(out['pred'], out['gt'], out['image_id'], out['n_obj_exist']):
            pred_results.append({
                "pred": pred,
                "gt": gt,
                "image_id": image_id,
                "n_obj_exist": n_obj_exist
            })

    save_filename = f"count_result_{split}_{model_name}"
    result_file = save_result(
        pred_results,
        result_dir="pred_results/count",
        filename=save_filename
    )

    metrics, result_bin = _report_metrics(result_file)
    for k, v in metrics.items():
        print(k, "{:.4f}".format(v))

    keys = sorted(list(result_bin.keys()))
    new_result_bin = OrderedDict({k: result_bin[k] for k in keys})

    for k, v in new_result_bin.items():
        print(f"{k*3}~{(k+1)*3}:", stat(v))
