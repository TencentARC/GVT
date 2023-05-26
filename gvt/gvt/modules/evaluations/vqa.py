import os
import sys
import json
import logging
import torch.distributed as dist

import os.path as osp
from gvt.modules.evaluations.vqa_tools.vqa import VQA
from gvt.modules.evaluations.vqa_tools.vqa_eval import VQAEval

def save_result(result, result_dir, filename, remove_duplicate=""):
    import json

    result_file = os.path.join(
        result_dir, "%s_rank%d.json" % (filename, dist.get_rank())
    )
    final_result_file = os.path.join(result_dir, "%s.json" % filename)

    json.dump(result, open(result_file, "w"))

    if dist.get_rank() == 0:
        logging.warning("rank %d starts merging results." % dist.get_rank())
        # combine results from all processes
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

def _report_metrics(result_file, split):
    """
    Use official VQA evaluation script to report metrics.
    """

    dataroot = "eval_gt"
    ques_files = {
        "train": "v2_OpenEnded_mscoco_train2014_questions.json",
        "val" : "v2_OpenEnded_mscoco_val2014_questions.json",
        "test": "v2_OpenEnded_mscoco_test2015_questions.json",
        "test-dev": "v2_OpenEnded_mscoco_test-dev2015_questions.json"
    }
    for k, v in ques_files.items(): ques_files[k] = osp.join(dataroot, v)

    anno_files = {
        "train": "v2_mscoco_train2014_annotations.json",
        "val":  "v2_mscoco_val2014_annotations.json"
    }
    for k, v in anno_files.items(): anno_files[k] = osp.join(dataroot, v)

    metrics = {}

    if split in ques_files and split in anno_files:
        vqa = VQA(anno_files[split], ques_files[split])
        vqa_result = vqa.loadRes(
            resFile=result_file, quesFile=ques_files[split]
        )

        # create vqaEval object by taking vqa and vqaRes
        # n is precision of accuracy (number of places after decimal), default is 2
        vqa_scorer = VQAEval(vqa, vqa_result, n=2)
        logging.info("Start VQA evaluation.")
        vqa_scorer.evaluate()

        # print accuracies
        overall_acc = vqa_scorer.accuracy["overall"]
        metrics["agg_metrics"] = overall_acc

        logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
        logging.info("Per Answer Type Accuracy is the following:")

        for ans_type in vqa_scorer.accuracy["perAnswerType"]:
            logging.info(
                "%s : %.02f"
                % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
            )
            metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]


    return metrics
    
def eval(outputs, model_name, split="val"):
    pred_result = []
    for output in outputs:
        for qid, pred in zip(output['qids'], output['preds']):
            pred_result.append({
                "question_id": qid,
                "answer": pred.lower()
            })

    save_filename = f"vqa_result_{split}_{model_name}"
    result_file = save_result(
        pred_result,
        result_dir="pred_results",
        filename=save_filename,
        remove_duplicate="question_id"
    )

    metrics = _report_metrics(result_file, split=split)
    print("evaluated metrics:", metrics)


if __name__ == '__main__':
    result_file = "pred_result/vqa_result_val.json"
    metrics = _report_metrics(result_file, split="val")
    print("metrics:", metrics)

