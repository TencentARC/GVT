import os
import json

import torch.distributed as dist
from pycocotools.coco import COCO

from gvt.modules.evaluations.cider.cider import Cider
from gvt.modules.evaluations.spice.spice import Spice
from gvt.modules.evaluations.tokenizer.ptbtokenizer import PTBTokenizer


class COCOEvalCap:
    def __init__(self, coco, cocoRes):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {'image_id': cocoRes.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']

        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


def save_result(result, result_dir, filename, remove_duplicate=""):

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

def coco_caption_eval(coco_gt_root, results_file, split):

    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    annotation_file = os.path.join(coco_gt_root, filenames[split])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval


def _report_metrics(eval_result_file, split_name):

    coco_gt_root = "eval_gt"
    coco_val = coco_caption_eval(coco_gt_root, eval_result_file, split_name)
    return coco_val


def eval(outs, model_name):
    results = []
    for out in outs:
        captions = out['pred']
        iids = out["image_id"]
        for iid, caption in zip(iids, captions):
            results.append({"caption": caption,
                            "id": iid,
                            "image_id": iid})

    new_results =   []
    for item in results:
        if 'id' not in item:
            item['id'] = item['image_id']
        if 'caption' not in item:
            item['caption'] = item['caption:']
        new_results.append(item)

    result_file = save_result(
        result=new_results,
        result_dir="pred_results",
        filename="{}".format(model_name),
        remove_duplicate="id",
    )

    metrics = _report_metrics(result_file, split_name='val')

    print("metrics:")
    print(metrics)