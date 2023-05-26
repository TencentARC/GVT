from collections import defaultdict
import numpy as np

def acc(ls):
    return np.sum(ls) / len(ls)

def eval(outs, model_name):

    result_bin = defaultdict(list)

    count, correct = 0, 0
    for out in outs:
        for pred, gt, n_obj_exist in zip(out["pred"], out["gt"], out["n_obj_exist"]):
            
            count += 1
            if ("Yes" in gt and "yes" in pred.lower()) or ("No" in gt and "no" in pred.lower()): 
                
                correct += 1
                result_bin[n_obj_exist].append(1)

            else:
                result_bin[n_obj_exist].append(0)

    print("accuracy: {:.4f}".format(1.0 * correct / count))

    # 1 - 10; 10 - 20; > 20
    bins = defaultdict(list)
    for k, v in result_bin.items():
        if k < 10:
            bins[0].extend(result_bin[k])
        elif k < 20:
            bins[1].extend(result_bin[k])
        else:
            bins[2].extend(result_bin[k])

    print("1 - 9: {:.4f}".format(acc(bins[0])))
    print("10 - 19: {:.4f}".format(acc(bins[1])))
    print(">= 20: {:.4f}".format(acc(bins[2])))