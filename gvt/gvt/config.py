from sacred import Experiment

ex = Experiment("GVT")


@ex.config
def config():
    exp_name = "GVT"
    seed = 1
    datasets = ["coco_count"]
    batch_size = 1024 
    image_size = 224

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.05
    decay_power = "cosine"
    max_epoch = 100
    max_steps = 200000
    warmup_steps = 0.1
    end_lr = 0
    lr_mult = 1  

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    use_sharded_training = False
    resume_during_training = False

    per_gpu_batchsize = 10  
    num_gpus = 1
    num_nodes = 1
    num_workers = 32
    precision = 16

    print_detail = False
    finetune_lm = False
    finetune_enc = False
    accum_grad = True
    num_latents = 32
    test_on_val = True

    data_root = ""
    vicuna_path = ""
    visual_tokenizer_path = ""
    load_path = ""
    log_dir = ""



@ex.named_config
def task_eval_vqav2():
    exp_name = "eval_vqav2"
    datasets = ["vqa"]


@ex.named_config
def task_eval_coco_caption():
    exp_name = "eval_coco_caption"
    datasets = ["coco"]


@ex.named_config
def task_eval_coco_multiclass():
    exp_name = "eval_coco_multiclass"
    datasets = ["coco_multiclass"]


@ex.named_config
def task_eval_coco_count():
    exp_name = "eval_coco_count"
    datasets = ["coco_count"]


@ex.named_config
def task_eval_vcr_multiclass():
    exp_name = "eval_vcr_multiclass"
    datasets = ["vcr_multiclass"]


@ex.named_config
def task_eval_vcr_count():
    exp_name = "eval_vcr_count"
    datasets = ["vcr_count"]
