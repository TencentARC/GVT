# evaluate task accordingly
task="task_eval_vqav2"              # VQA
task="task_eval_coco_caption"       # COCO-Captioning
task="task_eval_coco_count"         # OC@COCO
# task="task_eval_coco_multiclass"    # MCI@COCO
# task="task_eval_vcr_count"          # OC@VCR
# task="task_eval_vcr_multiclass"     # MCI@VCR


python run.py with \
    ${task} \
    num_gpus=1 num_nodes=1 \
    test_only=True \
    test_on_val=True \
    image_size=224 \
    num_latents=32 \
    per_gpu_batchsize=4 \
    log_dir=output \
    data_root="/path/to/data" \
    vicuna_path="/path/to/vicuna7b"  \
    load_path="/path/to/gvt_ckpt" 



