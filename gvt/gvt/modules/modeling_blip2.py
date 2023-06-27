import logging
import transformers
from collections import OrderedDict

import copy
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast

import pytorch_lightning as pl

from gvt.modules import utils

from timm.models import create_model
import string

class MLLM_BLIP2(pl.LightningModule):
    """
    BLIP2 T5 model.
    Supported model types:
        - pretrain_flant5xl: pretrained model with FlanT5-XL
        - pretrain_flant5xxl: pretrained model with FlanT5-XXL
        - caption_coco_flant5xl: fintuned image captioning model with FlanT5-XL
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_t5", "pretrain_flant5xl")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_flant5base": "configs/models/blip2/blip2_pretrain_flant5base.yaml",
        "pretrain_flant5xl": "configs/models/blip2/blip2_pretrain_flant5xl.yaml",
        "pretrain_flant5xxl": "configs/models/blip2/blip2_pretrain_flant5xxl.yaml",
        "caption_coco_flant5xl": "configs/models/blip2/blip2_caption_flant5xl.yaml",
    }

    def __init__(self, config=None):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.img_size = config["image_size"]
        self.drop_path_rate = 0
        self.use_grad_checkpoint = False
        self.vit_precision = "fp16"

        self.llm_type = config["llm_type"]
        self.max_txt_len = 32
        self.patch_embed_type = config["patch_embed_type"]
        self.num_latents = config['num_latents']
        self.apply_lemmatizer = False
        self.use_perceiver = config['use_perceiver']
        self.patch_embed_dim = 768
        if self.patch_embed_type == "rclip":
            self.patch_embed_dim = 1408

        self.is_llama = True
        self.step_count = 0
        self._lemmatizer = None
        self.pad_left = False
        self.enc_val = 0.
        self.pec_val = 0.
        
        self.visual_encoder_type = self.patch_embed_type

        self.config = config
        self.hparams.config = config 
        from lavis.models import load_model_and_preprocess
        import lavis
   
        model, vis_processor, txt_processor = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl")
        self.llm = model
        self.vis_processor = vis_processor
        self.txt_processor = txt_processor 
        
    
    def generate(
            self, 
            samples,
            prompts = None,
            num_beams=3,
            max_len=32,
            min_len=5,
            length_penalty=-1):
        
        if 'caption' in self.config['exp_name']:
            prompts = ["Describe the image in detail in a sentence. This is an image of " for _ in range(len(samples["text_in"]))]
        else:
            prompts = samples["text_in"]
            
        samples["prompt"] = prompts
        samples["image"] = samples["image"][0]
        output_text = self.llm.generate(samples)
        new_output_text = []
        for text in output_text:
                
                text = text.replace(".", " . ")
                text = text.replace("\n", "")
                ls = text.strip().split(' ')
                ls = [w for w in ls if w]
                if len(ls) and ls[0] == '.':
                    ls = ls[1:]
                if "." in ls:
                    ls = ls[:ls.index(".")]

                new_output_text.append(
                    " ".join(ls) + "."
                )
                
        return new_output_text
        

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-2,
        repetition_penalty=1.5,
        **kwargs
    ):

        prompt = [f"Question: {q} Short answer: " for q in samples["text_in"]]
        
        samples["prompt"] = prompt
        samples["image"] = self.vis_processor['eval'](samples["raw"][0]).unsqueeze(0).cuda()
        return self.llm.generate(samples)
    

    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

    @classmethod
    def from_config(cls, cfg):
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        t5_model = cfg.get("t5_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        model = cls(
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            t5_model=t5_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
            apply_lemmatizer=apply_lemmatizer,
        )
        model.load_checkpoint_from_config(cfg)

        return model


    def test_step(self, batch, batch_index):
        if 'vqa' in self.config['exp_name'] and 'textvqa' not in self.config['exp_name']:
            answers = self.predict_answers(batch)
            qids = batch['qid']
            ret = {"qids": qids, "preds": answers} # qids and answers are both list[str]

            if 'vqa_answer' in batch: 
                ret["gts"] = batch["vqa_answer"]
                ret['scores'] = batch["vqa_scores"]

            return ret
        
        elif 'multilabel' in self.config['exp_name']:

            multilabel_results = self.list_objects(batch)
            ret = {}
            ret['pred'] = multilabel_results
            ret['gt'] = batch['class_gt']
            ret['image_id'] = batch['image_id']
            ret['n_obj_exist'] = batch['n_obj_exist']

            return ret

        else: 
            ret = {}
            pred = self.generate(batch)

            
            if 'iid' in batch or 'image_id' in batch:
                if 'iid' in batch: iid = batch['iid']
                else: iid = batch['image_id']

                ret = {
                    "image_id": iid,
                    "pred": pred,
                    "gt": batch["text_out"]
                }

                if 'n_obj_exist' in batch:
                    ret['n_obj_exist'] = batch['n_obj_exist']

                return ret

            
    def test_epoch_end(self, outs):
        model_name = 'blip2'

        if 'vqa' in self.config['exp_name'] and "textvqa" not in self.config['exp_name']:
            import vlmo.modules.evaluations.vqa as vqa
            vqa.eval(outs, model_name, split="val")

        if 'multilabel' in self.config['exp_name']:
            import vlmo.modules.evaluations.coco as coco
            coco.eval_multilabel(outs)

        if 'caption' in self.config['exp_name']:
            import vlmo.modules.evaluations.coco_cap as coco_cap
            coco_cap.eval_cap(outs, model_name)

        if 'count' in self.config['exp_name']:
            import vlmo.modules.evaluations.count as count
            count.eval(outs, model_name)


    def configure_optimizers(self):
        return utils.set_schedule(self)