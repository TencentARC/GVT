import logging
import transformers
from collections import OrderedDict

import copy
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast

import pytorch_lightning as pl
from timm.models import create_model

from lavis.models.blip2_models.blip2 import Blip2Base 
import string
import lavis

from gvt.modules import utils
from gvt.modules.visual_modules.perceiver import PerceiverResampler

import gvt.modules.evaluations.vqa as vqa
import gvt.modules.evaluations.count as count
import gvt.modules.evaluations.coco_cap as coco_cap
import gvt.modules.evaluations.multiclass as mc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class MLLM_MINIGPT4(pl.LightningModule, Blip2Base):

    def __init__(self, 
                 config,
                 vit_model="eva_clip_g",
                 q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
                 freeze_vit=True,
                 freeze_qformer=True,
                 img_size=224,
                 llama_model="params/vicuna7b",
                 num_query_token=32,
                 low_resource=False,
                 use_grad_checkpoint=False,
                 drop_path_rate=0,
                 vit_precision="fp16",
                 end_sym="\n",
                 device_8bit=0
                 ):
        super().__init__()

        self.config = config
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(config['flant5xxl_path'])

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        print('Loading Q-Former Done')

        from transformers import LlamaTokenizer, LlamaForCausalLM
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if self.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        self.max_txt_len = 32
        self.end_sym = end_sym


    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_img(self, image):
        device = image.device
        if self.low_resource:
            self.vit_to_cpu()
            image = image.to("cpu")

        # with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_llama = self.llama_proj(query_output.last_hidden_state)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img


    def forward(self, samples):
        image = samples["image"][0]

        inputs_llm = self.forward_vision(samples)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        with torch.cuda.amp.autocast(enabled=False):
            
            prompts = [p.strip() for p in samples["text_in"]]
            targets = samples["text_out"]

            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                labels=labels,
                attention_mask=masks,
                return_dict=True
            )
            loss = outputs.loss

            return {"loss": loss}
    
    def generate(
            self, 
            samples,
            prompts = None,
            num_beams=3,
            max_len=32,
            min_len=5,
            length_penalty=1,
            repetition_penalty=1):
        
        self.pad_left = True
        self.llama_tokenizer.padding_side = "left"

        image = samples["image"][0]
        inputs_llm, atts_llm = self.encode_img(image)

        with torch.cuda.amp.autocast(enabled=False):
   
            if 'caption' in self.config['exp_name']:
                prompts = ["### Human <Img><ImageHere></Img> Descibe the image in detail: This is an image of " for _ in range(len(samples["text_in"]))]
            else:
                prompts = samples["text_in"]
                prompts = [f"### Human <Img><ImageHere></Img> Question: {q} Answer:" for q in prompts]

            output_text_list = []

            if 'caption' in self.config["exp_name"]: num_beams = 5
            else: num_beams = 1

            for i, prompt in enumerate(prompts):

                inputs_embeds, encoder_atts = self.prompt_wrap(inputs_llm[i].unsqueeze(0), atts_llm[i].unsqueeze(0), prompt=prompt)
                outputs = self.llama_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    num_beams=num_beams,
                    max_new_tokens=32, 
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                    prefix_allowed_tokens_fn=None,
                    no_repeat_ngram_size=0,
                    length_penalty=1.0, #length_penalty,
                    num_return_sequences=1,
                    do_sample=False,
                    early_stopping=False
                )

                output_text = self.llama_tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )

                output_text_list.append(output_text[0])

            output_text = output_text_list

            
            if self.config['postprocess']:
                new_output_text = []
                for text in output_text:                    
                    text = text.replace(".", " . ")
                    text = text.replace("\n", "")
                    ls = text.strip().split(' ')
                    ls = [w for w in ls if w]
                    if len(ls):
                        if ls[0] == '.':
                            ls = ls[1:]
                        if "." in ls:
                            ls = ls[:ls.index(".")]

                        new_output_text.append(
                            " ".join(ls) + "."
                        )
                    else:
                        new_output_text.append('')
            else:
                new_output_text = output_text

        return new_output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=16,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-2,
        repetition_penalty=1.5,
        **kwargs
    ):
        
        self.llama_tokenizer.padding_side = "left"

        image = samples["image"][0]
        inputs_llm, atts_llm = self.encode_img(image)

        new_qs = [q.replace('Question: ', '') for q in samples["text_in"] ]
        prompts = [f"###Human: <Img><ImageHere></Img> {q}" for q in new_qs] # evaluating mcq

        output_text_list = []

        for i, prompt in enumerate(prompts):
            inputs_embeds, encoder_atts = self.prompt_wrap(inputs_llm[i].unsqueeze(0), atts_llm[i].unsqueeze(0), prompt)

            device_type = "cuda" if "cuda" in str(self.device) else "cpu"
            with torch.cuda.amp.autocast(enabled=False):

                outputs = self.llama_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False,
                    num_beams=num_beams,
                    max_new_tokens=max_len,
                    min_length=min_len,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty
                )
                
                sequence = outputs.sequences
                output_text = self.llama_tokenizer.batch_decode(
                    sequence, skip_special_tokens=True
                )

                output_text_list.append(output_text[0])

        output_text = output_text_list


        if self.config['postprocess']:
            output_text = [text.strip().split('\n')[0].strip(string.punctuation) for text in output_text]

        return output_text
    

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
        if 'vqa' in self.config['exp_name']:
            answers = self.predict_answers(batch)
            qids = batch['qid']
            ret = {"qids": qids, "preds": answers} 

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

            if self.config['print_detail']:
                print("gt:", batch['class_gt'])
                print("pred:", multilabel_results, "\n")

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

            else:
            
                return {
                    "pred": pred,
                    "gt": batch["text_out"]
                }

    def test_epoch_end(self, outs):
        
        if 'vqa' in self.config['exp_name'] and "textvqa" not in self.config['exp_name']:
            import vlmo.modules.evaluations.vqa as vqa
            vqa.eval(outs, model_name, split="val")

        if 'caption' in self.config['exp_name']:
            import vlmo.modules.evaluations.coco_cap as coco_cap
            coco_cap.eval_cap(outs, model_name)

        if 'count' in self.config['exp_name']:
            import vlmo.modules.evaluations.count as count
            count.eval(outs, model_name)

        if 'multiclass' in self.config['exp_name']:
            import vlmo.modules.evaluations.multiclass as mc
            mc.eval(outs, model_name)

        if 'count' in self.config['exp_name']:
            import vlmo.modules.evaluations.count as count
            count.eval(outs, model_name)

       
    def configure_optimizers(self):
        return utils.set_schedule(self)