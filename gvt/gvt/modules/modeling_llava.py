import logging
import transformers
from collections import OrderedDict

import copy
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast

import pytorch_lightning as pl

from transformers import AutoTokenizer
from transformers import StoppingCriteria
from transformers import CLIPImageProcessor, CLIPVisionModel

from gvt.modules import utils
from gvt.modules.llava import LlavaLlamaForCausalLM
from gvt.modules.visual_modules.perceiver import PerceiverResampler


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

class MLLM_LLAVA(pl.LightningModule):
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

        
        self.is_llama = True
        self.step_count = 0
        self._lemmatizer = None
        self.pad_left = False
        self.enc_val = 0.
        self.pec_val = 0.
        
        self.visual_encoder_type = self.patch_embed_type

        self.config = config
        self.hparams.config = config # for pytorch lightning

        self.model = LlavaLlamaForCausalLM.from_pretrained(config['llava7b_path'])
        self.model = self.model.cuda()
        self.model.model.vision_tower = [m.cuda() for m in self.model.model.vision_tower]
        
        self.tokenizer = AutoTokenizer.from_pretrained(config['llava7b_path'])
        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = self.model.get_model().vision_tower[0]
        vision_tower.to(device='cuda')
        
        vision_config = vision_tower.config
        vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        self.model.model.vision_tower[0].config = vision_config

    def _tokenize_fn(self, strings, tokenizer):
        """Tokenize a list of strings."""
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for text in strings
        ]
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        # print(input_ids)
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        attention_mask = [tokenized.attention_mask.squeeze() for tokenized in tokenized_list]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
            attention_mask=attention_mask
        )

    def _pad_left(self, id_lists, padding_token=2):
        max_len = max(len(id_list) for id_list in id_lists)
        device = id_lists[0].device

        new_id_lists = []
        pad_lengths = []
        for id_list in id_lists:
            if len(id_list) < max_len:
                pad_length = max_len - len(id_list)
                id_list = torch.cat([
                    padding_token * torch.ones(pad_length).long(),
                    id_list
                ])

                pad_lengths.append(pad_length)        
            else:
                pad_lengths.append(0)

            new_id_lists.append(id_list)

        return torch.stack(new_id_lists), pad_lengths


    def _decorate_image_embeds(self, image_embeds, image_atts, prompts, targets=None):
        B = image_embeds.shape[0]
        device = image_embeds.device
        
        text_input_left = [DEFAULT_IM_START_TOKEN] * B
        if self.use_perceiver:
            image_len = self.num_latents
        else:
            image_len = image_embeds.shape[1]
        text_image_dummy = [" ".join(["I"] * image_len) + " "] * B
        text_input_right = prompts
        text_input_right = [DEFAULT_IM_END_TOKEN + p for p in prompts]
    
        sources = [l + i + r for l, i, r in zip(text_input_left, text_image_dummy, text_input_right)]
        sources_tokenized = self._tokenize_fn(sources, self.tokenizer)

        sources_ids = sources_tokenized["input_ids"]

        if self.pad_left:
            sources_ids, num_padded = self._pad_left(sources_ids, padding_token=self.tokenizer.pad_token_id)
        else:
            sources_ids = torch.nn.utils.rnn.pad_sequence(sources_ids, batch_first=True, 
                                                          padding_value=self.tokenizer.pad_token_id)
        sources_ids = sources_ids.to(device)

        if self.config['use_t5']:
            source_embeds = self.llm.encoder.embed_tokens(sources_ids)
        else:
            source_embeds = self.llm.model.embed_tokens(sources_ids)
        source_attns = torch.ones(source_embeds.size()[:-1]).to(device)
        for i, src_len in enumerate(sources_tokenized["input_ids_lens"]):
            if self.pad_left:
                d = source_attns.shape[1] - src_len
                source_attns[i, :d] = 0
                assert d < source_attns.shape[1], f"error: {d} = {source_attns.shape[1]} - {src_len}"
            else:
                source_attns[i, src_len:] = 0

        left_tokenized = self.tokenizer(text_input_left[0]).input_ids
        left_len = len(left_tokenized) - 1

        if self.pad_left:
            source_embed_list = []
            for i in range(image_embeds.shape[0]):
                len_i = num_padded[i]
                delta_i = len_i + left_len

                assert delta_i + image_len < source_embeds.shape[1], f"error: {delta_i} + {self.num_latents} < {source_embeds.shape[1]}"
                source_embed_left = source_embeds[i, :delta_i, :]
                source_embed_right = source_embeds[i, delta_i + image_len:, :]
                source_embed = torch.cat([source_embed_left, image_embeds[i], source_embed_right])
                source_embed_list.append(source_embed)
            source_embeds = torch.stack(source_embed_list)
        else:
            left  = source_embeds[:, :left_len, :]
            right = source_embeds[:, left_len + image_len:, :]
            source_embeds = torch.cat([left, image_embeds, right], dim=1)

        
        if targets is None:
            return source_embeds, source_attns
        
        examples = [ f"{s} {t}" for s, t in zip(sources, targets)] 
        if self.config["add_eos"]:
            examples = [ e + "</s>" for e in examples ]

        examples_tokenized = self._tokenize_fn(examples, self.tokenizer)

        attention_masks = examples_tokenized["attention_mask"]
        maxlen = max([len(m) for m in attention_masks])
        mask_list = []
        for i in range(len(attention_masks)):
            this_mask = attention_masks[i]
            if len(this_mask) < maxlen:
                this_mask = torch.cat([this_mask, torch.zeros(maxlen - len(this_mask))])
            mask_list.append(this_mask)
        attention_masks = torch.stack(mask_list).to(device)


        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = -100
        
        # padding
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(device)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        input_embeds_left  = self.llm.model.embed_tokens(input_ids[:, :left_len])
        input_embeds_right = self.llm.model.embed_tokens(input_ids[:, left_len+image_len:])
        input_embeds = torch.cat([input_embeds_left, image_embeds, input_embeds_right], dim=1)

        return input_embeds, attention_masks, labels


    
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
        image = samples["image"][0]

        with torch.cuda.amp.autocast(enabled=False):
            
            image_token_len = 256
            prompts = samples["text_in"]
            if 'caption' in self.config['exp_name']:
                prompts = ["Describe this image in detail:" + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + '\n\n### Response:' for qs in prompts]
            else: 
                prompts = [prompt + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN + '\n\n### Response:' for prompt in prompts]
  
            tokenized = self.tokenizer(
                prompts,
                padding="longest",
                return_tensors="pt"
            )
            input_ids = tokenized.input_ids.to(image.device)
            masks = tokenized.attention_mask

            keywords = ['###']
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
            if 'caption' in self.config['exp_name']: num_beams=5
            else: num_beams = 1
            sequence = self.model.generate(
                input_ids=input_ids,
                attention_mask=masks,
                images=image,
                do_sample=True,
                num_beams=num_beams, 
                temperature=0.7,
                max_new_tokens=16,
            )
            
            output_text = self.tokenizer.batch_decode(
                sequence, skip_special_tokens=True
            )

            if self.config['postprocess']:
                new_output_text = []
                # print("using pp")
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
                
            last = "image of" if 'caption' in self.config['exp_name'] else 'Answer'
            new_output_text = [text.split("Response")[1] for text in new_output_text]
        
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
        
        self.tokenizer.padding_side = "left"
        self.pad_left = True

        image = samples["image"][0]
        img_tokens = DEFAULT_IMAGE_PATCH_TOKEN*256

        image_token_len = 256
        prompts = samples["text_in"]
        prompts = [qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len + DEFAULT_IM_END_TOKEN + '\n\n### Response:' for qs in prompts]

        tokenized = self.tokenizer(
            prompts,
            padding="longest",
            return_tensors="pt"
        )
        input_ids = tokenized.input_ids.to(image.device)
        masks = tokenized.attention_mask

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with torch.cuda.amp.autocast(enabled=False):

            keywords = ['###']
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            sequence = self.model.generate(
                input_ids=input_ids,
                attention_mask=masks,
                num_beams=1,
                max_new_tokens=32, 
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                prefix_allowed_tokens_fn=None,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=0,
                length_penalty=1.0, #length_penalty,
                num_return_sequences=1,
                do_sample=False,
                early_stopping=False
            )

            output_text = self.tokenizer.batch_decode(
                sequence, skip_special_tokens=True
            )


            if self.config['postprocess']:
                output_text = [text.strip().split('\n')[0].strip(string.punctuation) for text in output_text]

        output_text = [text.split("Response")[1] for text in output_text]
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

            else:
                return {
                    "pred": pred,
                    "gt": batch["text_out"]
                }


    def test_epoch_end(self, outs):
        
        model_name = "llava"
        if 'vqa' in self.config['exp_name']:
            import vlmo.modules.evaluations.vqa as vqa
            vqa.eval(outs, model_name, split="val")

        if 'multilabel' in self.config['exp_name']:
            import vlmo.modules.evaluations.coco as coco
            coco.eval_multilabel(outs)

        if 'caption' in self.config['exp_name']:
            import vlmo.modules.evaluations.coco_cap as coco_cap
            coco_cap.eval_cap(outs, model_name)

        if 'multiclass' in self.config['exp_name']:
            import vlmo.modules.evaluations.multiclass as mc
            mc.eval(outs, model_name)

        if 'count' in self.config['exp_name']:
            import vlmo.modules.evaluations.count as count
            count.eval(outs, model_name)

    def configure_optimizers(self):
        return utils.set_schedule(self)
