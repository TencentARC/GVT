from functools import partial
import string

import copy
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast

import pytorch_lightning as pl

from gvt.modules import utils
from gvt.modules.visual_modules.perceiver import PerceiverResampler        
from gvt.modules.visual_modules.eva import EVAVisionTransformer
from apex.normalization import FusedLayerNorm

# for evaluation
import gvt.modules.evaluations.vqa as vqa
import gvt.modules.evaluations.count as count
import gvt.modules.evaluations.coco_cap as coco_cap
import gvt.modules.evaluations.multiclass as mc


def postprocess(output_text):
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
    return new_output_text


class GVT(pl.LightningModule):
    

    def __init__(self, config=None):

        super().__init__()
        self.save_hyperparameters()
        
        self.img_size = config["image_size"]
        self.num_latents = config['num_latents']
        
        self.config = config
        self.hparams.config = config # for pytorch lightning

        
        self.visual_encoder = self.init_vision_encoder()
        self.ln_vision = nn.LayerNorm(self.patch_embed_dim)
        for n, p in self.visual_encoder.named_parameters():
            p.requires_grad = False

        self._init_llama()
        self.llm_proj = nn.Linear(
            self.patch_embed_dim, self.llm.config.hidden_size
        )
        self.input_ln = nn.LayerNorm(self.llm.config.hidden_size)

        self.perceiver = PerceiverResampler(
            dim = self.patch_embed_dim,
            dim_head = 96,
            depth = 6,
            heads = 16,
            num_latents = self.num_latents,
            num_media_embeds = 1
        )


    def _init_llama(self):
        def smart_tokenizer_and_embedding_resize(
            special_tokens_dict, tokenizer, model
        ):
            num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
            model.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = model.get_input_embeddings().weight.data
                output_embeddings = model.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg


        from transformers import AutoTokenizer, AutoModelForCausalLM   
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['vicuna_path'])
        self.llm = AutoModelForCausalLM.from_pretrained(self.config['vicuna_path'])
            
        token_dict = dict(pad_token='</s>')
        if not self.tokenizer.eos_token:
            token_dict['eos_token'] = '</s>'
        if not self.tokenizer.unk_token:
            token_dict['unk_token'] = '<unk>'

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=token_dict,
            tokenizer=self.tokenizer,
            model=self.llm,
        )

        for n, p in self.llm.named_parameters():
            p.requires_grad = False

        self.llm.eval()


    def __load_state_dict(self, model, state_dict, filter="decoder"):
        updated_keys = []
        missing_keys = []
        unexpected_keys = list(state_dict.keys())

        for n, p in model.named_parameters():
            if n in state_dict.keys():
                updated_keys.append(n)
                unexpected_keys.remove(n)
            else:
                missing_keys.append(n)

        if missing_keys:
            print("missing keys:", [k for k in missing_keys if filter not in k])
        if unexpected_keys:
            print("unexpected keys:", [k for k in unexpected_keys if filter not in k])

        model.load_state_dict(state_dict, strict=False)
        updated_keys = []
        for k, v in model.named_parameters():
            if k in state_dict: updated_keys.append(k)
                
        print("load state dict done!", "ours:", len(list(model.named_parameters())), "state dict:", len(list(state_dict.items())))


    def init_vision_encoder(self):

        encoder = EVAVisionTransformer(img_size=224, patch_size=14, depth=24,
                                        mlp_ratio=2.6667, num_heads=16, embed_dim=1024,
                                        drop_path_rate=0, xattn=True,
                                        qkv_bias=True,
                                        norm_layer=partial(FusedLayerNorm, eps=1e-6),
                                        rope=True, pt_hw_seq_len=16, intp_freq=True,
                                        naiveswiglu=True, subln=True)
        self.patch_embed_dim = 1024

        if self.config['visual_tokenizer_path']:
            params = torch.load(self.config['visual_tokenizer_path'], map_location="cpu")['model']
        
            new_state_dict = {}
            for k, v in params.items():
                if 'encoder.' in k:
                    new_k = k.replace("encoder.", "")
                    new_state_dict[new_k] = v
        
            new_state_dict = {}
            for k, v in params.items():
                if 'visual' in k and 'head' not in k:
                    new_k = k.replace("visual.", "")
                    new_state_dict[new_k] = v

            self.__load_state_dict(encoder, new_state_dict)
        
        return encoder


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

        text_input_left = ["<Img> "] * B
        image_len = self.num_latents

        text_image_dummy = [" ".join(["I"] * image_len) + " "] * B
        text_input_right = ["</Img>" + p for p in prompts]
    
        sources = [l + i + r for l, i, r in zip(text_input_left, text_image_dummy, text_input_right)]

        sources_tokenized = self._tokenize_fn(sources, self.tokenizer)
        sources_ids = sources_tokenized["input_ids"]

        sources_ids, num_padded = self._pad_left(sources_ids, padding_token=self.tokenizer.pad_token_id)
        sources_ids = sources_ids.to(device)


        source_embeds = self.llm.model.embed_tokens(sources_ids)
        source_attns = torch.ones(source_embeds.size()[:-1]).to(device)

        for i, src_len in enumerate(sources_tokenized["input_ids_lens"]):
            if self.pad_left:
                d = source_attns.shape[1] - src_len
                source_attns[i, :d] = 0
            else:
                source_attns[i, src_len:] = 0

        left_tokenized = self.tokenizer(text_input_left[0]).input_ids
        left_len = len(left_tokenized) - 1

        if self.pad_left:
            source_embed_list = []
            for i in range(image_embeds.shape[0]):
                len_i = num_padded[i]
                delta_i = len_i + left_len
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
        
        examples = [ f"{s} {t}</s>" for s, t in zip(sources, targets)] 
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


    def forward_vision(self, samples):
        image = samples["image"][0]

        image_embeds = self.visual_encoder.forward_features(image, return_all_features=True)
        image_embeds = self.ln_vision(image_embeds)
        inputs_llm = self.perceiver(image_embeds)
        return self.llm_proj(inputs_llm)

    
    def generate(self, samples):
        
        self.pad_left = True
        self.tokenizer.padding_side = "left"

        image = samples["image"][0]

        inputs_llm = self.forward_vision(samples)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        with torch.cuda.amp.autocast(enabled=False):

            prompts = samples["text_in"]
            if 'caption' in self.config['exp_name']:
                prompts = ["Descibe the image in detail in a sentence: This is an image of " for _ in range(len(samples["text_in"]))]
            elif 'count' in self.config['exp_name']:
                pass
                
            inputs_embeds, encoder_atts = self._decorate_image_embeds(inputs_llm, atts_llm, prompts)
            
            if 'caption' in self.config['exp_name']: num_beams = 5 
            else: num_beams = 1

            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                num_beams=num_beams,
                max_new_tokens=32, 
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                prefix_allowed_tokens_fn=None,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=0,
                length_penalty=1.0, 
                num_return_sequences=1,
                do_sample=False,
                early_stopping=False
            )

            sequence = outputs
            tokenizer_length = len(self.tokenizer)
            sequence[sequence >= tokenizer_length] = self.tokenizer.eos_token_id
            sequence[sequence < 0] = self.tokenizer.eos_token_id

            output_text = self.tokenizer.batch_decode(
                sequence, skip_special_tokens=True
            )

            output_text = postprocess(output_text)

            if self.config['print_detail']:
                print(prompts)
                print(output_text, "\n")

        return output_text



    def predict_answers(
        self,
        samples,
        num_beams=1,
        max_len=32,
        min_len=1,
        length_penalty=-2,
        repetition_penalty=1.5,
    ):
        
        self.tokenizer.padding_side = "left"
        self.pad_left = True

        image = samples["image"][0]
        inputs_llm = self.forward_vision(samples)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
        
        prompts = [f"Question: {q} Answer:" for q in samples["text_in"]]
        inputs_embeds, encoder_atts = self._decorate_image_embeds(inputs_llm, atts_llm, prompts)

        with torch.cuda.amp.autocast(enabled=False):

            outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                eos_token_id=self.tokenizer.eos_token_id,
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
            sequence[sequence < 0] = self.tokenizer.eos_token_id

            output_text = self.tokenizer.batch_decode(
                sequence, skip_special_tokens=True
            )

            output_text = [text.strip().split('\n')[0].strip(string.punctuation) for text in output_text]
            output_text = postprocess(output_text)

        if self.config['print_detail']:
            print("prompts:", prompts)
            print("gt:", samples["text_out"])
            print("pred:", output_text, "\n")

        return output_text
    

    def test_step(self, batch, batch_index):
        if 'vqa' in self.config['exp_name'] and 'textvqa' not in self.config['exp_name']:
            answers = self.predict_answers(batch)
            qids = batch['qid']
            ret = {"qids": qids, "preds": answers}
            
            if 'vqa_answer' in batch: 
                ret["gts"] = batch["vqa_answer"]
                ret['scores'] = batch["vqa_scores"]

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
        
        try:
            model_name = self.config['load_path'].split('/')[1] #f"mllm_{self.config['patch_embed_type']}"
        except:
            model_name = "dummy"

        if 'vqa' in self.config['exp_name']:
            vqa.eval(outs, model_name)

        if 'multiclass' in self.config['exp_name']:
            mc.eval(outs, model_name)

        if 'count' in self.config['exp_name']:
            count.eval(outs, model_name)

        if 'caption' in self.config['exp_name']:
            coco_cap.eval(outs, model_name)

    def configure_optimizers(self):
        return utils.set_schedule(self)
