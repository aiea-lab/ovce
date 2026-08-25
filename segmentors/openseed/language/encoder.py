import torch
from torch import nn
from torch.nn import functional as F

from timm.models.layers import trunc_normal_

from .registry import register_model
from ..utils import configurable
from .LangEncoder import build_tokenizer, build_lang_encoder
from ..utils.prompt_engineering import prompt_engineering, get_prompt_templates


class LanguageEncoder(nn.Module):

    @configurable
    def __init__(
        self,
        tokenizer,
        tokenizer_type,
        lang_encoder,
        lang_projection,
        max_token_num,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.lang_encoder = lang_encoder
        self.lang_proj = lang_projection
        self.max_token_num = max_token_num
        self.logit_scale = nn.Parameter(torch.ones([]))

    @classmethod
    def from_config(cls, cfg):
        # build up text encoder
        tokenizer = build_tokenizer(cfg['MODEL']['TEXT'])
        tokenizer_type = cfg['MODEL']['TEXT']['TOKENIZER']
        lang_encoder = build_lang_encoder(cfg['MODEL']['TEXT'], tokenizer, cfg['VERBOSE'])
        max_token_num = cfg['MODEL']['TEXT']['CONTEXT_LENGTH']
        
        dim_lang = cfg['MODEL']['TEXT']['WIDTH']
        dim_projection = cfg['MODEL']['DIM_PROJ']
        lang_projection = nn.Parameter(torch.empty(dim_lang, dim_projection))
        trunc_normal_(lang_projection, std=.02)

        return {
            "tokenizer": tokenizer,
            "tokenizer_type": tokenizer_type,
            "lang_encoder": lang_encoder,
            "lang_projection": lang_projection,
            "max_token_num": max_token_num,
        }

    # ADDED
    def set_text_embeddings(self, classes, name='default'):
        single_token_classes = []
        for classname in classes:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                single_token_classes.append(classname_splits[0])
            else:
                single_token_classes.append(classname)
        self.get_text_embeddings(single_token_classes, name=name)
        return single_token_classes

    def get_text_embeddings(self, class_names, name='default', add_bgd=False, norm=True):
        with torch.no_grad():
            def extract_mean_emb(txts):
                tokens = self.tokenizer(
                    txts, padding='max_length', truncation=True, max_length=self.max_token_num, return_tensors='pt'
                )
                clss_embedding = self.forward_language((tokens['input_ids'].cuda(), tokens['attention_mask'].cuda()), norm=norm)
                clss_embedding = clss_embedding.mean(dim=0)
                clss_embedding /= clss_embedding.norm()
                return clss_embedding

            templates = get_prompt_templates()
            clss_embeddings = []
            for clss in class_names:
                txts = [template.format(clss.replace('-other','').replace('-merged','').replace('-stuff','')) for template in templates]
                clss_embeddings.append(extract_mean_emb(txts))

            if add_bgd:
                txts = ["A background in coco."]
                clss_embeddings.append(extract_mean_emb(txts))

            text_emb = torch.stack(clss_embeddings, dim=0)
            setattr(self, '{}_text_embeddings'.format(name), text_emb)

    # @torch.no_grad()
    def forward_language(self, texts, norm=True):
        x = self.lang_encoder(*texts)
        x = x['last_hidden_state']

        if self.tokenizer_type == 'clip':
            x = x[torch.arange(x.size(0)), texts[0].argmax(dim=-1)]
        else:
            x = x[:, 0]

        x = x @ self.lang_proj
        if norm:
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-7)
        return x
    
    def compute_similarity(self, v_emb, name='default'):
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)
        t_emb = getattr(self, '{}_text_embeddings'.format(name))
        output = self.logit_scale.exp() * v_emb @ t_emb.unsqueeze(0).transpose(1, 2)
        return output


@register_model
def get_language_model(cfg, **kwargs):
    return LanguageEncoder(cfg)