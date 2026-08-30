from collections import OrderedDict
from typing import List, Tuple, Union

import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import clip
from torchvision.transforms.transforms import Normalize
from detectron2.data import MetadataCatalog

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, d_model: int, n_head: int):
        super().__init__(d_model, n_head)
        assert self._qkv_same_embed_dim
        self.new_q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, query, nq, attn_mask, need_weights=False):
        seq, bs, _ = query.shape
    
        # [Mask Class Tokens, (class token, image tokens)]
        q, k, v = F.linear(query[nq:].detach(), self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1)
        q = q / math.sqrt(self.head_dim)
        
        clip_attn = torch.bmm(q, k.transpose(-2, -1))
        clip_output = torch.bmm(F.softmax(clip_attn, dim=-1), v)
        clip_output = clip_output.transpose(0, 1).reshape(-1, bs, self.embed_dim)
        
        assert attn_mask.dtype == torch.bool
        attn_mask_float = torch.zeros_like(attn_mask, dtype=q.dtype)
        attn_mask_float = attn_mask_float.masked_fill(attn_mask, float("-inf"))
        
        # Mask Class Tokens
        new_q = self.new_q_proj(query[:nq])
        new_q = new_q.contiguous().view(-1, bs * self.num_heads, self.head_dim).transpose(0, 1)
        new_q = new_q / math.sqrt(self.head_dim)

        mask_attn = torch.bmm(new_q, k.transpose(-2, -1))
        mask_output = torch.bmm(F.softmax(mask_attn + attn_mask_float, dim=-1), v)
        mask_output = mask_output.transpose(0, 1).reshape(nq, bs, self.embed_dim)

        attn_output = torch.concat([mask_output, clip_output], dim=0).contiguous()
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(seq, bs, -1)

        if need_weights:
            attn_output_weights = mask_attn.view(bs, self.num_heads, nq, -1)
            attn_output_weights = attn_output_weights.mean(dim=1)
            return attn_output, attn_output_weights
        else:
            return attn_output, None


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, y, nq, attn_mask):
        return self.attn(y, nq, attn_mask, need_weights=False)[0]

    def forward(self, y, attn_mask):
        bs, nq, _ = attn_mask.shape
        attn_mask = attn_mask[:, None].repeat(1, self.n_head, 1, 1)
        attn_mask = attn_mask.view(bs * self.n_head, nq, -1)

        y = y + self.attention(self.ln_1(y), nq, attn_mask)
        y = y + self.mlp(self.ln_2(y))
        return y


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, patch_size: int):
        super().__init__()
        self.width = width
        self.layers = layers
        self.patch_size = patch_size
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])
    
    def forward(self, y, attn_mask):
        for layer in list(self.resblocks.modules())[0]:
            y = layer(y, attn_mask)
        return y


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.patch_size = patch_size
        self.layers = layers
        self.width = width

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, patch_size)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # normalize
        self.clip_prep_img = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def forward(self, img, masks, mask_pe):
        img_size = self.input_resolution
        x = F.interpolate(img / 255., (img_size, img_size), mode="bicubic")
        x = self.clip_prep_img(x)
        
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        clip_token = x + self.positional_embedding.to(x.dtype)
        mask_token = mask_pe + self.class_embedding  # Mask Class Tokens
        tokens = torch.concat([mask_token, clip_token], dim=1)
        
        attn_mask = self.get_attn_masks(masks)
        tokens = self.ln_pre(tokens).permute(1, 0, 2)  # NLD -> LND
        tokens = self.transformer(tokens, attn_mask)
        return tokens

    def get_final_embedding(self, tokens, nq: int):
        tokens = tokens.permute(1, 0, 2)  # LND -> NLD
        embedding = self.ln_post(tokens[:, :nq])
        embedding = embedding @ self.proj
        return embedding

    def get_attn_masks(self, pred_masks):
        img_size = self.input_resolution
        masks = F.interpolate(pred_masks, (img_size, img_size), mode="bilinear")
        masks = F.max_pool2d(masks, self.patch_size, self.patch_size)
        bin_masks = (masks > 0.).flatten(2)  # binary
        attn_mask = torch.concat((torch.ones_like(bin_masks[..., [0]]), bin_masks), dim=2)
        return attn_mask.logical_not()


class MasQCLIP(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        assert len(model_name) == 1
        self.model_clip_name = model_name[0]
        _clip, _ = clip.load(model_name[0], device="cuda")
        self.visual = self.load_clip_model(model_name[0])
        keys = self.visual.load_state_dict(_clip.visual.state_dict(), strict=False)
        
        self.text_embeddings = None
        # positional embedding
        self.mask_embeddings = nn.Parameter(self.visual.positional_embedding[0])
        del _clip, keys

    def set_text_embedding(self, class_names):
        _clip, _ = clip.load(self.model_clip_name, device="cuda")
        self.visual = self.load_clip_model(self.model_clip_name)
        keys = self.visual.load_state_dict(_clip.visual.state_dict(), strict=False)
        with torch.no_grad():
            self.text_embeddings = self.load_text_embedding(class_names, _clip, "cuda").float().detach()
        del _clip, keys
        return class_names

    def forward(self, img, masks):
        if self.text_embeddings is None:
            raise ValueError("Text Embeddings not set. Use set_text_embedding() to set the text embeddings.")
        bs, nq, device = masks.shape[0], masks.shape[1], img.device
        mask_pe = self.mask_embeddings.to(device) + torch.zeros((bs, nq, self.visual.width), device=device)
        tokens = self.visual(img, masks, mask_pe)

        # projection
        feature = self.visual.get_final_embedding(tokens, nq)
        feature = feature / feature.norm(p=2, dim=-1, keepdim=True)
        pred_logits = torch.einsum("bqc,nc->bqn", feature, self.text_embeddings) * 100
        return {"pred_logits": pred_logits}

    def load_clip_model(self, model_name):
        if model_name == "ViT-L/14@336px":
            return VisionTransformer(input_resolution=336, patch_size=14, width=1024, layers=24, heads=16, output_dim=768)
        elif model_name == "ViT-L/14":
            return VisionTransformer(input_resolution=224, patch_size=14, width=1024, layers=24, heads=16, output_dim=768)
        elif model_name == "ViT-B/16":
            return VisionTransformer(input_resolution=224, patch_size=16, width=768, layers=12, heads=12, output_dim=512)
        elif model_name == "ViT-B/32":
            return VisionTransformer(input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512)
        
        assert False


    # MODIFIED MATTIA   
    def load_text_embedding(self, classes, model, device):
        class_embedding = []
        classes = classes + ["background"]
        class_names = [c.split(", ") for c in classes]
        for names in class_names:
            text_embedding = model.encode_text(clip.tokenize(names).to(device))
            text_embedding = text_embedding.mean(dim=0, keepdim=True)
            class_embedding.append(text_embedding)
        class_embedding = torch.concat(class_embedding, dim=0).float().detach()
        class_embedding = class_embedding / class_embedding.norm(p=2, dim=-1, keepdim=True)
        return class_embedding