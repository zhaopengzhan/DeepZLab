import copy

import nltk
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import OmegaConf
from timm.loss import SoftTargetCrossEntropy
from .tokenizer import SimpleTokenizer
from torchvision import transforms

# from .formatting import ToDataContainer
# utils
from collections import OrderedDict

import torch
from einops import rearrange
from torch import nn


class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict([('c_fc', nn.Linear(d_model, d_model * 4)), ('gelu', QuickGELU()),
                         ('c_proj', nn.Linear(d_model * 4, d_model))]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask=None):
        x = x + self.attention(self.ln_1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


# transformer
import torch
import torch.utils.checkpoint as checkpoint
from torch import nn
from .misc import Result


class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, use_checkpoint=False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor):
        for resblock in self.resblocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(resblock, x)
            else:
                x = resblock(x)
        return x


class TextTransformer(nn.Module):

    def __init__(
            self,
            context_length: int = 77,
            width: int = 256,
            layers: int = 12,
            vocab_size=49408,
            use_checkpoint=False,
    ):
        super().__init__()
        heads = width // 64
        self.context_length = context_length
        self.width = width
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            attn_mask=self.build_attention_mask(),
            use_checkpoint=use_checkpoint)

        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, width))
        self.ln_final = nn.LayerNorm(width)
        self.token_embedding = nn.Embedding(vocab_size, width)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        # initialization
        nn.init.normal_(self.positional_embedding, std=0.01)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float('-inf'))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, *, as_dict=False):
        x = self.token_embedding(text)
        outs = Result(as_dict=as_dict)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        outs.append(x, name='x')

        return outs.as_return()


def getWeight(weight_path=r'F:\Projects\模型训练合集\weight\groupvit\group_vit_gcc_redcap_30e-3dd09a76.pth'):
    # **分层导入预训练参数**
    checkpoint = torch.load(weight_path)
    checkpoint_dict = {}
    for key in checkpoint['model'].keys():
        if 'text_encoder.' in key:
            # print(key)
            checkpoint_dict[f'{key}'.replace('text_encoder.', '')] = checkpoint['model'][key]

    return checkpoint_dict


def encode_text(text, *, as_dict=False):
    assert text.ndim in [2, 3], text.ndim
    squeeze_dim = False
    num_text = 1
    if text.ndim == 3:
        num_text = text.shape[1]
        text = rearrange(text, 'b n l -> (b n) l', n=num_text)
        squeeze_dim = True

    outs = Result(as_dict=as_dict)
    # [B, C]
    # TODO
    text_encoder = TextTransformer()
    text_projector = ProjectMLP()
    x = text_encoder(text)
    text_x = text_projector(x)

    outs.append(text_x, 'text_x')
    if squeeze_dim:
        text_x = rearrange(text_x, '(b n) c -> b n c', n=num_text)
        text_multi_label_x = text_x[:, 1:]
        text_x = text_x[:, 0]
        outs.update(text_x=text_x, text_multi_label_x=text_multi_label_x)

    return outs.as_return()


# Multi_Label_Contrastive
class ProjectMLP(nn.Module):

    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(ProjectMLP, self).__init__()
        # text in_dim = 256
        # image in_dim = 384

        # hidden layers
        linear_hidden = []
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Conv1d(in_dim if i == 0 else inner_dim, inner_dim, kernel_size=1))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Conv1d(
            in_dim if num_layers == 1 else inner_dim, out_dim, kernel_size=1) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): output of transformers, shape [B, L, C]

        Returns:

        """
        assert x.ndim in [2, 3], x.ndim
        add_dim = False
        if x.ndim == 2:
            # [B, C] -> [B, L, C]
            x = x.unsqueeze(1)
            add_dim = True

        x = rearrange(x, 'b l c -> b c l')
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        x = rearrange(x, 'b c l -> b l c')

        if add_dim:
            x = x.squeeze(1)

        return x


class ProjectMLP_noBN(nn.Module):

    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(ProjectMLP_noBN, self).__init__()
        # text in_dim = 256
        # image in_dim = 384

        # hidden layers
        linear_hidden = []
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Conv1d(in_dim if i == 0 else inner_dim, inner_dim, kernel_size=1))
            # linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Conv1d(
            in_dim if num_layers == 1 else inner_dim, out_dim, kernel_size=1) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): output of transformers, shape [B, L, C]

        Returns:

        """
        assert x.ndim in [2, 3], x.ndim
        add_dim = False
        if x.ndim == 2:
            # [B, C] -> [B, L, C]
            x = x.unsqueeze(1)
            add_dim = True

        x = rearrange(x, 'b l c -> b c l')
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        x = rearrange(x, 'b c l -> b l c')

        if add_dim:
            x = x.squeeze(1)

        return x


def build_text_embedding(text):
    """

    Args:
        text (torch.Tensor): [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH]

    Returns:

    """
    # text = text.cuda()
    num_classes, num_templates = text.shape[:2]
    text = rearrange(text, 'n t l -> (n t) l', n=num_classes, t=num_templates)
    text_tokens = encode_text(text)
    # [N, T, C]
    text_tokens = rearrange(text_tokens, '(n t) c -> n t c', n=num_classes, t=num_templates)
    # [N, C]
    # TODO：_t防止权重变化
    text_tokens_t = text_tokens.mean(dim=1)
    text_tokens = F.normalize(text_tokens_t, dim=-1)

    return text_tokens


def build_text_transform(is_train, config, with_dc=True):
    local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0
    if config.multi_label and is_train:
        # only down on local rank 0
        if local_rank == 0:
            nltk.download('popular')
        transform = WordAugTokenizeWrapper(
            Tokenize(SimpleTokenizer(), max_seq_len=config.max_seq_len),
            max_word=config.multi_label,
            word_type=config.word_type)

    else:
        transform = Tokenize(SimpleTokenizer(), max_seq_len=config.max_seq_len)

    # if with_dc:
    #     transform = transforms.Compose([transform, ToDataContainer()])

    return transform


class Tokenize:

    def __init__(self, tokenizer, max_seq_len=77, truncate=True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.truncate = truncate

    def __call__(self, texts):
        expanded_dim = False
        if isinstance(texts, str):
            texts = [texts]
            expanded_dim = True

        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), self.max_seq_len, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.max_seq_len:
                if self.truncate:
                    tokens = tokens[:self.max_seq_len]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f'Input {texts[i]} is too long for context length {self.max_seq_len}')
            result[i, :len(tokens)] = torch.tensor(tokens)

        if expanded_dim:
            return result[0]

        return result


class WordAugTokenizeWrapper:

    def __init__(self, tokenize, max_word=3, template_set='full', word_type='noun'):
        self.tokenize = tokenize
        self.max_word = max_word
        # from .imagenet_template import (full_imagenet_templates, sub_imagenet_template, simple_imagenet_template,
        #                                 identity_template)
        # assert template_set in ['full', 'subset', 'simple', 'identity']
        # if template_set == 'full':
        #     templates = full_imagenet_templates
        # elif template_set == 'subset':
        #     templates = sub_imagenet_template
        # elif template_set == 'simple':
        #     templates = simple_imagenet_template
        # elif template_set == 'identity':
        #     templates = identity_template
        # else:
        #     raise ValueError
        # self.templates = templates
        assert word_type in ['noun', 'noun_phrase']
        self.word_type = word_type

    def get_tag(self, tokenized, tags):
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        ret = []
        for (word, pos) in nltk.pos_tag(tokenized):
            for tag in tags:
                if pos == tag:
                    ret.append(word)
        return ret

    def get_noun_phrase(self, tokenized):
        # Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """
        chunker = nltk.RegexpParser(grammar)

        chunked = chunker.parse(nltk.pos_tag(tokenized))
        continuous_chunk = []
        current_chunk = []

        for subtree in chunked:
            if isinstance(subtree, nltk.Tree):
                current_chunk.append(' '.join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = ' '.join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk

    def __call__(self, text):
        assert isinstance(text, str)
        tokenized = nltk.word_tokenize(text)
        nouns = []
        if len(tokenized) > 0:
            if self.word_type == 'noun':
                nouns = self.get_tag(tokenized, ['NN', 'NNS', 'NNP', 'VBG', 'VB', 'VBD', 'VBN', 'VBP', 'VBZ'])
            elif self.word_type == 'noun_phrase':
                nouns = self.get_noun_phrase(tokenized)
            else:
                raise ValueError('word_type must be noun or noun_phrase')

        prompt_texts = []
        if len(nouns) > 0:
            select_nouns = np.random.choice(nouns, min(self.max_word, len(nouns)), replace=False)
            prompt_texts = [np.random.choice(self.templates).format(noun) for noun in select_nouns]
        if len(prompt_texts) < self.max_word:
            prompt_texts += [text] * (self.max_word - len(prompt_texts))

        texts = [text] + prompt_texts
        return self.tokenize(texts)


def build_dataset_class_tokens(text_transform, classnames):
    tokens = []
    templates = 'a photo of a {}.'
    for classname in classnames:
        # format with class
        sentence = templates.format(classname)
        sentence_ts = text_transform(sentence)
        tokens.append(torch.stack([sentence_ts]))

    # [N, T, L], N: number of instance, T: number of captions (including ensembled), L: sequence length
    tokens = torch.stack(tokens)

    return tokens


def main1():
    text_encoder = TextTransformer()
    print(text_encoder)

    checkpoint_dict = getWeight()
    text_encoder.load_state_dict(checkpoint_dict)
    pass


def get_text_embedding():
    # 根据类别创建token

    # classnames = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    #               'table', 'dog', 'horse', 'motorbike', 'person', 'plant', 'sheep', 'sofa', 'train', 'monitor')
    classnames = ('Impervious','Tree canopy','Low vegetation','Water')

    text_aug = {'max_seq_len': 77, 'multi_label': 0, 'word_type': 'noun'}
    cfg_text_aug = OmegaConf.create(text_aug)
    text_transform = build_text_transform(False, cfg_text_aug, with_dc=False)
    text_tokens = build_dataset_class_tokens(text_transform, classnames)

    # 根据token转化为embedding
    text_embedding = build_text_embedding(text=text_tokens)
    # print(text_embedding)
    text_embedding = text_embedding.cuda()
    return text_embedding
    pass



if __name__ == '__main__':
    # main1()
    get_text_embedding()

    pass
