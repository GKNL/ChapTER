from abc import ABC
from copy import deepcopy

import torch
import torch.nn as nn

from dataclasses import dataclass
from transformers import AutoModel, AutoConfig
# from transformers import BertConfig

from input_temporal.quadruple_mask import construct_mask
from ptuning_models.prefix_encoder import PrefixEncoder


def build_model(args) -> nn.Module:
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model)
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)

        # Bert models
        self.hr_bert = AutoModel.from_pretrained(args.pretrained_model)
        self.tail_bert = deepcopy(self.hr_bert)

        # Freeze PLM parameters
        if args.prompt_length > 0:
            for p in self.hr_bert.parameters():
                p.requires_grad = False
            for k in self.tail_bert.parameters():
                k.requires_grad = False

        self.n_layer = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size // self.config.num_attention_heads

        # Prompt Embedding
        self.config.prefix_projection = args.prefix_projection
        self.config.prompt_length = args.prompt_length
        self.config.prefix_hidden_size = args.prompt_hidden_dim

        self.prefix_tokens_hr = torch.arange(args.prompt_length).long()
        self.prefix_tokens_tail = torch.arange(args.prompt_length).long()
        self.prefix_encoder_hr = PrefixEncoder(self.config)
        self.prefix_encoder_tail = PrefixEncoder(self.config)

        bert_param = 0
        for name, param in self.hr_bert.named_parameters():
            bert_param += param.numel()
        for name, param in self.tail_bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        tuning_param = all_param - bert_param
        print('Total param is {}'.format(all_param))
        print('Total tuning param is {}'.format(tuning_param))
        print('Total bert param is {}'.format(bert_param))

    def get_prompt_hr(self, encoder, batch_size):
        prefix_tokens = self.prefix_tokens_hr.unsqueeze(0).expand(batch_size, -1).to(encoder.device)
        # 得到连续Prompt
        past_key_values = self.prefix_encoder_hr(prefix_tokens)
        # 改变形状
        past_key_values = past_key_values.view(
            batch_size,
            self.args.prompt_length,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_prompt_tail(self, encoder, batch_size):
        prefix_tokens = self.prefix_tokens_tail.unsqueeze(0).expand(batch_size, -1).to(encoder.device)
        past_key_values = self.prefix_encoder_tail(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.args.prompt_length,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def _encode(self, encoder, token_ids, mask, token_type_ids, past_key_values):
        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True,
                          past_key_values=past_key_values)

        last_hidden_state = outputs.last_hidden_state
        cls_output = outputs.pooler_output  # outputs[1]
        cls_output = self.dropout(cls_output)
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)
        return cls_output

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                only_ent_embedding=False, **kwargs) -> dict:

        batch_size = hr_token_ids.size(0)
        hr_past_key_values = self.get_prompt_hr(self.hr_bert, batch_size=batch_size)
        tail_past_key_values = self.get_prompt_tail(self.tail_bert, batch_size=batch_size)

        # Concat prompt mask
        prefix_attention_mask = torch.ones(batch_size, self.config.prompt_length).to(self.hr_bert.device)
        hr_attention_mask = torch.cat((prefix_attention_mask, hr_mask), dim=1)
        tail_attention_mask = torch.cat((prefix_attention_mask, tail_mask), dim=1)
        head_attention_mask = torch.cat((prefix_attention_mask, head_mask), dim=1)

        if only_ent_embedding:
            return self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                              tail_mask=tail_attention_mask,
                                              tail_token_type_ids=tail_token_type_ids,
                                              past_key_values=tail_past_key_values)

        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_attention_mask,
                                 token_type_ids=hr_token_type_ids,
                                 past_key_values=hr_past_key_values)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_attention_mask,
                                   token_type_ids=tail_token_type_ids,
                                   past_key_values=tail_past_key_values)

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_attention_mask,
                                   token_type_ids=head_token_type_ids,
                                   past_key_values=tail_past_key_values)

        # DataParallel only support tensor/dict
        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)

        return {'logits': logits,
                'labels': labels,
                'inv_t': self.log_inv_t.detach().exp(),
                'hr_vector': hr_vector.detach(),
                'tail_vector': tail_vector.detach()}

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, past_key_values, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids,
                                   past_key_values=past_key_values)
        return {'ent_vectors': ent_vectors.detach()}


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        sum_embeddings = torch.sum(last_hidden_state, 1)
        seq_lens = last_hidden_state.size(1)
        output_vector = sum_embeddings / seq_lens
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
