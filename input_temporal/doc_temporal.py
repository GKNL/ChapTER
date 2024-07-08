import os
import json
import random
import torch
import torch.utils.data.dataset

from typing import Optional, List

from config import args
from .quadruple import reverse_quadruple
from .quadruple_mask import construct_mask, construct_self_negative_mask
from .dict_hub_temporal import get_entity_dict, get_tokenizer, get_timehistory_graph
from logger_config import logger

entity_dict = get_entity_dict()

if args.use_timehistory_graph:
    # make the lazy data loading happen
    get_timehistory_graph()

month2text_dict = {
    '01': ['January', 'Jan'],
    '02': ['February', 'Feb'],
    '03': ['March', 'Mar'],
    '04': ['April', 'Apr'],
    '05': ['May', 'May'],
    '06': ['June', 'Jun'],
    '07': ['July', 'Jul'],
    '08': ['August', 'Aug'],
    '09': ['September', 'Sep'],
    '10': ['October', 'Oct'],
    '11': ['November', 'Nov'],
    '12': ['December', 'Dec'],
}



def convert_time(time):
    if time == '':
        return time

    y, m, d = time.split('-')
    return month2text_dict[m][0] + '-' + d


def _custom_tokenize(text: str,
                     text_pair: Optional[str] = None) -> dict:
    tokenizer = get_tokenizer()
    encoded_inputs = tokenizer(text=text,
                               text_pair=text_pair if text_pair else None,
                               add_special_tokens=True,
                               max_length=args.max_num_tokens,
                               return_token_type_ids=True,
                               truncation=True)
    return encoded_inputs


def _parse_entity_name(entity: str) -> str:
    return entity or ''


def _concat_name_desc(entity: str, entity_desc: str) -> str:
    """
    Since name and desc are the same, we only keep one of them.
    """
    if entity_desc.startswith(entity):
        entity_desc = entity_desc[len(entity):].strip()
    if entity_desc:
        return '{}: {}'.format(entity, entity_desc)
    return entity


def get_historical_complete_quadruple(head_id: str, relation: str, time: str, tail_id: str = None) -> str:
    """
    search (tail_id, time) in timehistory graph
    return String in format "time head relation tail"
    """
    head_name = entity_dict.get_entity_by_id(head_id).entity
    head_name = _parse_entity_name(head_name)

    neighbor_ids_times = get_timehistory_graph().get_historical_neighbor_ids_times(head_id, relation, time)
    if len(neighbor_ids_times) == 0:
        return ""
    # avoid label leakage during training
    if not args.is_test:
        neighbor_ids_times = [x for x in neighbor_ids_times if x[0] != tail_id]
    entities = [entity_dict.get_entity_by_id(x[0]).entity for x in neighbor_ids_times]
    entities = [_parse_entity_name(entity) for entity in entities]
    times = [x[1].strftime('%Y-%m-%d') for x in neighbor_ids_times]
    # times = [x[1] for x in neighbor_ids_times]
    times = [convert_time(x) for x in times]
    concat_string = ''
    if len(entities) != 0:
        for i in range(len(entities) - 1):
            concat_string += times[i] + ' ' + head_name + ' ' + relation + ' ' + entities[i] + ' | '
        concat_string += times[-1] + ' ' + head_name + ' ' + relation + ' ' + entities[-1]
    return concat_string.strip()


class Example:

    def __init__(self, head_id, relation, tail_id, time, **kwargs):
        self.head_id = head_id
        self.tail_id = tail_id
        self.relation = relation
        self.time = time
        self.desc_time = convert_time(time)


    @property
    def head_desc(self):  # Only store name in train.txt.json, so need to get desc from entity_dict
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity_desc

    @property
    def tail_desc(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity_desc

    @property
    def head(self):
        if not self.head_id:
            return ''
        return entity_dict.get_entity_by_id(self.head_id).entity

    @property
    def tail(self):
        return entity_dict.get_entity_by_id(self.tail_id).entity

    def vectorize_timehistory(self) -> dict:
        head_desc, tail_desc = self.head_desc, self.tail_desc
        relation = self.relation
        head_history_context = ''
        tail_history_context = ''

        if args.use_timehistory_graph:
            if self.time != '':
                head_history_context += get_historical_complete_quadruple(head_id=self.head_id,
                                                                          relation=relation,
                                                                          time=self.time,
                                                                          tail_id=self.tail_id)


        head_desc = head_desc.strip()
        tail_desc = tail_desc.strip()
        head_history_context = head_history_context.strip()
        tail_history_context = tail_history_context.strip()


        head_word = _parse_entity_name(self.head)
        head_text = _concat_name_desc(head_word, head_desc)

        hr_input_text = self.desc_time + ' ' + head_text + ' | ' + self.relation
        hr_encoded_inputs = _custom_tokenize(text=hr_input_text,
                                             text_pair=head_history_context)


        head_input_text = self.desc_time + ' ' + head_text
        head_encoded_inputs = _custom_tokenize(text=head_input_text,
                                               text_pair=head_history_context)

        tail_word = _parse_entity_name(self.tail)
        tail_text = _concat_name_desc(tail_word, tail_desc)
        tail_input_text = tail_text
        tail_encoded_inputs = _custom_tokenize(tail_input_text)

        return {'hr_token_ids': hr_encoded_inputs['input_ids'],
                'hr_token_type_ids': hr_encoded_inputs['token_type_ids'],
                'tail_token_ids': tail_encoded_inputs['input_ids'],
                'tail_token_type_ids': tail_encoded_inputs['token_type_ids'],
                'head_token_ids': head_encoded_inputs['input_ids'],
                'head_token_type_ids': head_encoded_inputs['token_type_ids'],
                'obj': self}




class Dataset(torch.utils.data.dataset.Dataset):

    def __init__(self, path, task, examples=None):
        self.path_list = path.split(',')
        self.task = task
        assert all(os.path.exists(path) for path in self.path_list) or examples
        if examples:
            self.examples = examples
        else:
            self.examples = []
            for path in self.path_list:
                if not self.examples:
                    self.examples = load_data(path)  # load data from json
                else:
                    self.examples.extend(load_data(path))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index].vectorize_timehistory()



def load_data(path: str,
              add_forward_quadruple: bool = True,
              add_backward_quadruple: bool = True) -> List[Example]:
    assert path.endswith('.json'), 'Unsupported format: {}'.format(path)
    assert add_forward_quadruple or add_backward_quadruple
    logger.info('In test mode: {}'.format(args.is_test))

    data = json.load(open(path, 'r', encoding='utf-8'))
    logger.info('Load {} examples from {}'.format(len(data), path))

    cnt = len(data)
    examples = []
    for i in range(cnt):
        obj = data[i]  # head_id, head, relation, tail_id, tail, time, time_id
        if add_forward_quadruple:
            examples.append(Example(**obj))
        if add_backward_quadruple:
            examples.append(Example(**reverse_quadruple(obj)))
        data[i] = None

    return examples


def collate(batch_data: List[dict]) -> dict:
    hr_token_ids, hr_mask = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    hr_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['hr_token_type_ids']) for ex in batch_data],
        need_mask=False)

    tail_token_ids, tail_mask = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    tail_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['tail_token_type_ids']) for ex in batch_data],
        need_mask=False)

    head_token_ids, head_mask = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_ids']) for ex in batch_data],
        pad_token_id=get_tokenizer().pad_token_id)
    head_token_type_ids = to_indices_and_mask(
        [torch.LongTensor(ex['head_token_type_ids']) for ex in batch_data],
        need_mask=False)

    batch_exs = [ex['obj'] for ex in batch_data]
    batch_dict = {
        'hr_token_ids': hr_token_ids,
        'hr_mask': hr_mask,
        'hr_token_type_ids': hr_token_type_ids,
        'tail_token_ids': tail_token_ids,
        'tail_mask': tail_mask,
        'tail_token_type_ids': tail_token_type_ids,
        'head_token_ids': head_token_ids,
        'head_mask': head_mask,
        'head_token_type_ids': head_token_type_ids,
        'batch_data': batch_exs,
        'triplet_mask': construct_mask(row_exs=batch_exs) if not args.is_test else None,
        'self_negative_mask': construct_self_negative_mask(batch_exs) if not args.is_test else None,
    }

    return batch_dict


def to_indices_and_mask(batch_tensor, pad_token_id=0, need_mask=True):
    mx_len = max([t.size(0) for t in batch_tensor])
    batch_size = len(batch_tensor)
    indices = torch.LongTensor(batch_size, mx_len).fill_(pad_token_id)
    # For BERT, mask value of 1 corresponds to a valid position
    if need_mask:
        mask = torch.ByteTensor(batch_size, mx_len).fill_(0)
    for i, t in enumerate(batch_tensor):
        indices[i, :len(t)].copy_(t)
        if need_mask:
            mask[i, :len(t)].fill_(1)
    if need_mask:
        return indices, mask
    else:
        return indices
