import os
import json
import argparse
import multiprocessing as mp

from multiprocessing import Pool
from typing import List

parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--task', default='YAGO1830', type=str, metavar='N',
                    help='dataset name')
parser.add_argument('--workers', default=2, type=int, metavar='N',
                    help='number of workers')
parser.add_argument('--train-path', type=str, metavar='N',
                    help='path to training data')
parser.add_argument('--valid-path', type=str, metavar='N',
                    help='path to valid data')
parser.add_argument('--test-path', metavar='N',
                    help='path to valid data')

args = parser.parse_args()
mp.set_start_method('fork')


def _check_sanity(relation_id_to_str: dict):
    # We directly use normalized relation string as a key for training and evaluation,
    # make sure no two relations are normalized to the same surface form
    relation_str_to_id = {}
    for rel_id, rel_str in relation_id_to_str.items():
        if rel_str is None:
            continue
        if rel_str not in relation_str_to_id:
            relation_str_to_id[rel_str] = rel_id
        elif relation_str_to_id[rel_str] != rel_id:
            assert False, 'ERROR: {} and {} are both normalized to {}'\
                .format(relation_str_to_id[rel_str], rel_id, rel_str)
    return


def _normalize_relations(examples: List[dict], normalize_fn, is_train: bool):
    """
    Convert relation string to a normalized form
    "/location/country/form_of_government": "form of government country location ",
    """
    relation_id_to_str = {}
    for ex in examples:
        rel_str = normalize_fn(ex['relation'])
        relation_id_to_str[ex['relation']] = rel_str
        ex['relation'] = rel_str

    _check_sanity(relation_id_to_str)

    if is_train:
        out_path = '{}/relations.json'.format(os.path.dirname(args.train_path))
        with open(out_path, 'w', encoding='utf-8') as writer:
            json.dump(relation_id_to_str, writer, ensure_ascii=False, indent=4)
            print('Save {} relations to {}'.format(len(relation_id_to_str), out_path))


"""
Preprocess ICEWS18 dataset  ------------------------------------------------------------------
"""

icews18_id2ent = {}
icews18_id2time = {}
icews18_id2rel = {}


def _load_icews18_texts(path: str):
    global icews18_id2ent
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        entity_id, word, desc = fs[1], fs[0], fs[0]
        icews18_id2ent[entity_id] = (entity_id, word, desc)
    print('Load {} entities from {}'.format(len(icews18_id2ent), path))


def _load_icews18_time(path: str):
    global icews18_id2time
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        time_id, time = fs[1], fs[0]
        icews18_id2time[time_id] = (time_id, time)
    print('Load {} entities from {}'.format(len(icews18_id2time), path))


def _load_icews18_rel(path: str):
    global icews18_id2rel
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        relation_id, relation = fs[1], fs[0]
        icews18_id2rel[relation_id] = (relation_id, relation)
    print('Load {} entities from {}'.format(len(icews18_id2rel), path))


def _process_line_icews18(line: str) -> dict:
    fs = line.strip().split('\t')
    head_id, relation_id, tail_id, time_id = fs[0].strip(), fs[1].strip(), fs[2].strip(), fs[3].strip()
    _, head, _ = icews18_id2ent[head_id]
    _, tail, _ = icews18_id2ent[tail_id]
    _, relation = icews18_id2rel[relation_id]
    _, time = icews18_id2time[time_id]
    example = {'head_id': head_id,
               'head': head,
               'relation': relation,
               'tail_id': tail_id,
               'tail': tail,
               'time_id': time_id,
               'time': time}
    return example


def preprocess_icews18(path):
    if not icews18_id2ent:
        _load_icews18_texts('{}/entity2id.txt'.format(os.path.dirname(path)))
    if not icews18_id2rel:
        _load_icews18_rel('{}/relation2id.txt'.format(os.path.dirname(path)))
    if not icews18_id2time:
        _load_icews18_time('{}/time2id.txt'.format(os.path.dirname(path)))
    lines = open(path, 'r', encoding='utf-8').readlines()
    pool = Pool(processes=args.workers)
    examples = pool.map(_process_line_icews18, lines)
    pool.close()
    pool.join()

    # relation name is standard, no need to process
    _normalize_relations(examples, normalize_fn=lambda rel: rel.strip(),
                         is_train=(path == args.train_path))

    out_path = path + '.json'
    json.dump(examples, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('Save {} examples to {}'.format(len(examples), out_path))
    return examples


"""
Preprocess ICEWS0515 dataset  ------------------------------------------------------------------
"""

icews0515_id2ent = {}
icews0515_id2time = {}
icews0515_id2rel = {}


def _load_icews0515_texts(path: str):
    global icews0515_id2ent
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        entity_id, word, desc = fs[1], fs[0], fs[0]
        word = word.replace('_', ' ').strip()  # remove underscore
        desc = desc.replace('_', ' ').strip()  # remove underscore
        icews0515_id2ent[entity_id] = (entity_id, word, desc)
    print('Load {} entities from {}'.format(len(icews0515_id2ent), path))


def _load_icews0515_time(path: str):
    global icews0515_id2time
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        time_id, time = fs[1], fs[0]
        icews0515_id2time[time_id] = (time_id, time)
    print('Load {} entities from {}'.format(len(icews0515_id2time), path))


def _load_icews0515_rel(path: str):
    global icews0515_id2rel
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        relation_id, relation = fs[1], fs[0]
        relation = relation.replace('_', ' ').strip()  # remove underscore
        icews0515_id2rel[relation_id] = (relation_id, relation)
    print('Load {} entities from {}'.format(len(icews0515_id2rel), path))


def _process_line_icews0515(line: str) -> dict:
    fs = line.strip().split('\t')
    head_id, relation_id, tail_id, time_id = fs[0].strip(), fs[1].strip(), fs[2].strip(), fs[3].strip()
    _, head, _ = icews0515_id2ent[head_id]
    _, tail, _ = icews0515_id2ent[tail_id]
    _, relation = icews0515_id2rel[relation_id]
    _, time = icews0515_id2time[time_id]
    example = {'head_id': head_id,
               'head': head,
               'relation': relation,
               'tail_id': tail_id,
               'tail': tail,
               'time_id': time_id,
               'time': time}
    return example


def preprocess_icews0515(path):
    if not icews0515_id2ent:
        _load_icews0515_texts('{}/entity2id.txt'.format(os.path.dirname(path)))
    if not icews0515_id2rel:
        _load_icews0515_rel('{}/relation2id.txt'.format(os.path.dirname(path)))
    if not icews0515_id2time:
        _load_icews0515_time('{}/time2id.txt'.format(os.path.dirname(path)))
    lines = open(path, 'r', encoding='utf-8').readlines()
    pool = Pool(processes=args.workers)
    examples = pool.map(_process_line_icews0515, lines)
    pool.close()
    pool.join()

    # relation name is standard, no need to process
    _normalize_relations(examples, normalize_fn=lambda rel: rel.strip(),
                         is_train=(path == args.train_path))

    out_path = path + '.json'
    json.dump(examples, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('Save {} examples to {}'.format(len(examples), out_path))
    return examples


"""
Preprocess ICEWS14 dataset  ------------------------------------------------------------------
"""

icews14_id2ent = {}
icews14_id2time = {}
icews14_id2rel = {}


def _load_icews14_texts(path: str):
    global icews14_id2ent
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        entity_id, word, desc = fs[1], fs[0], fs[0]
        word = word.replace('_', ' ').strip()  # remove underscore
        desc = desc.replace('_', ' ').strip()  # remove underscore
        icews14_id2ent[entity_id] = (entity_id, word, desc)
    print('Load {} entities from {}'.format(len(icews14_id2ent), path))


def _load_icews14_time(path: str):
    global icews14_id2time
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        time_id, time = fs[1], fs[0]
        icews14_id2time[time_id] = (time_id, time)
    print('Load {} entities from {}'.format(len(icews14_id2time), path))


def _load_icews14_rel(path: str):
    global icews14_id2rel
    lines = open(path, 'r', encoding='utf-8').readlines()
    for line in lines:
        fs = line.strip().split('\t')
        assert len(fs) == 2, 'Invalid line: {}'.format(line.strip())
        relation_id, relation = fs[1], fs[0]
        relation = relation.replace('_', ' ').strip()  # remove underscore
        icews14_id2rel[relation_id] = (relation_id, relation)
    print('Load {} entities from {}'.format(len(icews14_id2rel), path))


def _process_line_icews14(line: str) -> dict:
    fs = line.strip().split('\t')
    head_id, relation_id, tail_id, time_id = fs[0].strip(), fs[1].strip(), fs[2].strip(), fs[3].strip()
    _, head, _ = icews14_id2ent[head_id]
    _, tail, _ = icews14_id2ent[tail_id]
    _, relation = icews14_id2rel[relation_id]
    _, time = icews14_id2time[time_id]
    example = {'head_id': head_id,
               'head': head,
               'relation': relation,
               'tail_id': tail_id,
               'tail': tail,
               'time_id': time_id,
               'time': time}
    return example


def preprocess_icews14(path):
    if not icews14_id2ent:
        _load_icews14_texts('{}/entity2id.txt'.format(os.path.dirname(path)))
    if not icews14_id2rel:
        _load_icews14_rel('{}/relation2id.txt'.format(os.path.dirname(path)))
    if not icews14_id2time:
        _load_icews14_time('{}/time2id.txt'.format(os.path.dirname(path)))
    lines = open(path, 'r', encoding='utf-8').readlines()
    pool = Pool(processes=args.workers)
    examples = pool.map(_process_line_icews14, lines)
    pool.close()
    pool.join()

    # relation name is standard, no need to process
    _normalize_relations(examples, normalize_fn=lambda rel: rel.strip(),
                         is_train=(path == args.train_path))

    out_path = path + '.json'
    json.dump(examples, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    print('Save {} examples to {}'.format(len(examples), out_path))
    return examples


def dump_all_entities(examples, out_path, id2text: dict):
    id2entity = {}
    relations = set()
    for ex in examples:
        head_id = ex['head_id']
        relations.add(ex['relation'])
        if head_id not in id2entity:
            id2entity[head_id] = {'entity_id': head_id,
                                  'entity': ex['head'],
                                  'entity_desc': id2text[head_id]}
        tail_id = ex['tail_id']
        if tail_id not in id2entity:
            id2entity[tail_id] = {'entity_id': tail_id,
                                  'entity': ex['tail'],
                                  'entity_desc': id2text[tail_id]}
    print('Get {} entities, {} relations in total'.format(len(id2entity), len(relations)))

    json.dump(list(id2entity.values()), open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


def main():
    all_examples = []
    for path in [args.train_path, args.valid_path, args.test_path]:
        assert os.path.exists(path)
        print('Process {}...'.format(path))
        if args.task.lower() == 'icews18':
            all_examples += preprocess_icews18(path)
        elif args.task.lower() == 'icews0515':
            all_examples += preprocess_icews0515(path)
        elif args.task.lower() == 'icews14':
            all_examples += preprocess_icews14(path)
        else:
            assert False, 'Unknown task: {}'.format(args.task)

    if args.task.lower() == 'icews18':
        id2text = {k: v[2] for k, v in icews18_id2ent.items()}
    elif args.task.lower() == 'icews0515':
        id2text = {k: v[2] for k, v in icews0515_id2ent.items()}
    elif args.task.lower() == 'icews14':
        id2text = {k: v[2] for k, v in icews14_id2ent.items()}
    else:
        assert False, 'Unknown task: {}'.format(args.task)

    dump_all_entities(all_examples,
                      out_path='{}/entities.json'.format(os.path.dirname(args.train_path)),
                      id2text=id2text)
    print('Done')


if __name__ == '__main__':
    main()
