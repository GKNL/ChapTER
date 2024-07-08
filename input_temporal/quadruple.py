import math
import os
import json

from typing import List
from dataclasses import dataclass
from collections import deque

from logger_config import logger
from datetime import datetime
import random


@dataclass
class EntityExample:
    entity_id: str
    entity: str
    entity_desc: str = ''


class TripletDict:

    def __init__(self, path_list: List[str]):
        self.path_list = path_list
        logger.info('Triplets path: {}'.format(self.path_list))
        self.relations = set()
        self.hr2tails = {}  # Used to find neighbors
        self.triplet_cnt = 0

        for path in self.path_list:
            self._load(path)  # Load triplets from file
        logger.info('Triplet statistics: {} relations, {} triplets'.format(len(self.relations), self.triplet_cnt))

    def _load(self, path: str):
        examples = json.load(open(path, 'r', encoding='utf-8'))
        examples += [reverse_quadruple(obj) for obj in examples]
        for ex in examples:
            self.relations.add(ex['relation'])
            key = (ex['head_id'], ex['relation'])
            if key not in self.hr2tails:
                self.hr2tails[key] = set()
            self.hr2tails[key].add(ex['tail_id'])
        self.triplet_cnt = len(examples)

    def get_neighbors(self, h: str, r: str) -> set:
        return self.hr2tails.get((h, r), set())


class FilterQuadrupleDict:  # For time filter

    def __init__(self, path_list: List[str]):
        self.path_list = path_list
        logger.info('Quadruplets path: {}'.format(self.path_list))
        self.relations = set()
        self.hrt2tails = {}  # Used to find neighbors
        self.quadruplelet_cnt = 0

        for path in self.path_list:
            self._load(path)
        logger.info('Quadruplet statistics: {} relations, {} Quadruplets'.format(len(self.relations), self.quadruplelet_cnt))

    def _load(self, path: str):
        examples = json.load(open(path, 'r', encoding='utf-8'))
        examples += [reverse_quadruple(obj) for obj in examples]
        for ex in examples:
            self.relations.add(ex['relation'])
            key = (ex['head_id'], ex['relation'], ex['time'])
            if key not in self.hrt2tails:
                self.hrt2tails[key] = set()
            self.hrt2tails[key].add(ex['tail_id'])
        self.quadruplelet_cnt = len(examples)

    def get_neighbors(self, h: str, r: str, t: str) -> set:
        return self.hrt2tails.get((h, r, t), set())


class TimeDict:
    def __init__(self, test_path: str):
        assert os.path.exists(test_path)
        self.times = set()
        test_examples = json.load(open(test_path, 'r', encoding='utf-8'))
        for ex in test_examples:
            self.times.add(ex['time'])

        self.times = sorted(list(self.times))
        self.time2idx = {time: i for i, time in enumerate(self.times)}
        logger.info('Load {} times from {}'.format(len(self.times), test_path))

    def get_time_list(self):
        return list(self.times)

    def time_to_idx(self, time: str) -> int:
        return self.time2idx[time]


class EntityDict:

    def __init__(self, entity_dict_dir: str):
        path = os.path.join(entity_dict_dir, 'entities.json')
        assert os.path.exists(path)
        self.entity_exs = [EntityExample(**obj) for obj in json.load(open(path, 'r', encoding='utf-8'))]

        self.id2entity = {ex.entity_id: ex for ex in self.entity_exs}
        self.entity2idx = {ex.entity_id: i for i, ex in enumerate(self.entity_exs)}
        logger.info('Load {} entities from {}'.format(len(self.id2entity), path))

    def entity_to_idx(self, entity_id: str) -> int:
        return self.entity2idx[entity_id]

    def get_entity_by_id(self, entity_id: str) -> EntityExample:
        return self.id2entity[entity_id]

    def get_entity_by_idx(self, idx: int) -> EntityExample:
        return self.entity_exs[idx]

    def __len__(self):
        return len(self.entity_exs)


class TimeHistoryGraph:

    def __init__(self, train_path: str):
        logger.info('Start to build time history graph from {}'.format(train_path))
        # id -> set(id)
        self.link_graph = {}  # {head: tail}
        self.time_graph = {}  # {(head, relation): tail}
        self.head_time_graph = {}  # {head: (relation, tail, time)}
        examples = json.load(open(train_path, 'r', encoding='utf-8'))  # quadruplets
        for ex in examples:
            head_id, tail_id = ex['head_id'], ex['tail_id']
            relation = ex['relation']
            time = ex['time']
            # Build link graph
            if head_id not in self.link_graph:
                self.link_graph[head_id] = set()
            self.link_graph[head_id].add(tail_id)
            if tail_id not in self.link_graph:
                self.link_graph[tail_id] = set()
            self.link_graph[tail_id].add(head_id)

            # Build time graph
            query_id = (head_id, relation)
            query_id_rev = (tail_id, 'inverse {}'.format(relation))
            if query_id not in self.time_graph:
                self.time_graph[query_id] = set()
            self.time_graph[query_id].add((tail_id, time))
            if query_id_rev not in self.time_graph:
                self.time_graph[query_id_rev] = set()
            self.time_graph[query_id_rev].add((head_id, time))

            # Build head time graph
            if head_id not in self.head_time_graph:
                self.head_time_graph[head_id] = set()
            self.head_time_graph[head_id].add((relation, tail_id, time))
            if tail_id not in self.head_time_graph:
                self.head_time_graph[tail_id] = set()
            self.head_time_graph[tail_id].add(('inverse {}'.format(relation), head_id, time))
        logger.info('Done build time history graph with {} nodes'.format(len(self.time_graph)))

    def get_neighbor_ids(self, entity_id: str, max_to_keep=10) -> List[str]:
        # make sure different calls return the same results
        neighbor_ids = self.link_graph.get(entity_id, set())
        return sorted(list(neighbor_ids))[:max_to_keep]

    def get_historical_neighbor_ids_times(self, entity_id: str, relation: str, time: str, max_to_keep=15):
        """
        Get neighbor (tail_id, time) before the given time
        """
        # make sure different calls return the same results
        neighbor_ids = self.time_graph.get((entity_id, relation), set())
        time = datetime.strptime(time, "%Y-%m-%d")
        res_ids_times = []
        for e_id, t in neighbor_ids:
            t = datetime.strptime(t, "%Y-%m-%d")
            if t <= time:  # 相同时间或者更早的
                res_ids_times.append((e_id, t))
        # sort by time
        # res = sorted(res_ids_times, key=lambda x: x[1])[:max_to_keep]
        res = sorted(res_ids_times, key=lambda x: x[1], reverse=True)[:max_to_keep]
        # random.shuffle(res)  # Ablation on history shuffle
        return res

    def get_n_hop_entity_indices(self, entity_id: str,
                                 entity_dict: EntityDict,
                                 n_hop: int = 2,
                                 # return empty if exceeds this number
                                 max_nodes: int = 100000) -> set:
        if n_hop < 0:
            return set()

        seen_eids = set()
        seen_eids.add(entity_id)
        queue = deque([entity_id])
        for i in range(n_hop):
            len_q = len(queue)
            for _ in range(len_q):
                tp = queue.popleft()
                for node in self.link_graph.get(tp, set()):
                    if node not in seen_eids:
                        queue.append(node)
                        seen_eids.add(node)
                        if len(seen_eids) > max_nodes:
                            return set()
        return set([entity_dict.entity_to_idx(e_id) for e_id in seen_eids])

    def get_one_hop_entity_rel_indices(self, entity_id: str,
                                            relation: str,
                                            entity_dict: EntityDict,
                                            # return empty if exceeds this number
                                            size_to_left = 0.3,  # percentage to abandon
                                            max_nodes: int = 100000) -> set:

        seen_eids = set()
        seen_eids.add(entity_id)

        # find neighbors of same head,relation to filter out
        neighbors = self.time_graph.get((entity_id, relation), set())
        # filter out neighbors occur most recently
        neighbors = sorted(neighbors, key=lambda x: x[1])
        neighbor_length = len(neighbors)
        # cut back
        # max_to_keep = neighbor_length - math.ceil(size_to_left * neighbor_length)
        # neighbors = neighbors[:max_to_keep]
        # cut front
        keep_start_idx = math.ceil(size_to_left * neighbor_length)
        neighbors = neighbors[keep_start_idx:]
        neighbors = [x[0] for x in neighbors]
        neighbors = list(set(neighbors))  # remove duplicates

        return set([entity_dict.entity_to_idx(e_id) for e_id in neighbors])


def reverse_quadruple(obj):
    return {
        'head_id': obj['tail_id'],
        'head': obj['tail'],
        'relation': 'inverse {}'.format(obj['relation']),
        'tail_id': obj['head_id'],
        'tail': obj['head'],
        'time_id': obj['time_id'],
        'time': obj['time'],
    }
