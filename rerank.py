import torch

from typing import List

from config import args
from input_temporal.quadruple import EntityDict
from input_temporal.dict_hub_temporal import get_timehistory_graph
from input_temporal.doc_temporal import Example


def rerank_by_graph(batch_score: torch.tensor,
                    examples: List[Example],
                    entity_dict: EntityDict):

    if args.neighbor_weight < 1e-6:
        return

    for idx in range(batch_score.size(0)):
        cur_ex = examples[idx]

        # Re-rank all n-hop neighbors
        n_hop_indices = get_timehistory_graph().get_n_hop_entity_indices(cur_ex.head_id,
                                                                  entity_dict=entity_dict,
                                                                  n_hop=args.rerank_n_hop)
        delta = torch.tensor([args.neighbor_weight for _ in n_hop_indices]).to(batch_score.device)
        n_hop_indices = torch.LongTensor(list(n_hop_indices)).to(batch_score.device)

        batch_score[idx].index_add_(0, n_hop_indices, delta)


        # Re-rank all one-hop relation-related neighbors
        one_hop_hr_indices = get_timehistory_graph().get_one_hop_entity_rel_indices(entity_id=cur_ex.head_id,
                                                                                    relation=cur_ex.relation,
                                                                                    size_to_left=0,
                                                                                    entity_dict=entity_dict)
        delta2 = torch.tensor([0.1 for _ in one_hop_hr_indices]).to(batch_score.device)
        one_hop_hr_indices = torch.LongTensor(list(one_hop_hr_indices)).to(batch_score.device)

        batch_score[idx].index_add_(0, one_hop_hr_indices, delta2)
