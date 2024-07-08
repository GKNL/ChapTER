import os
import glob

from transformers import AutoTokenizer

from config import args
from .quadruple import TripletDict, FilterQuadrupleDict, EntityDict, TimeHistoryGraph
from logger_config import logger

train_triplet_dict: TripletDict = None
all_triplet_dict: TripletDict = None
all_quadruple_dict: FilterQuadrupleDict = None
timehistory_graph: TimeHistoryGraph = None
entity_dict: EntityDict = None
tokenizer: AutoTokenizer = None


def _init_entity_dict():
    global entity_dict
    if not entity_dict:
        entity_dict = EntityDict(entity_dict_dir=os.path.dirname(args.valid_path))


def _init_train_triplet_dict():
    global train_triplet_dict
    if not train_triplet_dict:
        train_triplet_dict = TripletDict(path_list=[args.train_path])


def _init_all_triplet_dict():
    global all_triplet_dict
    if not all_triplet_dict:
        path_pattern = '{}/*.txt.json'.format(os.path.dirname(args.train_path))
        all_triplet_dict = TripletDict(path_list=glob.glob(path_pattern))

def _init_all_quadruple_dict():
    global all_quadruple_dict
    if not all_quadruple_dict:
        path_pattern = '{}/*.txt.json'.format(os.path.dirname(args.train_path))
        all_quadruple_dict = FilterQuadrupleDict(path_list=glob.glob(path_pattern))



def _init_timehistory_graph():
    global timehistory_graph
    if not timehistory_graph:
        timehistory_graph = TimeHistoryGraph(train_path=args.train_path)


def get_entity_dict():
    _init_entity_dict()
    return entity_dict


def get_train_triplet_dict():
    _init_train_triplet_dict()
    return train_triplet_dict


def get_all_triplet_dict():
    _init_all_triplet_dict()
    return all_triplet_dict

def get_all_filter_quadruple_dict():
    _init_all_quadruple_dict()
    return all_quadruple_dict

def get_timehistory_graph():
    _init_timehistory_graph()
    return timehistory_graph


def build_tokenizer(args):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        logger.info('Build tokenizer from {}'.format(args.pretrained_model))


def get_tokenizer():
    if tokenizer is None:
        build_tokenizer(args)
    return tokenizer
