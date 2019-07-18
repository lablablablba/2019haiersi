from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss, Softmax, NLLLoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, bert_context_classification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

logger = logging.getLogger(__name__)

# ---------------
from typing import List

eval_result = []
look_take = 2
threshold_multi_sent = 0.8
k_zhe = 5
use_k_zhe = True
train_k_zhe_pred = []
all_eval_pred = []
# 服务器跑的话，这里得改改
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    # line = list(unicode(cell, 'utf-8') for cell in line)
                    line = list(cell.decode('utf-8') for cell in line)
                lines.append(line)
            return lines


class HaiersiProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self.load_xml(os.path.join(data_dir, "train.xml"), "train")

    def get_dev_examples(self, data_dir):
        return self.load_xml(os.path.join(data_dir, "dev.xml"), "dev")

    def get_test_examples(self, data_dir):
        return self.load_xml(os.path.join(data_dir, "test.xml"), "test")

    def get_labels(self):
        return ["0", "1", "2"]

    def load_xml(self, dir, set_type):
        from xml.dom.minidom import parse

        examples = []
        path = os.path.abspath(dir)
        dom = parse(path)
        root = dom.documentElement
        doc_list = root.getElementsByTagName('Doc')
        k = 0
        for doc in doc_list:
            sent_list = doc.getElementsByTagName('Sentence')
            all_sents = []
            query_sents = []
            kk = 0
            for sent in sent_list:
                sentence = sent.firstChild.data
                label = sent.getAttribute("label")
                all_sents.append(sentence)
                # ---------------------------------------
                if set_type == "test":
                    label = "0"
                    query_sents.append((sentence, label, kk))
                else:
                    if label != "":
                        if label == "0" or label == "1" or label == "2":
                            query_sents.append((sentence, label, kk))
                kk += 1
            for query in query_sents:
                sentence = query[0]
                label = query[1]
                current_no = query[2]
                guid = "%s-%s" % (set_type, k)
                #
                max_len = len(all_sents) - 1
                # near_sentence = sentence if current_no == 0 else all_sents[current_no - 1] + sentence
                # if current_no < max_len: near_sentence = near_sentence + all_sents[current_no + 1]
                # examples.append(InputExample(guid=guid, text_a=sentence, text_b=near_sentence, label=label))
                # examples.append(InputExample(guid=guid, text_a=sentence, text_b=None, label=label))
                # examples.append(InputExample(guid=guid, text_a=near_sentence, text_b=None, label=label))
                # -----------------头和尾取句子数-------------------

                start_pos = max(0, current_no - look_take)
                end_pos = min(max_len, current_no + look_take)
                relative_pos = current_no - start_pos
                # examples.append(multi_sent_in_examples(guid=guid, sents=all_sents, no_target=current_no, label=label))
                examples.append(
                    multi_sent_in_examples(guid=guid, sents=all_sents[start_pos:end_pos + 1], no_target=relative_pos,
                                           label=label))
                # print(len(all_sents[start_pos:end_pos + 1]),relative_pos)
                k += 1
        print("loading", str(set_type), "set, total", str(len(examples)))
        return examples

    def eval_ana(self, data_dir):
        from xml.dom.minidom import parse

        examples = []
        path = os.path.abspath(data_dir)
        dom = parse(path)
        root = dom.documentElement
        doc_list = root.getElementsByTagName('Doc')
        k = 0
        for doc in doc_list:
            sent_list = doc.getElementsByTagName('Sentence')
            all_sents = []
            query_sents = []
            label_sents = []
            for sent in sent_list:
                sentence = sent.firstChild.data
                all_sents.append(sentence)
                if sent.getAttribute("label") != "":
                    query_sents.append(sentence)
                    label_sents.append(sent.getAttribute("label").replace('"', ""))
            for query in zip(query_sents, label_sents):
                sentence = query[0]
                label = query[1]
                guid = "%s-%s" % ("eval", k)
                examples.append((guid, sentence, label, all_sents))
                k += 1
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            pass
            # logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            pass
            # logger.info("*** Example ***")
            # logger.info("guid: %s" % (example.guid))
            # logger.info("tokens: %s" % " ".join(
            #     [str(x) for x in tokens]))
            # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            # logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            # logger.info(
            #     "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
    return features


class multi_sent_in_examples:
    def __init__(self, guid, sents, no_target, label=None):
        self.guid = guid
        self.sents = sents
        self.no_target = no_target  # 编号得从0开始
        self.label = label


def examples2features_sents(examples: List[multi_sent_in_examples], label_list, max_seq_length,
                            tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    index_cls = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            pass
            # logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_arr = list(map(lambda x: tokenizer.tokenize(x), example.sents))
        trucated_tokens_arr = _truncate_multi_seq(tokens_arr, max_seq_length - 2 * len(tokens_arr))
        from functools import reduce
        tokens = list(reduce(lambda x, y: x + ["[SEP]"] + ["[CLS]"] + y, trucated_tokens_arr))
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        segment_ids = list(
            reduce(lambda a, b: a + b,
                   list(map(lambda x: [x[0] % 2] * (len(x[1]) + 2), enumerate(trucated_tokens_arr)))))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        # print(tokens)
        # print(trucated_tokens_arr)
        # print(segment_ids)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

        index_cls.append(
            reduce(lambda a, b: a + b,
                   list(map(lambda x: len(x) + 2, trucated_tokens_arr[:example.no_target])))
            if len(trucated_tokens_arr[:example.no_target]) > 0 else 0)  # 确认一下这里是不是需要+1
    return features, index_cls


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def _truncate_multi_seq(tokens_arr, max_length):
    from functools import reduce
    while True:
        if reduce(lambda x, y: x + y, list(map(lambda e: len(e), tokens_arr))) <= max_length: break
        max = reduce(lambda a, b: a if len(a[1]) > len(b[1]) else b, list(enumerate(tokens_arr)))
        tokens_arr[max[0]].pop()
    return tokens_arr


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        # return {"mcc": matthews_corrcoef(labels, preds)}
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "haiersi":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def parser_init_para(parser):
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=4.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    return args


def simple_main():
    parser = argparse.ArgumentParser()
    args = parser_init_para(parser)
    processors = {"haiersi": HaiersiProcessor, }
    output_modes = {"haiersi": "classification", }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples, num_train_optimization_steps = train_configs(args, processor)

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    # model = BertForSequenceClassification.from_pretrained(args.bert_model,
    #                                                       cache_dir=cache_dir,
    #                                                       num_labels=num_labels)

    if use_k_zhe:
        for k_th in range(k_zhe):
            # 每次重新读入model
            model = bert_context_classification.from_pretrained(args.bert_model,
                                                                cache_dir=cache_dir,
                                                                num_labels=num_labels)  # modify
            model.to(device)
            if args.local_rank != -1:
                try:
                    from apex.parallel import DistributedDataParallel as DDP
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
                model = DDP(model)
            elif n_gpu > 1:
                model = torch.nn.DataParallel(model)
            # Prepare optimizer
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            global_step = 0
            nb_tr_steps = 0
            tr_loss = 0
            # -------------------------------------------------
            train_model_multi_sent(args, optimizer_grouped_parameters, num_train_optimization_steps,
                                   train_examples, label_list, tokenizer, output_mode, model, device,
                                   num_labels, n_gpu, global_step, processor, task_name, k_th)
        # 处理并输出预测值
        import pickle
        global train_k_zhe_pred, all_eval_pred
        pickle.dump(train_k_zhe_pred, open(os.path.join(args.output_dir, "oof_train.pkl"), "wb"))

        from functools import reduce
        all_eval_pred = reduce(lambda ele1, ele2: ele1 + ele2, list(map(lambda x: np.array(x), all_eval_pred))) / len(
            all_eval_pred)
        all_eval_pred = list(map(lambda x: list(x), all_eval_pred))
        pickle.dump(all_eval_pred, open(os.path.join(args.output_dir, "oof_test.pkl"), "wb"))
    else:
        # 读入model
        model = bert_context_classification.from_pretrained(args.bert_model,
                                                            cache_dir=cache_dir,
                                                            num_labels=num_labels)  # modify
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        # -------------------------------------------------
        train_model_multi_sent(args, optimizer_grouped_parameters, num_train_optimization_steps,
                               train_examples, label_list, tokenizer, output_mode, model, device,
                               num_labels, n_gpu, global_step, processor, task_name, 0)


def train_configs(args, processor):
    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    return train_examples, num_train_optimization_steps


def train_split_k_zhe(train_examples, k, k_total):
    """k=[0, k_total)"""
    length = len(train_examples)
    part_len = int(length / k_total)
    big_zhe = train_examples[0:k * part_len] + train_examples[(k + 1) * part_len:length]
    small_zhe = train_examples[k * part_len:(k + 1) * part_len] if k < k_total - 1 else train_examples[
                                                                                        k * part_len:length]
    print("big train set", len(big_zhe))
    print("small train set", len(small_zhe))
    return big_zhe, small_zhe


# def train_model(args, optimizer_grouped_parameters, num_train_optimization_steps,
#                 train_examples, label_list, tokenizer, output_mode, model, device,
#                 num_labels, n_gpu, global_step, processor, task_name):
#     if args.do_train:
#         optimizer = BertAdam(optimizer_grouped_parameters,
#                              lr=args.learning_rate,
#                              warmup=args.warmup_proportion,
#                              t_total=num_train_optimization_steps)
#         # ---------------------------------
#         train_features = convert_examples_to_features(
#             train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
#         logger.info("***** Running training *****")
#         logger.info("  Num examples = %d", len(train_examples))
#         logger.info("  Batch size = %d", args.train_batch_size)
#         logger.info("  Num steps = %d", num_train_optimization_steps)
#         all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
#         all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
#         all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
#
#         if output_mode == "classification":
#             all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
#         elif output_mode == "regression":
#             all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
#
#         train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
#         if args.local_rank == -1:
#             train_sampler = RandomSampler(train_data)
#         else:
#             train_sampler = DistributedSampler(train_data)
#         train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
#
#         model.train()
#
#         for _ in trange(int(args.num_train_epochs), desc="Epoch"):
#             tr_loss = 0
#             nb_tr_examples, nb_tr_steps = 0, 0
#             num_currect = 0.
#             for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
#                 batch = tuple(t.to(device) for t in batch)
#                 input_ids, input_mask, segment_ids, label_ids = batch
#
#                 # define a new function to compute loss values for both output_modes
#                 logits = model(input_ids, segment_ids, input_mask, labels=None)
#
#                 # 直接默认classify
#                 # 这里开始魔改
#                 # logit: (batch_size, num_classes)
#                 sm_func = Softmax(dim=1)
#                 prob = sm_func(logits)
#                 threshold = 0.8
#                 mask = torch.le(prob, threshold).float().to(device)
#                 loss_func = NLLLoss()
#                 loss = loss_func(torch.log(prob).view(-1, num_labels) * mask, label_ids.view(-1))
#                 # 计算正确个数
#                 preds = np.argmax(prob.cpu().detach().numpy(), axis=1)
#                 label = label_ids.cpu().detach().numpy()
#                 for i in range(len(preds)):
#                     if preds[i] == label[i]:
#                         num_currect += 1
#
#                 if n_gpu > 1:
#                     loss = loss.mean()  # mean() to average on multi-gpu.
#                 if args.gradient_accumulation_steps > 1:
#                     loss = loss / args.gradient_accumulation_steps
#
#                 loss.backward()
#
#                 tr_loss += loss.item()
#                 nb_tr_examples += input_ids.size(0)
#                 nb_tr_steps += 1
#                 if (step + 1) % args.gradient_accumulation_steps == 0:
#                     optimizer.step()
#                     optimizer.zero_grad()
#                     global_step += 1
#             logger.info("")
#             logger.info("***** train results *****")
#             logger.info("\tloss = %s", str(tr_loss / nb_tr_steps))
#             logger.info("\taccuracy = %s", str(num_currect / nb_tr_examples))
#
#             eval_model(processor, args, label_list, tokenizer, output_mode,
#                        model, device, num_labels, task_name, tr_loss,
#                        nb_tr_steps, global_step)
#         # ------------------------------------
#         # 我选择不存.jpg
#         # # Save a trained model, configuration and tokenizer
#         # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
#         #
#         # # If we save using the predefined names, we can load using `from_pretrained`
#         # output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
#         # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
#         #
#         # torch.save(model_to_save.state_dict(), output_model_file)
#         # model_to_save.config.to_json_file(output_config_file)
#         # tokenizer.save_vocabulary(args.output_dir)
#         #
#         # # Load a trained model and vocabulary that you have fine-tuned
#         # model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
#         # tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
#
#
# def eval_model(processor, args, label_list, tokenizer, output_mode,
#                model, device, num_labels, task_name, tr_loss,
#                nb_tr_steps, global_step):
#     eval_examples = processor.get_dev_examples(args.data_dir)  # 读数据
#     eval_features = convert_examples_to_features(
#         eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)
#     logger.info("***** Running evaluation *****")
#     logger.info("  Num examples = %d", len(eval_examples))
#     logger.info("  Batch size = %d", args.eval_batch_size)
#     all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
#     all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
#     all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
#
#     if output_mode == "classification":
#         all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
#     elif output_mode == "regression":
#         all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
#
#     eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
#     # Run prediction for full data
#     eval_sampler = SequentialSampler(eval_data)
#     eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
#
#     model.eval()
#     eval_loss = 0
#     nb_eval_steps = 0
#     preds = []
#
#     for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
#         input_ids = input_ids.to(device)
#         input_mask = input_mask.to(device)
#         segment_ids = segment_ids.to(device)
#         label_ids = label_ids.to(device)
#
#         with torch.no_grad():
#             logits = model(input_ids, segment_ids, input_mask, labels=None)
#
#         # create eval loss and other metric required by the task
#         if output_mode == "classification":
#             loss_fct = CrossEntropyLoss()
#             tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
#         elif output_mode == "regression":
#             loss_fct = MSELoss()
#             tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
#
#         eval_loss += tmp_eval_loss.mean().item()
#         nb_eval_steps += 1
#         if len(preds) == 0:
#             preds.append(logits.detach().cpu().numpy())
#         else:
#             preds[0] = np.append(
#                 preds[0], logits.detach().cpu().numpy(), axis=0)
#
#     eval_loss = eval_loss / nb_eval_steps
#     preds = preds[0]
#     if output_mode == "classification":
#         preds = np.argmax(preds, axis=1)
#     elif output_mode == "regression":
#         preds = np.squeeze(preds)
#     result = compute_metrics(task_name, preds, all_label_ids.numpy())
#     loss = tr_loss / nb_tr_steps if args.do_train else None
#
#     # 展示在验证集上的结果
#     result['eval_loss'] = eval_loss
#     result['global_step'] = global_step
#     result['loss'] = loss
#
#     logger.info("***** Eval results *****")
#     for key in sorted(result.keys()):
#         logger.info("  %s = %s", key, str(result[key]))
#
#     # # 输出错误编号
#     # with open(os.path.join(args.output_dir, "misjudge.txt"), "w", encoding="utf-8") as out_file:
#     #     label = all_label_ids.numpy()
#     #     full_data = processor.eval_ana(os.path.join(args.data_dir, "dev.xml"))
#     #     for i in range(len(preds)):
#     #         if preds[i] != label[i]:
#     #             current = full_data[i]
#     #             guid = current[0]
#     #             sentence = current[1]
#     #             doc = current[3]
#     #             for sent in doc:
#     #                 if sent != sentence:
#     #                     out_file.write("\t\t" + sent + "\n")
#     #                 else:
#     #                     out_file.write(str(preds[i]) + "\t" + str(label[i]) + "\t" + sent + "\n")
#     #             out_file.write("-" * 50 + "\n")
#     #
#     # output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
#     # with open(output_eval_file, "w") as writer:
#     #
#     #         writer.write("%s = %s\n" % (key, str(result[key])))


def train_model_multi_sent(args, optimizer_grouped_parameters, num_train_optimization_steps,
                           train_examples, label_list, tokenizer, output_mode, model, device,
                           num_labels, n_gpu, global_step, processor, task_name, k_th):
    if args.do_train:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        # ---------------------------------
        if use_k_zhe:
            train_examples, eval_on_train_examples = train_split_k_zhe(train_examples, k_th, k_zhe)
        # -------------------------------
        train_features, index_cls = examples2features_sents(
            train_examples, label_list, args.max_seq_length, tokenizer, output_mode)  # Modify
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_cls_pos = torch.tensor(index_cls, dtype=torch.long)  # Modify
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_cls_pos)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        # print("all cla pos",all_cls_pos.size())
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            num_currect = 0.
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, cls_pos = batch  # 理论上来讲，cls_pos就是一个数

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, cls_pos, segment_ids, input_mask, labels=None)  # modify

                # 直接默认classify
                # 这里开始魔改
                # logit: (batch_size, num_classes)
                sm_func = Softmax(dim=1)
                prob = sm_func(logits)

                mask = torch.le(prob, threshold_multi_sent).float().to(device)
                loss_func = NLLLoss()

                loss = loss_func(torch.log(prob).view(-1, num_labels) * mask, label_ids.view(-1))
                # 计算正确个数
                preds = np.argmax(prob.cpu().detach().numpy(), axis=1)
                label = label_ids.cpu().detach().numpy()
                for i in range(len(preds)):
                    if preds[i] == label[i]:
                        num_currect += 1

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            logger.info("")
            logger.info("***** train results *****")
            logger.info("\tloss = %s", str(tr_loss / nb_tr_steps))
            logger.info("\taccuracy = %s", str(num_currect / nb_tr_examples))
            # """训练时查看在验证集上的准确率"""
            # eval_examples = processor.get_dev_examples(args.data_dir)  # 读数据
            # eval_model_multi_sent(eval_examples, processor, args, label_list, tokenizer, output_mode,
            #                       model, device, num_labels, task_name, tr_loss,
            #                       nb_tr_steps, global_step, False)

        for ele in eval_result:
            print(ele)

        # ---------------------------------
        if use_k_zhe:
            # 在预留数据上得到预测值
            k_th_train_pred = eval_model_multi_sent(eval_on_train_examples, processor, args, label_list, tokenizer,
                                                    output_mode,
                                                    model, device, num_labels, task_name, tr_loss,
                                                    nb_tr_steps, global_step, False)
            global train_k_zhe_pred, all_eval_pred
            train_k_zhe_pred = train_k_zhe_pred + k_th_train_pred
            # 得到测试集上的预测值
            eval_examples = processor.get_dev_examples(args.data_dir)  # 读dev数据
            # eval_examples = processor.get_dev_examples(args.data_dir)  # 读test数据 modify
            eval_pred = eval_model_multi_sent(eval_examples, processor, args, label_list, tokenizer, output_mode,
                                              model, device, num_labels, task_name, tr_loss,
                                              nb_tr_steps, global_step, False)
            all_eval_pred.append(eval_pred)
        # ------------------------------------
        # # 我选择不存.jpg
        # # Save a trained model, configuration and tokenizer
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        #
        # # If we save using the predefined names, we can load using `from_pretrained`
        # output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        # output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        #
        # torch.save(model_to_save.state_dict(), output_model_file)
        # model_to_save.config.to_json_file(output_config_file)
        # tokenizer.save_vocabulary(args.output_dir)

    if args.do_eval:
        tr_loss = 0
        nb_tr_steps = 0
        model = bert_context_classification.from_pretrained(args.output_dir, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(device)
        eval_examples = processor.get_dev_examples(args.data_dir)  # 读数据
        eval_model_multi_sent(eval_examples, processor, args, label_list, tokenizer, output_mode,
                              model, device, num_labels, task_name, tr_loss,
                              nb_tr_steps, global_step, True)


def eval_model_multi_sent(eval_examples, processor, args, label_list, tokenizer, output_mode,
                          model, device, num_labels, task_name, tr_loss,
                          nb_tr_steps, global_step, out_prob):
    """return List of List of float"""
    # eval_examples = processor.get_train_examples(args.data_dir)

    eval_features, index_cls = examples2features_sents(
        eval_examples, label_list, args.max_seq_length, tokenizer, output_mode)  # modify
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_cls_pos = torch.tensor(index_cls, dtype=torch.long)  # Modify
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_cls_pos)  # modify
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    prob_res = []

    for input_ids, input_mask, segment_ids, label_ids, cls_pos in tqdm(eval_dataloader, desc="Evaluating"):  # modify
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        cls_pos = cls_pos.to(device)  # modify

        with torch.no_grad():
            logits = model(input_ids, cls_pos, segment_ids, input_mask, labels=None)  # modify

        # create eval loss and other metric required by the task
        if output_mode == "classification":
            loss_fct = CrossEntropyLoss()
            tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        elif output_mode == "regression":
            loss_fct = MSELoss()
            tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
        # -----------------------------------
        sm_func = Softmax(dim=1)
        prob = sm_func(logits)  # [batch_size,num_class]
        prob_res = prob_res + list(prob.detach().cpu().numpy())
    prob_list = list(map(lambda x: list(x), prob_res))
    if out_prob:
        print(type(prob_list))
        print(type(prob_list[0]))
        print(prob_list[:3])
        import pickle
        pickle.dump(prob_list, open(os.path.join(args.output_dir, "oof_test.pkl"), "wb"))
    # ---------------------------------
    eval_loss = eval_loss / nb_eval_steps
    preds = preds[0]
    if output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_mode == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(task_name, preds, all_label_ids.numpy())
    loss = tr_loss / nb_tr_steps if args.do_train else None

    # 展示在验证集上的结果
    result['eval_loss'] = eval_loss
    result['global_step'] = global_step
    result['loss'] = loss

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    eval_result.append(result)
    return prob_list


if __name__ == "__main__":
    simple_main()
