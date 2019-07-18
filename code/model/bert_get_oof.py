from __future__ import absolute_import, division, print_function
import copy
import argparse
import logging
import os
import sys
import random
from tqdm import tqdm, trange
from sklearn.model_selection import StratifiedKFold
import numpy as np

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from custbert import cusBertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

from run_classifier_dataset_utils import processors, output_modes, convert_examples_to_features, compute_metrics,load_data
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger = logging.getLogger(__name__)


def eval(model,eval_dataloader,device,args,Num_examples):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    out_label_ids = None
    f = torch.nn.Softmax()
    logger.info("***** Running evaluating *****")
    logger.info("  Num examples = %d", Num_examples)
    logger.info("  Batch size = %d", args.eval_batch_size)
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

        # create eval loss and other metric required by the task
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        
        logits = f(logits)
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
            out_label_ids = label_ids.detach().cpu().numpy()
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, label_ids.detach().cpu().numpy(), axis=0)
    
    preds = preds[0]
    preds_y = np.argmax(preds, axis=1)
    result = compute_metrics(preds_y, out_label_ids)
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    return preds

def train(model,train_dataloader,args,device,n_gpu,Num_examples):
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", Num_examples)
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)

    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            #决定是否使用vat和tsa
            # # define a new function to compute loss values for both output_modes
            logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), label_ids.view(-1))
            # define a new function to compute loss values for both output_modes
            # logits = model(input_ids, segment_ids, input_mask, labels=None)
            # #logit: (batch_size, num_classes)
            # sm_func = torch.nn.Softmax(dim=1)
            # prob = sm_func(logits)
            # threshold = 0.5 + 0.5*np.exp( 3*(global_step/float(num_train_optimization_steps) - 1 ))
            # mask = torch.le(prob, threshold).float().to(device)
            # mask.requires_grad = False
            # loss_func = torch.nn.NLLLoss(weight=torch.tensor([1.,4.,1.]).to(device))
            # loss = loss_func(torch.log(prob).view(-1, 3) * mask, label_ids.view(-1))

            # wordmask = (torch.rand(input_mask.shape[0],args.max_seq_length)>0.2).to(device).long()
            # wordmask *= input_mask
            # logits2 = model(input_ids, segment_ids, wordmask , labels=None)
            # prob2 = sm_func(logits2)
            # klf = torch.nn.KLDivLoss()
            # loss2 = klf(prob,prob2)
            # loss += loss2
            
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
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
    return model

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default="/data1/lyh/data/smp/en_med",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default="bert-base-multilingual-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default="sentiment",
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default="/data1/lyh/data/smp/eval/oof_final/oof1",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="/data1/lyh/data/smp/lm-test",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=True,
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
                        default=4e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=2.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.3,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=123123,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)


    tokenizer = BertTokenizer.from_pretrained(args.cache_dir, do_lower_case=args.do_lower_case)
    model = BertForSequenceClassification.from_pretrained(args.cache_dir, num_labels=num_labels)
    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(device)
    model = torch.nn.DataParallel(model)
    #获取数据
    all_input_ids, all_input_mask, all_segment_ids, all_label_ids = \
        load_data(processor,args,label_list,tokenizer,output_mode,logger,mode=True)
    test_all_input_ids, test_all_input_mask, test_all_segment_ids, test_all_label_ids = \
        load_data(processor,args,label_list,tokenizer,output_mode,logger,mode=False)
    
    test_data = TensorDataset(test_all_input_ids, test_all_input_mask, test_all_segment_ids, test_all_label_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)
    #k折验证
    stratified_folder = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=False)

    oof_train = np.zeros((all_label_ids.shape[0],3))
    oof_test = np.zeros((test_all_label_ids.shape[0],3))

    for train_index, test_index in stratified_folder.split(all_input_ids.numpy(),all_label_ids.numpy()):

        kfall_input_ids, kfall_input_mask, kfall_segment_ids, kfall_label_ids = \
            all_input_ids[train_index], all_input_mask[train_index], all_segment_ids[train_index], all_label_ids[train_index]
        train_data = TensorDataset(kfall_input_ids, kfall_input_mask, kfall_segment_ids, kfall_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        kf_dev_all_input_ids, kf_dev_all_input_mask, kf_dev_all_segment_ids, kf_dev_all_label_ids = \
            all_input_ids[test_index], all_input_mask[test_index], all_segment_ids[test_index], all_label_ids[test_index]
        dev_data = TensorDataset(kf_dev_all_input_ids, kf_dev_all_input_mask, kf_dev_all_segment_ids, kf_dev_all_label_ids)
        dev_sampler = SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.eval_batch_size)

        new_model = copy.deepcopy(model)
        kf_model = train(new_model,train_dataloader,args,device,n_gpu,kfall_label_ids.shape[0])
        kftrain = eval(kf_model,dev_dataloader,device,args,kf_dev_all_label_ids.shape[0])
        kftest = eval(kf_model,test_dataloader,device,args,test_all_label_ids.shape[0])

        oof_train[test_index] = kftrain
        oof_test += kftest/5

    with open(args.output_dir+'/oof_train', "wb") as writer:
        pickle.dump(oof_train.tolist(), writer)
    with open(args.output_dir+'/oof_test', "wb") as writer:
        pickle.dump(oof_test.tolist(), writer)

if __name__ == "__main__":
    main()
