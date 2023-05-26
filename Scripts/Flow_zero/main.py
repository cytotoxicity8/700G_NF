import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc, average_precision_score, classification_report, roc_auc_score

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ignite.utils import setup_logger

from utils import CustomCosineAnnealingWarmUpRestarts
import constant as const

import os
import itertools
import random

def save_model(model, optimizer, scheduler, epoch, checkpoint_dir):
    torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            },
            os.path.join(checkpoint_dir, "%d.pt" % epoch),
    )


def build_scheduler(optimizer):
    return CustomCosineAnnealingWarmUpRestarts(optimizer, T_0=const.SCHEDULER_T_0, eta_max = const.LR, T_up=0, gamma=1, T_mult=1)

def build_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr = 0, weight_decay=const.WEIGHT_DECAY), torch.optim.SGD(model.base.parameters(),lr=0)

def train(model, x_train, x_test, y_test, c_train, c_test, multiple=False):
    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    tb_logger = SummaryWriter(log_dir=checkpoint_dir)
    eval_logger = setup_logger(name='eval', filepath=os.path.join(checkpoint_dir, 'eval.txt'))

    optimizer, optimizer_altub = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    for epoch in range(const.MAX_EPOCH):
        train_one_epoch(model, x_train, c_train, epoch, optimizer, optimizer_altub, scheduler, tb_logger)
        if (epoch+1) % const.EVAL_EPOCH == 0:
            test_one_epoch(model, x_test, y_test, c_test, epoch, tb_logger, eval_logger, multiple)
        if (epoch+1) % const.SAVE_EPOCH == 0:
            save_model(model, optimizer, scheduler, epoch, checkpoint_dir)

def train_one_epoch(model, x_train, c_train, epoch, optimizer, optimizer_altub, scheduler, tb_logger):
    model.train()
    ret = model(x_train, c_train)
    loss = ret["loss"]
    # backward
    model.zero_grad()
    loss.backward()
    if (epoch+1)%const.ALTUB_EPOCH == 0 and (epoch+1)>const.SCHEDULER_T_0:
        lr = scheduler.get_lr()[0]
        for param_group in optimizer_altub.param_groups:
            torch.nn.utils.clip_grad_value_(model.base.base_mean, 100)
            torch.nn.utils.clip_grad_value_(model.base.base_cov, 100)
            param_group['lr'] = lr * const.ALTUB_LR
        optimizer_altub.step()
    else:
        optimizer.step()
    scheduler.step()

    tb_logger.add_scalar('loss', torch.mean(loss), epoch)

def test_one_epoch(model, x_test, y_test, c_test, epoch, tb_logger, eval_logger, multiple=False):
    model.eval()
    with torch.no_grad():
        pred = model(x_test, c_test)
    score = np.array(pred['score'].detach().cpu())

    auroc, auprc = evaluate_with_ratio(y_test, score, multiple)

    eval_logger.info(f'[epoch {epoch}] [AUROC {auroc}] [AUPRC {auprc}]')
    tb_logger.add_scalar('AUROC', auroc, epoch)
    tb_logger.add_scalar('AUPRC', auprc, epoch)


def evaluate_with_ratio(y_test, score, multiple = False):
    score = pd.DataFrame(score, index=y_test.index)
    positive_index = y_test[y_test['diseases']==1].index
    drop = int(round(len(positive_index) * 0.8))

    if multiple is False:
        drop_index = random.sample(positive_index, drop)
        score_eval = score.drop(drop_index)
        y_test_eval = y_test.drop(drop_index)
        return roc_auc_score(y_test_eval, score_eval), average_precision_score(y_test_eval, score_eval)

    
    auroc_list = []
    auprc_list = []

    for drop_index in itertools.combinations(positive_index, drop):
        drop_index = list(drop_index)
        score_eval = score.drop(drop_index)
        y_test_eval = y_test.drop(drop_index)
        auroc_list.append(roc_auc_score(y_test_eval, score_eval))
        auprc_list.append(average_precision_score(y_test_eval, score_eval))

    return sum(auroc_list)/len(auroc_list), sum(auprc_list)/len(auprc_list)