# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable

import copy
import numpy as np
from collections import defaultdict, OrderedDict

try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)

ALGORITHMS = [
    'ERM',
    # 'Fish',
    # 'IRM',
    # 'GroupDRO',
    # 'Mixup',
    # 'MLDG',
    # 'CORAL',
    # 'MMD',
    # 'DANN',
    # 'CDANN',
    # 'MTL',
    # 'SagNet',
    # 'ARM',
    # 'VREx',
    # 'RSC',
    # 'SD',
    # 'ANDMask',
    # 'SANDMask',
    # 'IGA',
    # 'SelfReg',
    # "Fishr",
    # 'TRM',
    # 'IB_ERM',
    # 'IB_IRM',
    # 'CAD',
    # 'CondCAD',
    'PGE',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)


class Queue(nn.Module):
    def __init__(self):
        super(Queue, self).__init__()

    def initialize_queue(self, size=[0, 0, 0, 0], float=False, ld=False):
        if float == True:
            self.queue = torch.zeros((0, size[1]), dtype=torch.float32)
        elif ld == True:
            self.queue = torch.zeros((0, size[1], size[2], size[3]), dtype=torch.float32)
        else:
            self.queue = torch.zeros((0), dtype=torch.long)

    def dequeue(self, n_deq):
        self.queue = self.queue[:self.queue.size(0) - n_deq]

    def enqueue(self, x):
        self.queue = torch.cat((x, self.queue), dim=0)


class PGE(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(PGE, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)

        self.num_classes = num_classes

        self.queue_flag = int(hparams['queue_flag'])
        self.N_p = hparams['N_p']

        self.w_ic = hparams['w_ic']
        self.gam2 = hparams['gam2']
        self.gam1 = hparams['gam1']

        self.featurizer = networks.Featurizer(input_shape, self.hparams)

        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

        self.que_label = []  # [l, dim]
        self.que_domain = []  # corresponding domain [l, 1]
        for qlen in range(self.num_classes):
            que_l = Queue()
            que_l.initialize_queue([0, self.featurizer.n_outputs], float=True)
            que_d = Queue()
            que_d.initialize_queue()
            self.que_label.append(que_l)
            self.que_domain.append(que_d)

        self.pivot_set = Queue()
        self.pivot_set.initialize_queue([0, self.num_classes, 3, self.featurizer.n_outputs], ld=True)  # [length, l, d, dim]

        # clone
        self.que_label_clone = []  # [l, dim]
        for qlen in range(self.num_classes):
            que_l = Queue()
            que_l.initialize_queue([0, self.featurizer.n_outputs], float=True)
            self.que_label_clone.append(que_l)

        self.pivot_set_clone = Queue()
        self.pivot_set_clone.initialize_queue([0, self.num_classes, 3, self.featurizer.n_outputs], ld=True)  # [length, l, d, dim]

    def update(self, minibatches, unlabeled=None, ii=None):
        d_count = 0
        all_d = torch.empty((0),dtype=torch.long) # [batch_size * domain_num]
        for x,y,x2 in minibatches:
            all_d = torch.cat((all_d,torch.ones((x.size(0)),dtype=torch.long)*d_count),dim=0)
            d_count += 1

        all_x = torch.cat([x for x,y,x2 in minibatches]) # [batch_size * domain_num, 3, 224, 224]
        all_y = torch.cat([y for x,y,x2 in minibatches]) #  [batch_size * domain_num]
        all_x2 = torch.cat([x2 for x,y,x2 in minibatches])
        data_size = all_x.size(0)

        outputA_embedding_original = self.featurizer(all_x)
        outputA_embedding_clone = self.featurizer(all_x2)
        outputA = torch.cat((outputA_embedding_original, outputA_embedding_clone),dim=0)
        label_all = torch.cat((all_y,all_y),dim=0)

        loss_rep = 0
        loss_att = 0

        if not (ii < self.queue_flag + self.N_p):
            #------------------------------------ repulsion ------------------------------------#
            #------------------------------------ original ------------------------------------#
            label_sample = self.pivot_set.queue.cuda()[:, :, all_d, :]  # n, l, bs, dim
            label_sample = torch.transpose(label_sample, 0, 1)  # l, n, bs, dim
            label_sample = torch.transpose(label_sample, 1, 2)  # l, bs, n, dim
            label_sample = torch.transpose(label_sample, 0, 1)  # bs, l, n, dim

            mask_label = torch.ones_like(label_sample).scatter_ \
                (1, all_y[:, None, None, None].repeat(1, 1, label_sample.size(2), label_sample.size(3)), 0.)
            label_sample = label_sample[mask_label.bool()].view \
                (label_sample.size(0), label_sample.size(1) - 1, label_sample.size(2), label_sample.size(3))
            # bs, l-1, n, 512

            sim_label = torch.zeros((0, label_sample.size(1), label_sample.size(2))).cuda()  # 128, l-1, n
            for one, two in zip(outputA_embedding_original, label_sample):  # dim <-> l-1,n,dim
                sim_label = torch.cat((sim_label, torch.cdist(one.view(1, -1), two.reshape(-1, self.featurizer.n_outputs), p=2)
                     .view(1, label_sample.size(1), label_sample.size(2))), dim=0)  # 1,d,n

            sim_label = torch.min(sim_label, dim=2).values  # bs,l-1
            loss_rep -= torch.sum(sim_label)

            #------------------------------------ clone ------------------------------------#
            label_sample_clone = self.pivot_set_clone.queue.cuda()[:, :, all_d,:]
            label_sample_clone = torch.transpose(label_sample_clone, 0, 1)
            label_sample_clone = torch.transpose(label_sample_clone, 1, 2)
            label_sample_clone = torch.transpose(label_sample_clone, 0, 1)

            mask_label_clone = torch.ones_like(label_sample_clone).scatter_ \
                (1, all_y[:, None, None, None].repeat(1, 1, label_sample_clone.size(2), label_sample_clone.size(3)), 0.)
            label_sample_clone = label_sample_clone[mask_label_clone.bool()].view \
                (label_sample_clone.size(0), label_sample_clone.size(1) - 1, label_sample_clone.size(2),
                 label_sample_clone.size(3))

            sim_label_clone = torch.zeros(
                (0, label_sample_clone.size(1), label_sample_clone.size(2))).cuda()
            for one, two in zip(outputA_embedding_clone, label_sample_clone):
                sim_label_clone = torch.cat((sim_label_clone, torch.cdist(one.view(1, -1), two.reshape(-1, self.featurizer.n_outputs), p=2)
                    .view(1, label_sample_clone.size(1), label_sample_clone.size(2))), dim=0)

            sim_label_clone = torch.min(sim_label_clone, dim=2).values
            loss_rep -= torch.sum(sim_label_clone)
            loss_rep = loss_rep / sim_label_clone.size(0) / sim_label_clone.size(1) / 2

            #------------------------------------ attraction ------------------------------------#
            #------------------------------------ original ------------------------------------#
            label_sample_same = self.pivot_set.queue.cuda()[:, all_y, :, :]  # n, bs, d, dim
            label_sample_same = torch.transpose(label_sample_same, 0, 1)  # bs, n, d, dim
            label_sample_same = torch.transpose(label_sample_same, 1, 2)  # bs, d, n, dim

            sim_label_same = torch.zeros((0, label_sample_same.size(1), label_sample_same.size(2))).cuda()  # bs, d, n
            for one, two in zip(outputA_embedding_original, label_sample_same):  # dim <-> d,n,dim
                sim_label_same = torch.cat((sim_label_same, torch.cdist(one.view(1, -1), two.reshape(-1, self.featurizer.n_outputs), p=2)
                     .view(1, label_sample_same.size(1), label_sample_same.size(2))), dim=0)  # 1,d,n

            sim_label_same = torch.max(sim_label_same, dim=2).values
            loss_att += torch.sum(sim_label_same)

            # ------------------------------------ clone ------------------------------------#
            label_sample_same_clone = self.pivot_set_clone.queue.cuda()[:, all_y, :, :]
            label_sample_same_clone = torch.transpose(label_sample_same_clone, 0, 1)
            label_sample_same_clone = torch.transpose(label_sample_same_clone, 1, 2)

            sim_label_same_clone = torch.zeros((0, label_sample_same_clone.size(1), label_sample_same_clone.size(2))).cuda()
            for one, two in zip(outputA_embedding_clone, label_sample_same_clone):
                sim_label_same_clone = torch.cat((sim_label_same_clone, torch.cdist(one.view(1, -1), two.reshape(-1, self.featurizer.n_outputs), p=2)
                     .view(1, label_sample_same_clone.size(1), label_sample_same_clone.size(2))), dim=0)

            sim_label_same_clone = torch.max(sim_label_same_clone, dim=2).values
            loss_att += torch.sum(sim_label_same_clone)
            loss_att = loss_att / sim_label_same_clone.size(0) / sim_label_same_clone.size(1) / 2
            #####################################################################################

            self.pivot_set.dequeue(1)
            self.pivot_set_clone.dequeue(1)

        for seq, lb in enumerate(all_y):
            self.que_label[lb].enqueue(outputA_embedding_original[seq].view(1, self.featurizer.n_outputs).data.cpu())
            self.que_domain[lb].enqueue(all_d[seq].view(1, -1).data.cpu())
            self.que_label_clone[lb].enqueue(outputA_embedding_clone[seq].view(1, self.featurizer.n_outputs).data.cpu())

        if not (ii < self.queue_flag):
            pres0_topk = torch.zeros((0, self.featurizer.n_outputs), dtype=torch.float32).cuda()
            pres1_topk = torch.zeros((0, self.featurizer.n_outputs), dtype=torch.float32).cuda()
            pres2_topk = torch.zeros((0, self.featurizer.n_outputs), dtype=torch.float32).cuda()
            pres0_topk_clone = torch.zeros((0, self.featurizer.n_outputs), dtype=torch.float32).cuda()
            pres1_topk_clone = torch.zeros((0, self.featurizer.n_outputs), dtype=torch.float32).cuda()
            pres2_topk_clone = torch.zeros((0, self.featurizer.n_outputs), dtype=torch.float32).cuda()
            for ql_seq, qd_seq, ql_clone_seq in zip(self.que_label, self.que_domain, self.que_label_clone):
                outdom0 = ql_seq.queue[torch.where(qd_seq.queue == 0)[0]].cuda()
                outdom1 = ql_seq.queue[torch.where(qd_seq.queue == 1)[0]].cuda()
                outdom2 = ql_seq.queue[torch.where(qd_seq.queue == 2)[0]].cuda()
                outdom0_clone = ql_clone_seq.queue[torch.where(qd_seq.queue == 0)[0]].cuda()
                outdom1_clone = ql_clone_seq.queue[torch.where(qd_seq.queue == 1)[0]].cuda()
                outdom2_clone = ql_clone_seq.queue[torch.where(qd_seq.queue == 2)[0]].cuda()

                pres0_topk = torch.cat((pres0_topk, outdom0[torch.topk(
                    torch.sum(torch.cdist(outdom0, outdom0, p=2), dim=1), 1, largest=False).indices]), dim=0)
                pres1_topk = torch.cat((pres1_topk, outdom1[torch.topk(
                    torch.sum(torch.cdist(outdom1, outdom1, p=2), dim=1), 1, largest=False).indices]), dim=0)
                pres2_topk = torch.cat((pres2_topk, outdom2[torch.topk(
                    torch.sum(torch.cdist(outdom2, outdom2, p=2), dim=1), 1, largest=False).indices]), dim=0)
                pres0_topk_clone = torch.cat((pres0_topk_clone, outdom0_clone[torch.topk(
                    torch.sum(torch.cdist(outdom0_clone, outdom0_clone, p=2), dim=1), 1, largest=False).indices]), dim=0)
                pres1_topk_clone = torch.cat((pres1_topk_clone, outdom1_clone[torch.topk(
                    torch.sum(torch.cdist(outdom1_clone, outdom1_clone, p=2), dim=1), 1, largest=False).indices]), dim=0)
                pres2_topk_clone = torch.cat((pres2_topk_clone, outdom2_clone[torch.topk(
                    torch.sum(torch.cdist(outdom2_clone, outdom2_clone, p=2), dim=1), 1, largest=False).indices]), dim=0)

            pres0_topk = pres0_topk.view(pres0_topk.size(0), 1, pres0_topk.size(1))
            pres1_topk = pres1_topk.view(pres1_topk.size(0), 1, pres1_topk.size(1))
            pres2_topk = pres2_topk.view(pres2_topk.size(0), 1, pres2_topk.size(1))
            grid_put = torch.cat((pres0_topk, pres1_topk, pres2_topk), dim=1)
            self.pivot_set.enqueue(grid_put.view(1, grid_put.size(0), grid_put.size(1), grid_put.size(2)).data.cpu())

            pres0_topk_clone = pres0_topk_clone.view(pres0_topk_clone.size(0), 1, pres0_topk_clone.size(1))
            pres1_topk_clone = pres1_topk_clone.view(pres1_topk_clone.size(0), 1, pres1_topk_clone.size(1))
            pres2_topk_clone = pres2_topk_clone.view(pres2_topk_clone.size(0), 1, pres2_topk_clone.size(1))
            grid_put_clone = torch.cat((pres0_topk_clone, pres1_topk_clone, pres2_topk_clone), dim=1)
            self.pivot_set_clone.enqueue(grid_put_clone.view(1, grid_put_clone.size(0), grid_put_clone.size(1), grid_put_clone.size(2)).data.cpu())

            for lb in all_y:
                self.que_label[lb].dequeue(1)
                self.que_domain[lb].dequeue(1)
                self.que_label_clone[lb].dequeue(1)

        pd = torch.nn.PairwiseDistance(p=2)
        pd_dis = pd(outputA_embedding_original, outputA_embedding_clone)
        loss_ic = torch.sum(pd_dis) / data_size / 2

        outputA = self.classifier(outputA)
        loss_CE = F.cross_entropy(outputA, label_all)

        loss = loss_CE + loss_ic * self.w_ic
        if not (ii < self.queue_flag + self.N_p):
            loss = loss + (loss_rep * self.gam2) + (loss_att * self.gam1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

