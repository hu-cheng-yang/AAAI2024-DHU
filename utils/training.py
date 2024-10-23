# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
from argparse import Namespace
from typing import Tuple

import torch
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset
from models.utils.continual_model import ContinualModel

from utils.loggers import *
from utils.status import ProgressBar

import wandb

from utils.FASstatics import get_TPRatFPR_states, countMetric, calculate
from utils.FASstatics_ori import my_metrics


def test_evaluate(model: ContinualModel, dataset: ContinualDataset, level="video", best=0):
    if model.NAME == "mas":
        model.net.tmodel.eval()
    else:
        model.net.eval()
    AUC = []
    HTER = []
    TPR = []
    APCER = []
    BPCER = []
    ACER = []
    FPR_list = []
    TPR_list = []
    model.net.eval()
    progress_bar = ProgressBar(verbose=True)
    dataloaders = dataset.test_loaders
    for k, test_loader in enumerate(dataloaders):
        scoremap = {}
        labelmap = {}
        scorelist = []
        labellist = []
        totaltest = len(test_loader)
        for index_test, data in enumerate(test_loader):
            with torch.no_grad():
                inputs, labels, _, path = data
                inputs, labels = inputs.to(model.device), labels.to(model.device)

                if model.args.backbone == "maddg-inter":
                    outputs = model.net(inputs, inference=True)
                elif model.NAME == "mas":
                    outputs = model.net.tmodel(inputs)
                else:
                    outputs = model(inputs)

                outputs = torch.softmax(outputs, dim=1).cpu().detach().numpy()[:, 1]
                labels = labels.cpu().data.numpy()

                for index, p in enumerate(path):
                    if level == "video":
                        id = p[:-9]
                    else:
                        id = p
                    if id in scoremap.keys():
                        if best == 1:
                            if labels[index] == 0:
                                scoremap[id][0] = outputs[index] if outputs[index] < scoremap[id][0] else scoremap[id][0]
                            else:
                                scoremap[id][0] = outputs[index] if outputs[index] > scoremap[id][0] else scoremap[id][0]
                        else:
                            scoremap[id].append(outputs[index])
                            labelmap[id].append(labels[index])
                    else:
                        scoremap[id] = []
                        labelmap[id] = []
                        scoremap[id].append(outputs[index])
                        labelmap[id].append(labels[index])
            progress_bar.progtest(index_test, totaltest, 0, k)

        for key in scoremap.keys():
            scorelist.append(sum(scoremap[key]) / len(scoremap[key]))
            labellist.append(sum(labelmap[key]) / len(labelmap[key]))

        print("scorelist: ", len(scorelist))
        
        scorelist = np.array(scorelist)
        labellist = np.array(labellist)

        hter, auc = countMetric(scorelist, labellist)
        apcer, bpcer, acer, _, _, _, [tpr, _, _], fpr_l, tpr_l = my_metrics(labellist, scorelist, val_phase=True)

        HTER.append(hter)
        AUC.append(auc)
        TPR.append(tpr)
        APCER.append(apcer)
        BPCER.append(bpcer)
        ACER.append(acer)
        FPR_list.append(fpr_l)
        TPR_list.append(tpr_l)

        if model.NAME == "mas":
            model.net.tmodel.train()
        else:
            model.net.train()

    return HTER, AUC, TPR, APCER, BPCER, ACER, FPR_list, TPR_list



def savemodel(model, dataset, epoch, task, last=False):
    savelist = model.save_model
    if last:
        for m in savelist:
            savepath = os.path.join(wandb.run.dir, "ckpts", "{}_{}_{}_task{}_last.pt".format(model.NAME, dataset.NAME, m, task + 1))
            torch.save(getattr(model, m).state_dict(), savepath)
            print("save model to {}".format(savepath))
        return
    for m in savelist:
        savepath = os.path.join(wandb.run.dir, "ckpts", "{}_{}_{}_task{}_iteration{}.pt".format(model.NAME, dataset.NAME, m, task + 1, (epoch)))
        torch.save(getattr(model, m).state_dict(), savepath)
        print("save model to {}".format(savepath))


def FAStrain(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    print(args)

    flagFirstTrain = True

    if not args.nowand:
        assert wandb is not None, "Wandb not installed, please install it or run without wandb"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args), name=args.wandb_name)

    if not os.path.exists(os.path.join(wandb.run.dir, "ckpts")):
        os.makedirs(os.path.join(wandb.run.dir, "ckpts"))

    if not os.path.exists(os.path.join(wandb.run.dir, "samples")):
        os.makedirs(os.path.join(wandb.run.dir, "samples"))


    print("Info | Project: {}|".format(model.args.name))
    print(model.args.description)

    model.net.to(model.device)
    # print(model.NAME)
    # if "phystd" in model.NAME:
    #     model.to_device()
    results, results_mask_classes = [], []

    if not args.disable_log:
        logger = Logger(dataset.SETTING, dataset.NAME, model.NAME)

    progress_bar = ProgressBar(verbose=not args.non_verbose)


    print(file=sys.stderr)
    if args.resume == 1:
        task_range = range(args.resume_task, dataset.N_TASKS)
    else:
        task_range = range(dataset.N_TASKS)
    for t in task_range:
        model.net.train()
        if args.resume == 1 and args.resume_iter < 0:
            if hasattr(model, 'end_task'):
                try:
                    model.current_task = args.resume_task - 1
                except:
                    print("Model do not contain current task.")
                for i in range(0, args.resume_task):
                    dataset.i = i
                    dataset.train_loader, _ = dataset.get_data_loaders()
                model.end_task(dataset)

        dataset.i = t
        train_loader, test_loader = dataset.get_data_loaders()

        from torch.optim import SGD
        model.opt = SGD(model.net.parameters(), lr=model.args.lr)
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)

        iteration = 0
        end = False

        try:
            dataset.RESUME = 0
        except:
            print("This dataset do not contain RESUME.")
        if args.resume == 1 and args.resume_iter >= 0 and flagFirstTrain:
            epoch_range = range(args.resume_iter, 10000000)
            flagFirstTrain = False
        else:
            epoch_range = range(10000000)
        flag_endtask = False
        if args.resume_iter == args.n_epochs:
            flag_endtask = True

        break_flag = False
        for epoch in epoch_range:

            if flag_endtask:
                break
            if hasattr(model, 'begin_epoch'):
                model.begin_epoch(dataset, epoch)
            if args.model == 'joint':
                continue
            for i, data in enumerate(train_loader):
                if dataset.NAME == "fas-domain":
                    inputs, labels, domain_labels, not_aug_inputs = data
                    inputs, labels, domain_labels = inputs.to(model.device), labels.to(
                        model.device), domain_labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    if "stm" in model.NAME:
                        labels = [labels, domain_labels]

                elif dataset.NAME == "seq-fas-task":
                    inputs, labels, not_aug_inputs, path = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)

                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(
                        model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)

                loss = model.meta_observe(inputs, labels, not_aug_inputs)
                # assert not math.isnan(loss)
                progress_bar.prog(iteration, len(train_loader), epoch, t, loss)
                iteration += 1
                wandb.log({"Training Loss": loss})

                if iteration == args.n_epochs:
                    savemodel(model, dataset, epoch, t, last=True)
                    end = True
                    break

                if iteration % args.interval_test_epoch == 0:
                    savemodel(model, dataset, iteration, t)

                if iteration % args.interval_test_epoch == 0: #and args.backbone != "resnet-s" and args.backbone != "resnet-p":

                    HTER, AUC, TPR, APCER, BPCER, ACER, fpr, tpr = test_evaluate(model, dataset, level="video", best=0)
                    print("iteration:", iteration)
                    print("HTER:", HTER, sum(HTER) / len(HTER))
                    print("AUC:", AUC, sum(AUC)/len(AUC))
                    print("TPR:", TPR, sum(TPR)/len(TPR))
                    print("APCER:", APCER, sum(APCER)/len(APCER))
                    print("BPCER:", BPCER, sum(BPCER)/len(BPCER))
                    print("ACER:", ACER, sum(ACER)/len(ACER))
                    # savemodel(model, dataset, iteration, t)
                    log = {}
                    for index, hter in enumerate(HTER):
                        log["HTER_" + str(index)] = hter
                    log["HTER_avg"] = sum(HTER) / len(HTER)

                    for index, hter in enumerate(AUC):
                        log["AUC_" + str(index)] = hter
                    log["AUC_avg"] = sum(AUC) / len(AUC)

                    for index, hter in enumerate(TPR):
                        log["TPR_" + str(index)] = hter
                    log["TPR_avg"] = sum(TPR) / len(TPR)

                    for index, hter in enumerate(APCER):
                        log["APCER_" + str(index)] = hter
                    log["APCER_avg"] = sum(APCER) / len(APCER)

                    for index, hter in enumerate(BPCER):
                        log["BPCER_" + str(index)] = hter
                    log["BPCER_avg"] = sum(BPCER) / len(BPCER)

                    for index, hter in enumerate(ACER):
                        log["ACER_" + str(index)] = hter
                    log["ACER_avg"] = sum(ACER) / len(ACER)

                    wandb.log(log)

                    model.net.train()
            
            if end:
                break


        if hasattr(model, 'end_task'):
            model.end_task(dataset)

    print("Model Training Over, the wandb dir is {}".format(wandb.run.dir))
