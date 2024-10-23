from models.utils.continual_model import ContinualModel
from backbone.diBuffer import diBufferSelfAttRealFake as diBuffer
from torch.optim import SGD
from backbone.adain import calc_mean_std
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from backbone.adain import calc_mean_std, adaptive_instance_normalization_meanstd as adaIN
from backbone.JSLoss import js_div
import faiss
import numpy as np
from utils.buffer import Buffer
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.draw import draw_tsne
import os
import wandb


def run_kmeans(x, nmb_clusters, verbose=True):
    n_data, d = x.shape

    clus = faiss.Clustering(d, nmb_clusters)

    clus.seed = np.random.randint(1234)

    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    clus.train(x, index)
    distance, I = index.search(x, 1)
    centroids = faiss.vector_float_to_array(clus.centroids)

    return [int(n[0]) for n in I], centroids.reshape(nmb_clusters, d)


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        x = torch.mean(F.softmax(x, dim=1), dim=0)
        b = x * torch.log(x)
        b = b.mean()
        return b
        

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class DHU(ContinualModel):
    NAME = 'dhu'
    COMPATIBILITY = ['domain-il']

    def __init__(self, backbone, loss, args, transform):
        super(DHU, self).__init__(backbone, loss, args, transform)
        self.current_task = 0
        self.diBuffer = diBuffer(512, args.buffer_size)
        print("Buffer size: ", args.buffer_size)
        self.save_model = ["net", "diBuffer"]
        self.Hloss = HLoss()
        self.cosLoss = torch.nn.CrossEntropyLoss()
        self.InfoLoss = torch.nn.CrossEntropyLoss()
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.draw_di = None
        self.draw_label = None
        self.step = 0
        self.dis = None
        self.reference_diBuffer = diBuffer(512, self.args.buffer_size)

    
    def resume(self, paths):
        from utils.FASutils import reload
        reload(self.net, paths[0])
        reload(self.diBuffer, paths[1])


    def observe(self, inputs, labels, not_aug_inputs):
        self.net.train()
        if self.buffer.is_empty():
            self.diBuffer.train()
        else:
            self.diBuffer.eval()

        if self.current_task  == 0:
            self.opt.zero_grad()
            outputs, features = self.net(inputs, returnt='all')
            loss_cls = self.loss(outputs, labels)

            di = self.cal_feat_meanstd(features)[:, :512]

            selected_di, choice, score, _ = self.diBuffer(di, labels)
            select_di = torch.argmax(choice, dim=1)


            loss_dinfo = self.cosLoss(score / 0.7, select_di)
            loss_entropy = self.Hloss(score)

            loss = loss_cls + loss_dinfo * 0.5 + loss_entropy

            loss.backward()

            self.opt.step()

            self.step += 1


        elif self.current_task > 0 and self.current_task < 3:

            self.opt.zero_grad()
            outputs, features = self.net(inputs, returnt='all')
            loss_cls = self.loss(outputs, labels)

            di_tgt = self.cal_feat_meanstd(features)[:, :512]

            selected_di, choice, score, self_att = self.diBuffer(di_tgt, labels)
            select_di = torch.argmax(choice, dim=1)

            loss_dinfo = self.cosLoss(score / 0.7, select_di)
            loss_entropy = self.Hloss(score)


            di_src, choice, score, _ = self.reference_diBuffer(di_tgt, labels)
            di_src, choice, score = di_src.detach(), choice.detach(), score.detach()

            di_src = F.normalize(di_src, dim=1)

            bts = inputs.shape[0]
            mean_0 = di_src[:, 0 : 64].view(bts, 64, 1, 1)
            std_0 = di_src[:, 64 : 128].view(bts, 64, 1, 1)


            feature0 = features["feature0"]
            syn_feature0 = adaIN(content_feat=feature0, di_mean=mean_0, di_std=std_0)
            syn_outputs, syn_features = self.net(syn_feature0, returnt='all', input=1)
            syn_di = self.cal_feat_meanstd(feat=syn_features)
            syn_di = syn_di[:, 128:512]

            loss_cls_syn = self.loss(syn_outputs, labels)

            loss_smp = 1 - torch.mean(F.cosine_similarity(syn_di, di_src[:, 128: 512], dim=1))

            feature3 = features["feature3"]
            feature4 = features["feature4"]

            syn_feature3 = syn_features["feature3"]
            syn_feature4 = syn_features["feature4"]

            logits_3, labels_info_3 = self.info_nce_loss(torch.cat([feature3[labels == 0], syn_feature3[labels == 0]]))
            loss_hal_3 = self.InfoLoss(logits_3, labels_info_3)

            logits_4, labels_info_4 = self.info_nce_loss(torch.cat([feature4[labels == 0], syn_feature4[labels == 0]]))
            loss_hal_4 = self.InfoLoss(logits_4, labels_info_4)

            loss_c = (loss_hal_3 + loss_hal_4) / inputs.shape[0] / 2

            loss_dist = js_div(syn_outputs[labels == 0], outputs[labels == 0])

            weight = [1, 1, 1, 1, 1]

            loss = loss_cls * weight[0] + loss_cls_syn * weight[1] + loss_smp * weight[2]  + loss_dist * weight[4] + loss_hal * weight[3] + loss_dinfo + loss_entropy

            loss.backward()
            self.opt.step()


        else:
            self.opt.zero_grad()
            outputs, features = self.net(inputs, returnt='all')
            loss_cls = self.loss(outputs, labels)

            di_tgt = self.cal_feat_meanstd(features)[:, :512]

            di_src, choice, score, _ = self.reference_diBuffer(di_tgt, labels)
            di_src, choice, score = di_src.detach(), choice.detach(), score.detach()

            di_src = F.normalize(di_src, dim=1)

            bts = inputs.shape[0]
            mean_0 = di_src[:, 0 : 64].view(bts, 64, 1, 1)
            std_0 = di_src[:, 64 : 128].view(bts, 64, 1, 1)


            feature0 = features["feature0"]
            syn_feature0 = adaIN(content_feat=feature0, di_mean=mean_0, di_std=std_0)
            syn_outputs, syn_features = self.net(syn_feature0, returnt='all', input=1)
            syn_di = self.cal_feat_meanstd(feat=syn_features)
            syn_di = syn_di[:, 128:512]

            loss_cls_syn = self.loss(syn_outputs, labels)

            loss_smp = 1 - torch.mean(F.cosine_similarity(syn_di, di_src[:, 128: 512], dim=1))

            feature3 = features["feature3"]
            feature4 = features["feature4"]

            syn_feature3 = syn_features["feature3"]
            syn_feature4 = syn_features["feature4"]

            logits_3, labels_info_3 = self.info_nce_loss(torch.cat([feature3[labels == 0], syn_feature3[labels == 0]]))
            loss_hal_3 = self.InfoLoss(logits_3, labels_info_3)

            logits_4, labels_info_4 = self.info_nce_loss(torch.cat([feature4[labels == 0], syn_feature4[labels == 0]]))
            loss_hal_4 = self.InfoLoss(logits_4, labels_info_4)

            loss_hal = (loss_hal_3 + loss_hal_4) / inputs.shape[0] / 2

            loss_dist = js_div(syn_outputs[labels == 0], outputs[labels == 0])

            weight = [1, 1, 1, 1, 1]

            loss = loss_cls * weight[0] + loss_cls_syn * weight[1] + loss_smp * weight[2]  + loss_dist * weight[4] + loss_hal * weight[3]

            loss.backward()
            self.opt.step()

        return loss.item()

            


    def begin_task(self, dataset):
        if self.current_task == dataset.N_TASKS - 1:
            print("Last task, no need to begin task.")
            self.opt = SGD(self.net.parameters(), lr=self.args.lr)
            return
        self.opt = SGD(list(self.net.parameters()) + list(self.diBuffer.parameters()), lr=self.args.lr)
        loader = dataset.train_loader
        all_di = []
        all_label = []
        i = 0
        self.diBuffer.eval()
        self.net.eval()
        with torch.no_grad():
            for x, y, _ in loader:
                i += 1
                if i % 100 == 0:
                    print("Begin Task: {} iters".format(i))
                if i == 1000:
                    break
                inputs = x
                _, features = self.net(inputs, returnt="all")
                di = self.cal_feat_meanstd(feat=features)
                all_di.append(di.detach())
                all_label.append(y.detach())
            all_di = F.normalize(torch.cat(all_di, dim=0)[:, :512], dim=1).detach().cpu().numpy()
            all_label = torch.cat(all_label, dim=0).detach().cpu().numpy()
        print("Start to run kmeans.")
        real_di = all_di[all_label == 0]
        print("Real di Shape", real_di.shape)
        fake_di = all_di[all_label == 1]
        print("Fake di Shape", fake_di.shape)
        _, center_real = run_kmeans(real_di, self.args.buffer_size // 2)
        _, center_fake = run_kmeans(fake_di, self.args.buffer_size // 2)
        center_real = torch.Tensor(center_real)
        center_fake = torch.Tensor(center_fake)
        center = torch.cat([center_real, center_fake], 0)
        print("Initial the buffer.")
        try:
            self.diBuffer.module.buffer.data.copy_(center)
        except:
            self.diBuffer.buffer.data.copy_(center)
        self.diBuffer.train()
        self.net.train()

    def end_task(self, dataset):
        if self.current_task == 3:
            print("Last task, no need to end task.")
            return
        self.fill_buffer_di(dataset=dataset)
        self.current_task += 1
        
        
        if self.current_task == 2:
            self.diBuffer.requires_grad_(False)
        self.net.module.conv1.requires_grad_(False)


    def fill_buffer_di(self, dataset):
        di = self.diBuffer.module.get_di()
        self.buffer.add_data(
            examples=di,)
        print("Add new buffer images | The buffer contain {} samples.".format(
            self.buffer.__len__()))

        if self.dis is None:
            self.dis = di
        else:
            di_real = torch.cat([self.dis[: self.dis.shape[0] // 2].cuda(), di[0:di.shape[0]//2].cuda()], 0).detach().cpu().numpy()
            di_fake = torch.cat([self.dis[self.dis.shape[0] // 2 :].cuda(), di[di.shape[0]// 2: ].cuda()], 0).detach().cpu().numpy()

            di_real_kmeans = run_kmeans(di_real, self.args.buffer_size // 2)[1]
            di_fake_kmeans = run_kmeans(di_fake, self.args.buffer_size // 2)[1]

            self.dis = torch.from_numpy(np.concatenate((di_real_kmeans, di_fake_kmeans), axis=0))  #torch.cat([di_real_kmeans, di_fake_kmeans], 0)

        self.reference_diBuffer.requires_grad_(False)
        self.reference_diBuffer.module.input_stype(self.dis)
        
    


    def cal_feat_meanstd(self, feat: dict):
        mean0, std0 = calc_mean_std(feat=feat['feature0'])
        mean1, std1 = calc_mean_std(feat=feat['feature1'])
        mean2, std2 = calc_mean_std(feat=feat['feature2'])
        mean3, std3 = calc_mean_std(feat=feat['feature3'])
        mean4, std4 = calc_mean_std(feat=feat['feature4'])
        mean0, std0, mean1, std1, mean2, std2, mean3, std3, mean4, std4 = mean0.view(mean0.shape[0], -1), std0.view(std0.shape[0], -1), \
                                                                          mean1.view(mean1.shape[0], -1), std1.view(std1.shape[0], -1), \
                                                                          mean2.view(mean2.shape[0], -1), std2.view(std2.shape[0], -1), \
                                                                          mean3.view(mean3.shape[0], -1), std3.view(std3.shape[0], -1), \
                                                                          mean4.view(mean4.shape[0], -1), std4.view(std4.shape[0], -1)

        return torch.cat((mean0, std0, mean1, std1, mean2, std2, mean3, std3, mean4, std4), dim=1)



    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(features.shape[0] // 2) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1))
        labels = labels.cuda()
        features = features.view(features.shape[0], -1)
        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / 0.7
        return logits, labels
