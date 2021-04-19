# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb, rank_roidb_ratio, filter_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
    adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

import pickle

imdb_list = ['KAIST_campus','KAIST_road','KAIST_downtown']
data_cache_path = 'data/cache'
imdb_classes =  ('__background__',  # always index 0
                          'person',
                          'people','cyclist'
                         )
parties = len(imdb_list)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101',
                        default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    parser.add_argument('--round', dest='round',
                    help='total rounds',
                    default=10, type=int)


    args = parser.parse_args()
    return args

def load_client_dataset(imdb_list):
    dataloader_list = []
    iter_epochs_list = []
    for imdb_name in imdb_list:
        pkl_file = os.path.join(data_cache_path, imdb_name + '_gt_roidb.pkl')

        with open(pkl_file, 'rb') as f:
            roidb = pickle.load(f)

        roidb = filter_roidb(roidb)

        ratio_list, ratio_index = rank_roidb_ratio(roidb)

        train_size = len(roidb)
        print(train_size)
        iters_per_epoch = int(train_size / args.batch_size)
        print('iters_per_epoch: ' + str(iters_per_epoch))
        iter_epochs_list.append(iters_per_epoch)
        sampler_batch = sampler(train_size, args.batch_size)

        dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, imdb_classes, training=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 sampler=sampler_batch, num_workers=args.num_workers)
        dataloader_list.append(dataloader)
    return dataloader_list, iter_epochs_list


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0, batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch * batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1, 1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover), 0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

    
def initial_network(args):
    
      # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(imdb_classes, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(imdb_classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(imdb_classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(imdb_classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()
    


    if args.cuda:
        fasterRCNN.cuda()

    if args.mGPUs:
        fasterRCNN = nn.DataParallel(fasterRCNN)
        

        
    return fasterRCNN

def getOptimizer(fasterRCNN,args):
    lr = args.lr
    params = []
    for key, value in dict(fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                    'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
                
    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)

    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM) 
    return optimizer


def train(args,dataloader,imdb_name,iters_per_epoch, fasterRCNN, optimizer, num_round):     
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)
    
    lr = args.lr

    
    
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter = iter(dataloader)
        for step in range(iters_per_epoch):
            data = next(data_iter)

            with torch.no_grad():
                im_data.resize_(data[0].size()).copy_(data[0])
                im_info.resize_(data[1].size()).copy_(data[1])
                gt_boxes.resize_(data[2].size()).copy_(data[2])
                num_boxes.resize_(data[3].size()).copy_(data[3])

            fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()
            #      print('loss={}'.format(loss))
            # backward
            optimizer.zero_grad()
            loss.backward()
            if args.net == "vgg16":
                clip_gradient(fasterRCNN, 10.)
            optimizer.step()

            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                print("[round %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                      % (num_round, epoch, step, iters_per_epoch, loss_temp, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
                

                loss_temp = 0
                start = time.time()

        save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}_{}.pth'.format(imdb_name,num_round, epoch, step))
        save_checkpoint({
          'session': args.session,
          'epoch': epoch + 1,
          'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
          'optimizer': optimizer.state_dict(),
          'pooling_mode': cfg.POOLING_MODE,
          'class_agnostic': args.class_agnostic,
        }, save_name)
        print('save model: {}'.format(save_name))
    return fasterRCNN    
        
def avgWeight(model_list):
    model_tmp=[None] * parties
    #optims_tmp=[None] * parties

    for idx, my_model in enumerate(model_list):
        
        model_tmp[idx] = my_model.state_dict()


    for key in model_tmp[0]:    
        #print(key)
        model_sum = 0

        for model_tmp_content in model_tmp:      
            
            model_sum += model_tmp_content[key]
            #print(model_tmp_content[key])
        for i in range(len(model_tmp)):
            #print("model_sum={}".format(model_sum))
            #print("len:{}".format(len(model_tmp)))
            model_avg = model_sum/len(model_tmp)
            #print("model_avg={}".format(model_avg))
            model_tmp[i][key] = model_sum/len(model_tmp)
    for i in range(len(model_list)):    
        model_list[i].load_state_dict(model_tmp[i])
        #optims_tmp[i] = Optims(workers, optim=optim.SGD(params=model_list[i].parameters(),lr=args.lr, momentum = args.momentum,weight_decay=args.weight_decay))
        #optims_tmp[i] = Optims(workers, optim=optim.Adam(params=model_list[i].parameters(),lr=args.lr))
    return model_list  #, optims_tmp
                    
    
if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")



    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    dataloader_list,iter_epochs_list = load_client_dataset(imdb_list)
    #dataloader = dataloader_list[0]
    print('# worker' + str(args.num_workers))
    # initilize the tensor holder here.
    


    if args.cuda:
        cfg.CUDA = True

#     fasterRCNN = initial_network(args)
#     optimizer = getOptimizer(fasterRCNN,args)
  
#     model_list  = train(args, dataloader_list[0], imdb_list[0], iter_epochs_list[0], fasterRCNN, optimizer,1)


    ROUND = args.round
    model_list=[None] * parties
    for i in range(1,ROUND+1):

        for idx,dataloader_item in enumerate(dataloader_list):    
            if i==1:
                model_list[idx] = initial_network(args)
            optimizer = getOptimizer(model_list[idx],args)
            model_list [idx] = train(args, dataloader_item,imdb_list[idx],iter_epochs_list[idx], model_list [idx], optimizer,i)

        model_list = avgWeight(model_list)

        save_name = os.path.join(output_dir, 'faster_rcnn_KAIST_AVG_{}.pth'.format(i))
        save_checkpoint({
          'session': 1,
          'epoch': ROUND,
          'model':  model_list[0].module.state_dict() if args.mGPUs else model_list[0].state_dict(), 
          'optimizer': optimizer.state_dict(),
          'pooling_mode': cfg.POOLING_MODE,
          'class_agnostic': args.class_agnostic,
        }, save_name)

