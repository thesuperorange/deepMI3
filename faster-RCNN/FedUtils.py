from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import torch.nn as nn
import torch


def initial_network(imdb_classes,args):
    
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

def getOptimizer(fasterRCNN,args,cfg):
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
    
    
def load_model(imdb_classes,model_path, args,cfg):
    model = initial_network(imdb_classes,args)
    checkpoint = torch.load(model_path)
    
    if args.mGPUs:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])
    
    optimizer = getOptimizer(model,args,cfg)
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    start_round = checkpoint['round']
    
    return model,optimizer, start_round
    


def avgWeight(model_list,ratio_list):
    parties = len(model_list)
    model_tmp=[None] * parties
    #optims_tmp=[None] * parties

    for idx, my_model in enumerate(model_list):
        
        model_tmp[idx] = my_model.state_dict()


    for key in model_tmp[0]:    
        #print(key)
        model_avg = 0

        for idx, model_tmp_content in enumerate(model_tmp):     # add each model              
            model_avg += ratio_list[idx] * model_tmp_content[key]
            
        for i in range(len(model_tmp)):  #copy to each model            
            model_tmp[i][key] = model_avg
    for i in range(len(model_list)):    
        model_list[i].load_state_dict(model_tmp[i])
        
    return model_list  #, optims_tmp
    
    
    