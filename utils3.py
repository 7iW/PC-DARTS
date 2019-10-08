""" Utilities """
import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import preproc


import os
import sys
#import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

    

import numpy as np
from sklearn.model_selection import StratifiedKFold

def kfold(files,labels,nfolds = 5, nsplit = 5):
  X = np.asarray(files)
  y = np.asarray(labels)
  skf = StratifiedKFold(n_splits=nfolds, random_state= 33,shuffle = True)
  skf.get_n_splits(X, y)
  i = 1
  for train_index, test_index in skf.split(X, y):
    if(i!=nsplit):
      i+=1
      continue
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    break
  return X_train,X_test,y_train,y_test

def load(path):
  import glob
  Xs,Ys = [],[]
  with open(path) as fp:
    for line in fp:
        if(len(line.split(','))<2):
           break
        name,label = line.split(',')
        Xs.append(name)
        Ys.append(int(label))
  return Xs,Ys

def get_data(dataset, data_path,val1_data_path,val2_data_path, cutout_length, validation,validation2 = False,n_class = 3,image_size = 64):
    """ Get torchvision dataset """
    dataset = dataset.lower()

    if dataset == 'cifar10':
        dset_cls = dset.CIFAR10
        n_classes = 10
    elif dataset == 'mnist':
        dset_cls = dset.MNIST
        n_classes = 10
    elif dataset == 'fashionmnist':
        dset_cls = dset.FashionMNIST
        n_classes = 10
    elif dataset == 'custom':
        dset_cls = dset.ImageFolder
        n_classes = n_class #2 to mama
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length,image_size)
    if dataset == 'custom':
        print("DATA PATH:", data_path)
        #trn_data = dset_cls(root=data_path, transform=trn_transform)
        
        Xs, Ys = load(data_path+'../labels.txt')
           
        X_train,X_test,y_train,y_test = kfold(Xs,Ys,2,1)#dividido em 5 folds, 1 forma de fold
        
        x_train_data = []
        x_test_data = []
        for x_path in X_train:
            x = cv2.imread(data_path+x_path) 
            #x_re = cv2.resize(x,(image_size,image_size))
            #rgb = cv2.merge([x_re,x_re,x_re])
            x_train_data.append(x)
        for x_path in X_test:
            x = cv2.imread(data_path+x_path) 
           
            #x_re = cv2.resize(x,(image_size,image_size))
            #rgb = cv2.merge([x_re,x_re,x_re])
            x_test_data.append(x)
            
        
        x_train_data = np.asarray(x_train_data)
        x_train_data = torch.from_numpy(x_train_data)
        print(x_train_data.shape)
        #x_train_data = x_train_data.permute(0, 3, 1, 2)

        x_test_data = np.asarray(x_test_data)
        x_test_data = torch.from_numpy(x_test_data)
        #x_test_data = x_test_data.permute(0, 3, 1, 2)
        
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        
        num_classes = 3
        one_hot_y_train = []
        one_hot_y_test = []
        for i in range(len(y_train)):
            #one_hot_y = np.zeros(num_classes)
            one_hot_y = torch.tensor([0, 0, 0])
            one_hot_y[y_train[i]] = 1
            one_hot_y_train.append(one_hot_y)
        for i in range(len(y_test)):
            #one_hot_y =np.zeros(num_classes)
            one_hot_y = torch.tensor([0, 0, 0])
            one_hot_y[y_test[i]] = 1
            one_hot_y_test.append(one_hot_y)
            
        tensor_train_x = torch.stack([torch.Tensor(i) for i in x_train_data]) # transform to torch tensors
        tensor_test_x = torch.stack([torch.Tensor(i) for i in x_test_data]) # transform to torch tensors
        
        #tensor_train_y = torch.from_numpy(np.asarray(one_hot_y_train))
        #tensor_test_y = torch.from_numpy(np.asarray(one_hot_y_test))
        
        #tensor_train_y = torch.stack([torch.Tensor(i) for i in one_hot_y_train])
        #tensor_test_y = torch.stack([torch.Tensor(i) for i in one_hot_y_test])
        tensor_train_y = torch.stack(one_hot_y_train)
        tensor_test_y = torch.stack(one_hot_y_test)
        import torch.utils.data as utils

        train_dataset = utils.TensorDataset(tensor_train_x,tensor_train_y) # create your datset
        test_dataset = utils.TensorDataset(tensor_test_x,tensor_test_y) # create your datset
        #train_dataloader = utils.DataLoader(train_dataset) # create your dataloader
        #dataset_loader = torch.utils.data.DataLoader(trn_data,
        #                                     batch_size=16, shuffle=True,
        #                                     num_workers=1)
        
    else:
        trn_data = dset_cls(root=data_path, train=True, download=True, transform=trn_transform)

    # assuming shape is NHW or NHWC
    if dataset == 'custom':
        shape = [1, image_size, image_size,3]
    else:
        shape = trn_data.train_data.shape
    print(shape)
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]
    print('input_size: uitls',input_size)
    
    ret = [input_size, input_channels, n_classes, train_dataset,test_dataset]
    
   
    return ret


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)
    #print('output:',output)
    #print('target:',target)
    #print('maxk:',maxk)
    ###TOP 5 NAO EXISTE NAS MAAMAS OU NO GEO. TEM QUE TRATAR
    maxk = 3 # Ignorando completamente o top5

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def save_checkpoint(model,epoch,w_optimizer,a_optimizer,loss, ckpt_dir, is_best=False, is_best_overall =False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'w_optimizer_state_dict': w_optimizer.state_dict(),
            'a_optimizer_state_dict': a_optimizer.state_dict(),
            'loss': loss
                }, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)
    if is_best_overall:
        best_filename = os.path.join(ckpt_dir, 'best_overall.pth.tar')
        shutil.copyfile(filename, best_filename)
        
def load_checkpoint(model,epoch,w_optimizer,a_optimizer,loss, filename='checkpoint.pth.tar'):
# Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        #print(checkpoint)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        w_optimizer.load_state_dict(checkpoint['w_optimizer_state_dict'])
        a_optimizer.load_state_dict(checkpoint['a_optimizer_state_dict'])
        loss = checkpoint['loss']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model,epoch,w_optimizer,a_optimizer,loss



def save_checkpoint2(model,epoch,optimizer,loss, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
                }, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'model.pth.tar')
        shutil.copyfile(filename, best_filename)
        
def load_checkpoint2(model,epoch,optimizer,loss, filename='model.pth.tar'):
    filename=filename+'checkpoint.pth.tar'
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        #print(checkpoint)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model,epoch,optimizer,loss
