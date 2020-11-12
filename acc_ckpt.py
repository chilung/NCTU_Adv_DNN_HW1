import os
import numpy as np
import torch


def acc_ckpt(dir_name):

    ckpt_list = os.listdir(dir_name)
    print('model check list: {}'.format(ckpt_list))

    epoch = []
    train_acc = []
    test_acc = []

    for ckpt_file in ckpt_list:
        if ckpt_file.find('.ckpt') != -1:
            ckpt = torch.load(os.path.join(dir_name, ckpt_file))
            print('epoch: {}, train_acc: {}, train_loss: {}, test_acc: {}, test_loss: {}'.format(
                ckpt['epoch'], ckpt['train_acc'], ckpt['train_loss'], ckpt['test_acc'], ckpt['test_loss']))
                
            epoch.append(ckpt['epoch'])
            train_acc.append(ckpt['train_acc'])
            test_acc.append(ckpt['test_acc'])

    sorted_idx = np.argsort(test_acc)[::-1]
    print([test_acc[i] for i in sorted_idx])
    print([epoch[i] for i in sorted_idx])

    return

acc_ckpt("./models/20201112_132148/")
