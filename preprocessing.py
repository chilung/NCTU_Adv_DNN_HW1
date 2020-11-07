#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import pandas as pd
import numpy as np
import torch


# In[2]:


root_path = './cs-t0828-2020-hw1'
train_src = 'training_data/training_data'
test_src = 'testing_data/testing_data'


# In[3]:


print('\n==> generate train file list: {}'.format('images.csv'))
print('train file directory: {}'.format(os.path.join(root_path, train_src)))

img_fn_list = os.listdir(os.path.join(root_path, train_src))
print('number of files: {}'.format(len(img_fn_list)))
#print(img_fn_list)

images = [[re.split(".jpg", img_fn)[0], img_fn] for img_fn in img_fn_list]
images.sort()
#print(images)

dfObj = pd.DataFrame(images, columns=['Image ID', 'Image File Name'])
print(dfObj)
dfObj.to_csv(os.path.join(root_path, 'images.csv'), index = False)


# In[4]:


print('\n==> generate test file list: {}'.format('test_images.csv'))
print('test file directory: {}'.format(os.path.join(root_path, test_src)))

img_fn_list = os.listdir(os.path.join(root_path, test_src))
print('number of files: {}'.format(len(img_fn_list)))
#print(img_fn_list)

images = [[re.split(".jpg", img_fn)[0], img_fn] for img_fn in img_fn_list]
images.sort()
#print(images)

dfObj = pd.DataFrame(images, columns=['Image ID', 'Image File Name'])
print(dfObj)
dfObj.to_csv(os.path.join(root_path, 'test_images.csv'), index = False)


# In[5]:


training_labels_csv_filename = 'training_labels.csv'
training_labels_pd = pd.read_csv(os.path.join(root_path, training_labels_csv_filename))
print(training_labels_pd)
mycar = training_labels_pd.values.tolist()
mycar.sort()
#print(mycar)


# In[6]:


brands = list(set([img_brand for img_idx, img_brand in mycar]))
brands.sort()
#print(len(brands))
#print(brands)

class_label = [[idx, brand] for idx, brand in enumerate(brands)]
#print(len(class_label))
#print(class_label)


# In[7]:


print('\n==> generate class file, containing class id and class name: {}'.format('class.csv'))
dfObj = pd.DataFrame(class_label, columns=['Class ID', 'Car Brand'])
print(dfObj)
dfObj.to_csv(os.path.join(root_path, 'class.csv'), index = False)


# In[8]:


from collections import OrderedDict 

name_to_id = {} 
for idx, brand in class_label:
    name_to_id[brand] = idx
#print(name_to_id)

id_to_name = {} 
for idx, brand in class_label:
    id_to_name[idx] = brand
#print(id_to_name)


# In[9]:


print('\n==> generate train label file and transfer train label name to label id: {}'.format('image_class_labels.csv'))
image_class_labels = [[img_idx, name_to_id[img_brand]] for img_idx, img_brand in mycar]
#print(image_class_labels)

dfObj = pd.DataFrame(image_class_labels, columns=['Image ID', 'Class ID'])
print(dfObj)
dfObj.to_csv(os.path.join(root_path, 'image_class_labels.csv'), index = False)


# In[10]:


print('\n==> generate train / train phase test split with ratio train/tets = 7/1: {}'.format('train_test_split.csv'))
train_test_split = []
split_ratio = 7+1
split = 1
for idx, class_label in image_class_labels:
    train_test_split.append([idx, 1 if (split%split_ratio) else 0])
    split += 1
#print(train_test_split)

dfObj = pd.DataFrame(train_test_split, columns=['Image ID', 'Train Test Split'])
print(dfObj)
dfObj.to_csv(os.path.join(root_path, 'train_test_split.csv'), index = False)


# In[11]:


def accuracy_log(f_name):

    f = open(f_name)
    lines = f.readlines()
    print(type(lines))

    epoch= []
    train_acc= []
    test_acc= []
    for line in lines:
        #print(line)
        matchObj = re.match(r'epoch:(\d+) - train loss: (\d+.\d+) and train acc: (\d+.\d+) total sample: (\d+)', line)
        if matchObj:
            #print('{}\n{}\n{}\n{}\n{}\n'.format(matchObj.group(0), matchObj.group(1), matchObj.group(2), matchObj.group(3), matchObj.group(4)))
            epoch.append(matchObj.group(1))
            train_acc.append(matchObj.group(3))
        matchObj = re.match(r'epoch:(\d+) - test loss: (\d+.\d+) and test acc: (\d+.\d+) total sample: (\d+)', line)
        if matchObj:
            #print('{}\n{}\n{}\n{}\n{}\n'.format(matchObj.group(0), matchObj.group(1), matchObj.group(2), matchObj.group(3), matchObj.group(4)))
            test_acc.append(matchObj.group(3))
    for idx in range(len(epoch)):
        print(epoch[idx], train_acc[idx], test_acc[idx])
    
    f.close()

    print(np.argmax(train_acc))
    test_max_idx = np.argmax(test_acc)
    print(epoch[test_max_idx], test_acc[test_max_idx])

    sorted_idx = np.argsort(test_acc)[::-1]
    print([test_acc[i] for i in sorted_idx])
    print([epoch[i] for i in sorted_idx])
    
    return

accuracy_log("./models/20201103_235321/train_test.log")


# In[12]:


def acc_ckpt(dir_name):

    ckpt_list = os.listdir(dir_name)
    print('model check list: {}'.format(ckpt_list))
    
    epoch= []
    train_acc= []
    test_acc= []

    for ckpt_file in ckpt_list:
        if ckpt_file.find('.ckpt') != -1: 
            ckpt = torch.load(os.path.join(dir_name, ckpt_file))
            print('epoch: {}, train_acc: {}, test_acc: {}'.format(ckpt['epoch'], ckpt['train_acc'], ckpt['test_acc']))
            
            epoch.append(ckpt['epoch'])
            train_acc.append(ckpt['train_acc'])
            test_acc.append(ckpt['test_acc'])

    sorted_idx = np.argsort(test_acc)[::-1]
    print([test_acc[i] for i in sorted_idx])
    print([epoch[i] for i in sorted_idx])

    return

acc_ckpt("./models/20201103_235321")


# In[ ]:





# In[ ]:





# In[ ]:




