import json
import torch
from torch.utils.data import DataLoader, Dataset
import os
import logging
import numpy as np
from PIL import Image
import random

class ImgTxtDataset(Dataset):

    def __init__(self, images, texts, transform):

        super(ImgTxtDataset, self).__init__()

        self.transform = transform
        self.images = images
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        
        image = self.images[index]
        with Image.open(image).convert("RGB") as image:
            image = self.transform(image)
            #image = image
        text = self.texts[index]
        return image, text, index

def collate_fn(data):

    images, texts, indexes = zip(*data)
    images = torch.stack(images, dim=0)
   
    return images, list(texts), torch.tensor(indexes)
 

def get_dataloaders(data_dir, data_file, transform, batch_size):
    with open(os.path.join(data_dir, data_file), 'r') as f:
        data = json.load(f)
    
    raw_data = {'train':{'images':[], 'texts':[]}, 
                'val':{'images':[], 'texts':[]},
                'test':{'images':[], 'texts':[]}}
    for item in data['images']:
        split = item['split']
        if split == 'restval':
            split = 'train'
        path = item['filepath'] if 'filepath' in item.keys() else 'images'
        filepath = os.path.join(data_dir, path, item['filename']) 
        for i in range(len(item['sentences'])):
            if i == 5 and split != 'train':
                print('warning: {image} has {num} captions.'.format(image=item['filename'], num=len(item['sentences'])))
                break
            raw_data[split]['images'].append(filepath)
            text = item['sentences'][i]['raw']
            raw_data[split]['texts'].append(text)

    dataloaders = {}
    datasets = {}
    for split in ['train', 'val', 'test']:
        dataset = ImgTxtDataset(raw_data[split]['images'], raw_data[split]['texts'], transform)
        dataloader = DataLoader(dataset, batch_size, shuffle=(split == 'train'), collate_fn=collate_fn, num_workers=8, pin_memory=True)
        dataloaders[split] = dataloader
        datasets[split] = dataset
    
    print('finish loading datasets.')
    print('train pairs: %d, validation pairs: %d, test pairs: %d' %(len(raw_data['train']['images']), len(raw_data['val']['images']), len(raw_data['test']['images'])))

    return dataloaders #, datasets


def get_logger(exp_name):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    log_file = './checkpoints/%s/gfga.log' %(exp_name)
    handler = logging.FileHandler(log_file, mode='a')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def load_checkpoint(exp_name, need_logger=True):
    
    checkpoint_dir = './checkpoints/' + exp_name
    if not os.path.exists(checkpoint_dir):
        print('experiment %s does not exists, creating a new checkpoint' %(exp_name))
        os.makedirs(checkpoint_dir)
    else:
        print('experiment %s already exists, loading last checkpoint' %(exp_name))
    
    if need_logger:
        logger = get_logger(exp_name)
    else:
        logger = None

    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    checkpoint = None
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
    
    return logger, checkpoint

def save_checkpoint(exp_name, config, epoch, best_rsum, last_param, best_val_param):
    checkpoint_dir = './checkpoints/' + exp_name
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    checkpoint = {'config':config, 'epoch':epoch, 'best_rsum':best_rsum, 
                'last_param':last_param, 'best_val_param':best_val_param}
    torch.save(checkpoint, checkpoint_file)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)