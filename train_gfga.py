from copy import deepcopy
import torch
from tqdm import tqdm
import json

from models.wrapper import Wrapper
from evaluation import cal_similarity, eval_rank

import argparse
from utils.utils import load_checkpoint, get_dataloaders, save_checkpoint, set_seed

def main():

    # config file
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./configs/flickr30k_gfga_config.json', help='path of config file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    set_seed(config['seed'])
    
    # checkpoint
    exp_name = config['exp_name']
    logger, checkpoint = load_checkpoint(exp_name)
    if config['data_name'] == 'flickr30k':
        _, coco_checkpoint = load_checkpoint('coco_gfga', False)
    else:
        coco_checkpoint = None
    config = checkpoint['config'] if checkpoint is not None else config
    print(config)
    _, clip_checkpoint = load_checkpoint(config['clip_checkpoint'], need_logger=False)

    # Construct the wrapper
    device = torch.device(config['device'])
    wrapper = Wrapper(clip_name=config['clip_name'],
                img_dim=config['img_dim'], 
                txt_dim=config['txt_dim'],
                global_dim=config['global_dim'],
                hid_dim=config['hid_dim'],
                num_concepts=config['num_concepts'],
                num_dustbins=config['num_dustbins'],
                num_cross_layers=config['num_cross_layers'],
                num_cross_heads=config['num_cross_heads'],
                sinkhorn_eps=config['sinkhorn_eps'],
                sinkhorn_iters=config['sinkhorn_iters'],
                device=device,
                grad_clip=config['grad_clip'],
                lr=config['lr'],
                clip_param=clip_checkpoint['best_val_param'],
                num_epochs=config['num_epochs'],
                warmup_steps=config['warmup_steps'])
    wrapper.to(device)
    if coco_checkpoint is not None:
        wrapper.load_state_dict(coco_checkpoint['best_val_param'])
    if checkpoint is not None:
        wrapper.load_state_dict(checkpoint['best_val_param'])
    if coco_checkpoint is not None:
        del coco_checkpoint
    del clip_checkpoint
        

    dataloaders = get_dataloaders('./data/'+config['data_name'], config['data_file'], 
                                  wrapper.clip_model.transform, config['batch_size'])
    print('start training')

    start_epoch = checkpoint['epoch'] if checkpoint is not None else 1
    best_rsum = checkpoint['best_rsum'] if checkpoint is not None else 0
    best_val_param = checkpoint['best_val_param'] if checkpoint is not None else None
    num_epochs = config['num_epochs']
    del checkpoint


    # Train
    for epoch in range(start_epoch, num_epochs+1):

        #train for one epoch
        train(dataloaders['train'], wrapper, epoch, num_epochs)

        # evaluate on validation set
        with torch.no_grad():
            r1, r5, r10, medr, meanr, r1i, r5i, r10i, medri, meanri, rsum = validate(wrapper, dataloaders['val'], config['k_candidates'])

        logger.info('Epoch [%d/%d] image to text: %.4f %.4f %.4f %.4f %.4f' %(epoch, num_epochs, r1, r5, r10, medr, meanr))
        logger.info('Epoch [%d/%d] text to image: %.4f %.4f %.4f %.4f %.4f' %(epoch, num_epochs, r1i, r5i, r10i, medri, meanri))

        if rsum > best_rsum:
            best_rsum = rsum
            best_val_param = {k: v.cpu() for k, v in wrapper.state_dict().items()}
            logger.info('Epoch [%d/%d] best validation model has been saved.' %(epoch, num_epochs))
        
        save_checkpoint(exp_name=exp_name, config=config, epoch=epoch+1, best_rsum=best_rsum,
                        last_param=wrapper.state_dict(), best_val_param=best_val_param)
    
    print('start testing')
    best_val_param = {k: v.cuda() for k,v in best_val_param.items()}
    wrapper.load_state_dict(best_val_param)
    with torch.no_grad():
        r1, r5, r10, medr, meanr, r1i, r5i, r10i, medri, meanri, rsum = validate(wrapper, dataloaders['test'], config['k_candidates'])

    logger.info('image to text: %.4f %.4f %.4f %.4f %.4f' %(r1, r5, r10, medr, meanr))
    logger.info('text to image: %.4f %.4f %.4f %.4f %.4f' %(r1i, r5i, r10i, medri, meanri))
    


def train(train_loader, wrapper, epoch, num_epochs):
    wrapper.train()

    bar = tqdm(total=len(train_loader), desc='Epoch [%d/%d]' %(epoch, num_epochs))

    for i, train_data in enumerate(train_loader):
        images, texts, ids = train_data
        itc_loss= wrapper.train_emb(images, texts)
        bar.set_postfix(itc='%.4f'%(itc_loss))
        bar.update(1)
    
    bar.close()
    wrapper.linear_scheduler.step()
    return

def validate(wrapper, dataloader, k_candidates):
    # compute the encoding for all the validation images and captions
    wrapper.eval()
    sim = cal_similarity(wrapper, dataloader, k_candidates)

    img_to_txt_r1, img_to_txt_r5, img_to_txt_r10, img_to_txt_medr, img_to_txt_meanr, \
    txt_to_img_r1, txt_to_img_r5, txt_to_img_r10, txt_to_img_medr, txt_to_img_meanr = eval_rank(sim)
    
    # sum of recalls to be used for early stopping
    rsum =  img_to_txt_r1 + img_to_txt_r5 + img_to_txt_r10 + txt_to_img_r1 + txt_to_img_r5 + txt_to_img_r10

    return img_to_txt_r1, img_to_txt_r5, img_to_txt_r10, img_to_txt_medr, img_to_txt_meanr, \
    txt_to_img_r1, txt_to_img_r5, txt_to_img_r10, txt_to_img_medr, txt_to_img_meanr, rsum


if __name__ == '__main__':
    main()
