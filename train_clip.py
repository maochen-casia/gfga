from copy import deepcopy
import torch
from tqdm import tqdm
import json
from evaluation import cal_similarity, eval_rank
from models.CLIP import FineTunedCLIP
import argparse
from utils.utils import load_checkpoint, save_checkpoint, get_dataloaders, set_seed

def main():

    # config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='./configs/flickr30k_clip_config.json', help='path of config file')
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    set_seed(config['seed'])
    
    # checkpoint
    exp_name = config['exp_name']
    logger, checkpoint = load_checkpoint(exp_name)
    config = checkpoint['config'] if checkpoint is not None else config
    print(config)

    # Construct the model
    device = torch.device(config['device'])
    model = FineTunedCLIP(clip_name=config['clip_name'],
                          lr=config['lr'],
                          device=device,
                          grad_clip=config['grad_clip'])
    if checkpoint is not None:
        model.load_state_dict(checkpoint['best_val_param'])
    model.to(device)

    dataloaders = get_dataloaders('./data/'+config['data_name'], config['data_file'], 
                                  model.transform, config['batch_size'])
    train_loader = dataloaders['train']
    print('finish loading data')

    print('start training')

    start_epoch = checkpoint['epoch'] if checkpoint is not None else 1
    best_rsum = checkpoint['best_rsum'] if checkpoint is not None else 0
    best_val_param = checkpoint['best_val_param'] if checkpoint is not None else None
    num_epochs = config['num_epochs']
    del checkpoint

    # Train the Model

    for epoch in range(start_epoch, num_epochs+1):

        #train for one epoch
        train(train_loader, model, epoch, num_epochs)

        # evaluate on validation set
        with torch.no_grad():
            r1, r5, r10, medr, meanr, r1i, r5i, r10i, medri, meanri, rsum = validate(model, dataloaders['val'], config['k_candidates'])

        logger.info('Epoch [%d/%d] image to text: %.4f %.4f %.4f %.4f %.4f' %(epoch, num_epochs, r1, r5, r10, medr, meanr))
        logger.info('Epoch [%d/%d] text to image: %.4f %.4f %.4f %.4f %.4f' %(epoch, num_epochs, r1i, r5i, r10i, medri, meanri))

        if rsum > best_rsum:
            best_rsum = rsum
            best_val_param = {k: v.cpu() for k, v in model.state_dict().items()}
            logger.info('Epoch [%d/%d] best validation model has been saved.' %(epoch, num_epochs))
        
        save_checkpoint(exp_name=exp_name, config=config, epoch=epoch+1, best_rsum=best_rsum,
                        last_param=model.state_dict(), best_val_param=best_val_param)
    
    print('start testing')
    best_val_param = {k: v.cuda() for k,v in best_val_param.items()}
    model.load_state_dict(best_val_param)
    with torch.no_grad():
        r1, r5, r10, medr, meanr, r1i, r5i, r10i, medri, meanri, rsum = validate(model, dataloaders['test'], config['k_candidates'])

    logger.info('image to text: %.4f %.4f %.4f %.4f %.4f' %(r1, r5, r10, medr, meanr))
    logger.info('text to image: %.4f %.4f %.4f %.4f %.4f' %(r1i, r5i, r10i, medri, meanri))
    


def train(train_loader, model, epoch, num_epochs):
    model.train()

    bar = tqdm(total=len(train_loader), desc='Epoch [%d/%d]' %(epoch, num_epochs))

    for i, train_data in enumerate(train_loader):
        images, texts, ids = train_data
        itc_loss = model.train_emb(images, texts)
        bar.set_postfix(itc='%.4f'%(itc_loss))
        bar.update(1)
    
    bar.close()
    return

def validate(model, dataloader, k_candidates):
    # compute the encoding for all the validation images and captions
    model.eval()
    sim = cal_similarity(model, dataloader, k_candidates)

    img_to_txt_r1, img_to_txt_r5, img_to_txt_r10, img_to_txt_medr, img_to_txt_meanr, \
    txt_to_img_r1, txt_to_img_r5, txt_to_img_r10, txt_to_img_medr, txt_to_img_meanr = eval_rank(sim)
    
    # sum of recalls to be used for early stopping
    rsum =  img_to_txt_r1 + img_to_txt_r5 + img_to_txt_r10 + txt_to_img_r1 + txt_to_img_r5 + txt_to_img_r10

    return img_to_txt_r1, img_to_txt_r5, img_to_txt_r10, img_to_txt_medr, img_to_txt_meanr, \
    txt_to_img_r1, txt_to_img_r5, txt_to_img_r10, txt_to_img_medr, txt_to_img_meanr, rsum


if __name__ == '__main__':
    main()
