import torch
from utils.utils import load_checkpoint, get_dataloaders
import json
from models.wrapper import Wrapper

import torch
from torchvision.transforms import Compose, Resize, CenterCrop
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
import matplotlib.patches as patches

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import random

def _convert_image_to_rgb(image: Image):
    return image.convert("RGB")

def _transform(n_px: int):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        #ToTensor(),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def visualize_image_grey(image: Image, score: torch.Tensor, n_px=224, patch_size=14, topk=10):
    img_width, img_height = image.size
    
    patch_size = 14
    num_patches = score.shape
    patch_rows, patch_cols = num_patches

    flat_scores = score.flatten().detach().cpu().numpy()
    top_k_indices = np.argsort(flat_scores)[-topk:]

    fig, ax = plt.subplots()
    ax.imshow(image)

    for i in range(patch_rows):
        for j in range(patch_cols):
            patch_x = j * patch_size
            patch_y = i * patch_size
            
            if (i * patch_cols + j) not in top_k_indices:
                rect = patches.Rectangle((patch_x, patch_y), patch_size, patch_size, linewidth=1, 
                                         edgecolor='none', facecolor='white', alpha=0.8)
                ax.add_patch(rect)

    plt.axis('off')
    plt.show()

def visualize_image(image: Image, score: torch.Tensor, n_px=224, patch_size=14, mask_thresh=0):
    assert score.shape[0] == score.shape[1] and score.shape[0] == n_px // patch_size
    processor = _transform(n_px)
    image = processor(image)
    score = score.reshape(1, 1, 16, 16)
    score = torch.where(score < mask_thresh, torch.zeros_like(score).to(score.device), score)
    score = F.interpolate(score,size=[224,224],mode='bilinear')
    
    #alpha = torch.where(score.reshape(224,224).cpu() ==0, torch.zeros(224,224), torch.ones(224,224)*0.5)
    alpha = torch.ones(224,224)*0.5
    if torch.sum(alpha) > 0:
        score[score == 0] = torch.min(score[score > 0])
    plt.imshow(image)
    plt.imshow(score.squeeze().detach().cpu().numpy(), 
               alpha=alpha, cmap='jet')

    plt.axis('off')
    plt.show()
    plt.close()

def visualize_text(words: list[str], score: torch.Tensor):
    assert len(words) == len(score)
    word_scores = dict(zip(words, score.tolist()))

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_scores)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    plt.close()


if __name__ == '__main__':

    with open('./configs/coco_gfga_config.json', 'r') as f:
        config = json.load(f)
    device = torch.device(config['device'])
    _, clip_checkpoint = load_checkpoint(config['clip_checkpoint'], need_logger=False)
    model = Wrapper(clip_name=config['clip_name'],
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
    model.to(device)
    dataloaders, datasets = get_dataloaders('./data/MS-COCO', 'dataset_coco.json', 
                                  model.clip_model.transform, 1)
    _, checkpoint = load_checkpoint('coco_gfga', False)
    model.load_state_dict(checkpoint['best_val_param'])

    test_dataset = datasets['test']
    transform = _transform(224)

    #while True:
    for i in [9884,9884,9884]:
        #i = random.randint(0, len(test_dataset))
        #i=9371
        print('@@@ ' + str(i) + ' @@@')
        image, text, _ = test_dataset[i]

        print('============ Before Encode ============')
        plt.imshow(transform(image))
        plt.axis('off')
        plt.show()
        plt.close()
        print(text)
        from clip.simple_tokenizer import SimpleTokenizer
        import clip
        tokenizer = SimpleTokenizer()
        tokens = clip.tokenize(text)
        length = torch.argmax(tokens) + 1
        new_text = clip.tokenize(text).tolist()[0][0:length.item()]
        words = [tokenizer.decode([word]) for word in new_text]
        print(words)
        print('=========================================')

        image_tensor = test_dataset.transform(image)
        #continue


        # others
        # s = input()
        # while('n' not in s):

        # dustbin
        text = input()
        while '@' not in text:
        #while('n' not in s):
            for s in range(8):

                txt_token_idx = -int(s)
                
                img_score,txt_score = model.visualize(image_tensor.unsqueeze(0), [text], None, txt_token_idx)
                img_score = img_score[0,1:257].reshape(16,16)
                visualize_image(transform(image), img_score)
                
                tokens = clip.tokenize(text)
                length = torch.argmax(tokens) + 1
                new_text = clip.tokenize(text).tolist()[0][0:length.item()]
                words = [tokenizer.decode([word]) for word in new_text]
                txt_score = txt_score[0,1:length.item()-1]
                visualize_text(words[1:length.item()-1], torch.nn.functional.softmax(txt_score, dim=-1))
                
                #s = input()
                # while 'n' not in s:
                #     topk = int(s)
                #     visualize_image_grey(transform(image), img_score, topk=topk)
                #     s = input()
                #s = input()
            text = input()
            
        image.close()
        # (img_score, txt_score) = model.visualize(image_tensor.unsqueeze(0), [text], None, None)
        # img_score = img_score[0,1:257].reshape(16,16)
        # visualize_image(transform(image),img_score)
        # txt_score = txt_score[0,1:length.item()-1]
        # visualize_text(words[1:length.item()-1], txt_score)
        # s = input()
        # if s == 'e':
        #     break
        
        
         


        