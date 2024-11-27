from tqdm import tqdm
import math
import torch

def global_encode_data(model, dataloader):

    img_global_embs = []
    img_nodes_embs = []
    
    txt_global_embs = []
    txt_nodes_embs = []
    txt_lens = []

    bar = tqdm(total=len(dataloader), desc='global encoding')

    for i, data in enumerate(dataloader):
        images, texts, ids = data
        image_mask = ids % 5 == 0
        images = images[image_mask]

        img_global_emb, img_nodes_emb = model.encode_image(images.to(model.device))
            
        img_global_embs.append(img_global_emb.detach().cpu())
        img_nodes_embs.append(img_nodes_emb.detach().cpu())

        txt_global_emb,txt_nodes_emb,txt_len = model.encode_text(texts)

        txt_global_embs.append(txt_global_emb.detach().cpu())
        txt_nodes_embs.append(txt_nodes_emb.detach().cpu())
        txt_lens.append(txt_len.detach().cpu())
        bar.update(1)
    bar.close()

    img_global_embs = torch.cat(img_global_embs, dim=0)
    img_nodes_embs = torch.cat(img_nodes_embs, dim=0)
    txt_global_embs = torch.cat(txt_global_embs, dim=0)
    txt_nodes_embs = torch.cat(txt_nodes_embs, dim=0)
    txt_lens = torch.cat(txt_lens, dim=0)

    return img_global_embs, img_nodes_embs, txt_global_embs, txt_nodes_embs, txt_lens

def local_encode_data(model, img_global_embs, img_nodes_embs, txt_global_embs, txt_nodes_embs,  lengths, candidates, batch_size):
    
    num_batch = math.ceil(len(candidates) / batch_size)
    bar = tqdm(total=num_batch, desc='local encoding')
    local_sim = []
    for i in range(num_batch):
        begin = batch_size*i
        end = min(batch_size*(i+1), len(candidates))
        candidate = candidates[begin:end]
        sim = model.forward_local_emb(img_global_embs[candidate[:, 0]].to(model.device),
                                                    img_nodes_embs[candidate[:,0]].to(model.device), 
                                                   txt_global_embs[candidate[:,1]].to(model.device),
                                                   txt_nodes_embs[candidate[:,1]].to(model.device), 
                                                   lengths[candidate[:,1]])
        local_sim.append(sim.detach().cpu())
        bar.update(1)
    bar.close()
    local_sim = torch.cat(local_sim, dim=0)
    return local_sim


def cal_similarity(model, dataloader, k_candidates):

    img_global_embs, img_nodes_embs, \
    txt_global_embs, txt_nodes_embs, lengths = global_encode_data(model, dataloader)
    sim = torch.matmul(img_global_embs.to(model.device), txt_global_embs.to(model.device).t()) # (B,B)
    if k_candidates == 0:
        return sim

    final_score = torch.ones_like(sim, device=sim.device) * -100.0 
    
    if k_candidates > 0:
        _, i2t_topk_index = torch.topk(sim, k=k_candidates, dim=-1)
        _, t2i_topk_index = torch.topk(sim, k=k_candidates, dim=0)
        t2i_topk_index = t2i_topk_index.t()

        candidates = set()
        for image_id in range(sim.shape[0]):
            for txt_id in i2t_topk_index[image_id]:
                candidates.add((image_id, txt_id.item()))

        for txt_id in range(sim.shape[1]):
            for image_id in t2i_topk_index[txt_id]:
                candidates.add((image_id.item(), txt_id))
        candidates = list(candidates)
        candidates = torch.tensor(candidates)

        local_sim = local_encode_data(model, 
                                img_global_embs,
                                img_nodes_embs, 
                                txt_global_embs,
                                txt_nodes_embs,
                                lengths,
                                candidates,
                                128)
    

        final_score[candidates[:,0], candidates[:,1]] = sim[candidates[:,0], candidates[:,1]] + 1.0*local_sim.to(sim.device)

    return final_score

def eval_rank(similarities):
    """
        evaluate the retrival rank of model.

        Args:
            similarities: (BI, BT)
            num_imgs: int, equal to BI
            num_txts: int, equat to BT
            num_txts_per_img: int, the number of texts for each image. 
                For example, if num_txts_per_img is 5, then texts[0:5] all corresponds to images[0]
        
        Returns:
            [(r1,r5,r10,rmed,rmean), (r1,r5,r10,rmed,rmean)] (image-to-text and text-to-image)
    """ 

    num_imgs, num_txts = similarities.shape
    assert num_txts % num_imgs == 0
    num_txts_per_img = num_txts // num_imgs

    img_to_txt_ranks = torch.zeros([num_imgs])
    txt_to_img_ranks = torch.zeros([num_txts])

    bar = tqdm(total=num_imgs + num_txts, desc='calculating ranks')

    for img_idx in range(num_imgs):

        txt_idxs = torch.argsort(similarities[img_idx], dim=-1, descending=True) # (BT)

        rank = 1e20
        for txt_idx in range(img_idx*num_txts_per_img, (img_idx+1)*num_txts_per_img):

            tmp = torch.where(txt_idxs == txt_idx)[0][0].item()
            if tmp < rank:
                rank = tmp

        img_to_txt_ranks[img_idx] = rank
        bar.update(1)

    for txt_idx in range(num_txts):

        img_idxs = torch.argsort(similarities[:, txt_idx], dim=0, descending=True) # (BI)

        img_idx = txt_idx // num_txts_per_img

        rank = torch.where(img_idxs == img_idx)[0][0].item()

        txt_to_img_ranks[txt_idx] = rank

        bar.update(1)
    bar.close()
    
    img_to_txt_r1 = 100.0 * len(torch.where(img_to_txt_ranks < 1)[0]) / num_imgs
    img_to_txt_r5 = 100.0 * len(torch.where(img_to_txt_ranks < 5)[0]) / num_imgs
    img_to_txt_r10 = 100.0 * len(torch.where(img_to_txt_ranks < 10)[0]) / num_imgs
    img_to_txt_medr = torch.floor(torch.median(img_to_txt_ranks)).item() + 1
    img_to_txt_meanr = img_to_txt_ranks.mean().item() + 1

    txt_to_img_r1 = 100.0 * len(torch.where(txt_to_img_ranks < 1)[0]) / num_txts
    txt_to_img_r5 = 100.0 * len(torch.where(txt_to_img_ranks < 5)[0]) / num_txts
    txt_to_img_r10 = 100.0 * len(torch.where(txt_to_img_ranks < 10)[0]) / num_txts
    txt_to_img_medr = torch.floor(torch.median(txt_to_img_ranks)).item() + 1
    txt_to_img_meanr = txt_to_img_ranks.mean().item() + 1

    return img_to_txt_r1, img_to_txt_r5, img_to_txt_r10, img_to_txt_medr, img_to_txt_meanr, \
            txt_to_img_r1, txt_to_img_r5, txt_to_img_r10, txt_to_img_medr, txt_to_img_meanr
