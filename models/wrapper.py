import torch
import torch.nn.init
from torch.nn.utils.clip_grad import clip_grad_norm_
from .CLIP import FineTunedCLIP
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, LinearLR
from .gfga import GFGA

class Wrapper(object):


    def __init__(self, clip_name, img_dim, txt_dim, global_dim, hid_dim, num_concepts, num_dustbins,
                num_cross_layers, num_cross_heads, sinkhorn_eps, sinkhorn_iters, device, grad_clip, lr, clip_param, 
                num_epochs, warmup_steps):

        self.device = device
        self.grad_clip = grad_clip
        self.num_concepts = num_concepts
        self.hid_dim = hid_dim
        self.img_dim = img_dim
        self.txt_dim = txt_dim

        # fine-tuned clip (frozen)
        self.clip_model = FineTunedCLIP(clip_name, 1e-8, device, grad_clip)
        self.clip_model.load_state_dict(clip_param)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        # GFGA
        self.model = GFGA(img_dim=img_dim, txt_dim=txt_dim, global_dim=global_dim, hid_dim=hid_dim, num_cross_heads=num_cross_heads,
                         num_concepts=num_concepts, num_dustbins=num_dustbins,
                          num_cross_layers=num_cross_layers, sinkhorn_eps=sinkhorn_eps, sinkhorn_iters=sinkhorn_iters, dropout=0)
        
        # optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=[0.9,0.98], weight_decay=0.05)
        self.linear_scheduler = LinearLR(self.optimizer, start_factor=lr, end_factor=1e-6, total_iters=num_epochs)
        self.warmup_steps = warmup_steps
        self.warmup_scheduler = LambdaLR(self.optimizer, lambda step: step/warmup_steps)

    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def to(self, device):
        self.device = device
        self.clip_model.to(device)
        self.model.to(device)
    
    def state_dict(self):
        state_dict = self.model.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
    
    def encode_image(self, images):
        with torch.no_grad():
            return self.clip_model.encode_image(images)

    def encode_text(self, texts):
        with torch.no_grad():
            return self.clip_model.encode_text(texts)

    def forward_global_emb(self, images, texts):
        with torch.no_grad():
            return self.clip_model.forward_global_emb(images, texts)
    
    def forward_local_emb(self, img_global_emb, img_emb, txt_global_emb, txt_emb, txt_lengths):
        return self.model(img_global_emb, img_emb, txt_global_emb, txt_emb, txt_lengths)

    def forward_gfga(self, img_global_emb, image_features, 
                    txt_global_emb, text_features, text_lengths):

        batch_size = image_features.shape[0]

        sim_matrix = img_global_emb @ txt_global_emb.t()
        sim_matrix[torch.arange(batch_size), torch.arange(batch_size)] = -float('inf')
        _, i2t_index = torch.max(sim_matrix, dim=-1)
        _, t2i_index = torch.max(sim_matrix, dim=0)

        pos_sim = self.model(img_global_emb, image_features, txt_global_emb, text_features, text_lengths, True)
        i2t_sim = self.model(img_global_emb, image_features, txt_global_emb[i2t_index], text_features[i2t_index], text_lengths[i2t_index], False)
        t2i_sim = self.model(img_global_emb[t2i_index], image_features[t2i_index], txt_global_emb, text_features, text_lengths, False)

        sim_matrix[torch.arange(batch_size), torch.arange(batch_size)] = pos_sim
        sim_matrix[torch.arange(batch_size), i2t_index] = i2t_sim
        sim_matrix[t2i_index, torch.arange(batch_size)] = t2i_sim

        logit_scale = self.clip_model.clip_model.logit_scale.exp()

        logits_per_image = sim_matrix * logit_scale
        logits_per_text = logits_per_image.t()
        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        itc_loss = (loss_i + loss_t) / 2.0

        return itc_loss
        

    def train_emb(self, images, texts):

        for param in self.model.parameters():
            param.grad = None
        img_global_emb, img_nodes_emb,  \
        txt_global_emb, txt_nodes_emb, lengths = self.forward_global_emb(images, texts)
        itc_loss = self.forward_gfga(img_global_emb, img_nodes_emb, txt_global_emb, txt_nodes_emb, lengths)
        loss = itc_loss
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        if self.warmup_steps > 0:
            self.warmup_steps -= 1
            self.warmup_scheduler.step()

        return itc_loss
    
    def visualize(self, images, texts, img_token_idx, txt_token_idx):
        for param in self.model.parameters():
            param.grad = None
        img_global_emb, img_nodes_emb,  \
        txt_global_emb, txt_nodes_emb, lengths = self.forward_global_emb(images, texts)
        scores = self.model.visualize_forward(img_global_emb, img_nodes_emb, txt_global_emb, txt_nodes_emb, lengths, img_token_idx, txt_token_idx)
        return scores


