import torch
import torch.nn.init
from torch.nn.utils.clip_grad import clip_grad_norm_
import clip
import torch.nn.functional as F

class FineTunedCLIP(object):

    def __init__(self, clip_name, lr, device, grad_clip):

        model, transform = clip.load(clip_name, device)
        self.clip_model = model
        self.transform = transform
        self.tune_params = self.clip_model.parameters()
        self.optimizer = torch.optim.Adam(self.tune_params, lr=lr, eps=1e-6)
        self.device = device
        self.grad_clip = grad_clip

        for name, layer in self.clip_model.named_modules():
            if name == 'transformer':
                layer.register_forward_hook(self.text_hook_fn)
            if name == 'visual.transformer':
                layer.register_forward_hook(self.image_hook_fn)
    
    def text_hook_fn(self, module, input, output):
        self.txt_feature = output.permute(1,0,2).float()
    
    def image_hook_fn(self, module, input, output):
        self.img_feature = output.permute(1,0,2).float()

    def train(self):
        return
    
    def eval(self):
        self.clip_model.eval()
    
    def to(self, device):
        self.clip_model.to(device)
    
    def state_dict(self):
        return self.clip_model.state_dict()

    def load_state_dict(self, state_dict):
        self.clip_model.load_state_dict(state_dict)
    
    def parameters(self):
        return self.clip_model.parameters()
    
    def encode_image(self, images):
        images = images.to(self.device)
        img_global_feature = self.clip_model.encode_image(images).float()
        img_local_feature = self.img_feature.float()
        img_global_feature = F.normalize(img_global_feature, p=2, dim=-1)
        return img_global_feature, img_local_feature

    def encode_text(self, texts):
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        lengths = torch.argmax(tokens, dim=-1) + 1
        txt_global_feature = self.clip_model.encode_text(tokens).float()
        txt_local_feature = self.txt_feature.float()
        txt_global_feature = F.normalize(txt_global_feature, p=2, dim=-1)
        return txt_global_feature, txt_local_feature, lengths

    def forward_global_emb(self, images, texts):
        img_global_emb, img_local_emb = self.encode_image(images)

        txt_global_emb, txt_local_emb, lengths = self.encode_text(texts)

        return img_global_emb, img_local_emb, \
                txt_global_emb, txt_local_emb, lengths
    
    def foward_logits(self, images, texts):
        images = images.to(self.device)
        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        return self.clip_model(images, tokens)

    def forward_nce_loss(self, logits_per_image, logits_per_text):
        labels = torch.arange(logits_per_image.size(0)).to(logits_per_image.device)
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2.0
        return loss.sum()

    def train_emb(self, images, texts):

        self.optimizer.zero_grad()
        logits_per_image, logits_per_text = self.foward_logits(images, texts)
        global_loss = self.forward_nce_loss(logits_per_image, logits_per_text)
        global_loss.backward()
        clip_grad_norm_(self.tune_params, self.grad_clip)
        self.optimizer.step()

        return global_loss.item()
