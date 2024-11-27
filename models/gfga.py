import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class CrossAttentionLayer(nn.Module):

    def __init__(self, hid_dim, num_heads, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(hid_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(hid_dim, 4*hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(4*hid_dim, hid_dim)
        self.activation = nn.GELU()

        self.norm1 = LayerNorm(hid_dim)
        self.norm2 = LayerNorm(hid_dim)
        self.norm3 = LayerNorm(hid_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward_cross_attention(self, src, tgt, key_padding_mask=None):
        x = self.cross_attn(tgt, src, src, key_padding_mask=key_padding_mask)[0]
        x = self.dropout1(x)
        return x

    def forward_ff(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, tgt, key_padding_mask=None):
        x1 = self.forward_cross_attention(self.norm1(src), self.norm2(tgt), key_padding_mask)
        tgt = tgt + x1
        tgt = tgt + self.forward_ff(self.norm3(tgt))

        return tgt

class GFGA(nn.Module):

    def __init__(self, img_dim, txt_dim, global_dim, hid_dim, num_concepts, num_dustbins,
                 num_cross_layers, num_cross_heads, 
                 sinkhorn_eps, sinkhorn_iters, dropout=0.1) -> None:
        super(GFGA, self).__init__()

        self.img_dim = img_dim
        self.txt_dim = txt_dim
        self.hid_dim = hid_dim
        self.global_dim = global_dim
        self.num_cross_layers = num_cross_layers
        self.num_concepts = num_concepts
        self.num_dustbins = num_dustbins
        self.sinkhorn_eps = sinkhorn_eps
        self.sinkhorn_iters = sinkhorn_iters

        self.ln1 = LayerNorm(hid_dim)
        self.ln2 = LayerNorm(hid_dim)
        self.ln3 = LayerNorm(hid_dim)
        self.ln4 = LayerNorm(hid_dim)
        self.ln5 = LayerNorm(hid_dim)
        self.ln6 = LayerNorm(hid_dim)

        self.img_in_proj = nn.Linear(img_dim, hid_dim)
        self.txt_in_proj = nn.Linear(txt_dim, hid_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.img_concepts = nn.Parameter(torch.randn([1, num_concepts, hid_dim]))
        self.img_concept_pre_ln = LayerNorm(hid_dim)
        self.txt_concepts = nn.Parameter(torch.randn([1, num_concepts, hid_dim]))
        self.txt_concept_pre_ln = LayerNorm(hid_dim)

        self.img_out_proj = nn.Linear(hid_dim, hid_dim)
        self.txt_out_proj = nn.Linear(hid_dim, hid_dim)

        self.i2t_layers = nn.ModuleList([CrossAttentionLayer(hid_dim, num_cross_heads, dropout) for _ in range(num_cross_layers)])
        self.i2i_layers = nn.ModuleList([CrossAttentionLayer(hid_dim, num_cross_heads, dropout) for _ in range(num_cross_layers)])
        self.t2i_layers = nn.ModuleList([CrossAttentionLayer(hid_dim, num_cross_heads, dropout) for _ in range(num_cross_layers)])
        self.t2t_layers = nn.ModuleList([CrossAttentionLayer(hid_dim, num_cross_heads, dropout) for _ in range(num_cross_layers)])
        self.img_dustbin_projs = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(num_cross_layers)])
        self.txt_dustbin_projs = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for _ in range(num_cross_layers)])
        
        self.img_query = nn.Linear(global_dim, hid_dim)
        self.img_key = nn.Linear(hid_dim, hid_dim)
        self.txt_query = nn.Linear(global_dim, hid_dim)
        self.txt_key = nn.Linear(hid_dim, hid_dim)

        self.img_dustbins = nn.Parameter(torch.randn([1, self.num_dustbins, self.hid_dim]))
        self.img_dustbin_ln = LayerNorm(hid_dim)
        self.img_dustbin_proj = nn.Linear(hid_dim, hid_dim)
        self.txt_dustbins = nn.Parameter(torch.randn([1, self.num_dustbins, self.hid_dim]))
        self.txt_dustbin_ln = LayerNorm(hid_dim)
        self.txt_dustbin_proj = nn.Linear(hid_dim, hid_dim)

    def sinkhorn(self, mu, mv, cost_matrix):

        dtype = cost_matrix.dtype
        mu = mu.float()
        mv = mv.float()
        cost_matrix = cost_matrix.float()
        K = torch.exp(-cost_matrix / self.sinkhorn_eps)

        u = torch.ones_like(mu, device=mu.device, dtype=torch.float) # (B, R+1)
        v = torch.ones_like(mv, device=mv.device, dtype=torch.float) # (B, L+1)

        for i in range(self.sinkhorn_iters):
            u = mu / torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)
            v = mv / torch.bmm(K.permute(0,2,1), u.unsqueeze(-1)).squeeze(-1)

        T = torch.bmm(torch.diag_embed(u), K) # (B, R+1, R+1) @ (B, R+1, L+1) -> (B, R+1, L+1)
        T = torch.bmm(T, torch.diag_embed(v)) # (B, R+1, L+1) @ (B, L+1, L+1) -> (B, R+1, L+1)

        T = T.to(dtype=dtype)
        return T
    
    def img_hook(self, grad):
        self.img_grad = grad
    
    def txt_hook(self, grad):
        self.txt_grad = grad

    def forward(self, img_global_emb, img_nodes_emb, txt_global_emb, txt_nodes_emb, lengths):

        batch_size, num_img_nodes, _ = img_nodes_emb.shape
        batch_size, num_txt_nodes, _ = txt_nodes_emb.shape
        dtype = img_global_emb.dtype
        device = img_global_emb.device
        lengths = lengths.to(device=device)

        img_nodes_emb = self.img_in_proj(img_nodes_emb)
        txt_nodes_emb = self.txt_in_proj(txt_nodes_emb)

        mask = torch.zeros(batch_size, num_txt_nodes, dtype=torch.bool).to(txt_nodes_emb.device)
        for i, length in enumerate(lengths):
            mask[i, 0:length] = False
            mask[i, length:] = True

        # concept-based fusion
        img_nodes_emb = self.ln1(img_nodes_emb)
        txt_nodes_emb = self.ln2(txt_nodes_emb)

        img_concepts = self.img_concepts.repeat([batch_size, 1, 1]).to(dtype=dtype)
        img_concepts = self.img_concept_pre_ln(img_concepts)
        txt_concepts = self.txt_concepts.repeat([batch_size, 1, 1]).to(dtype=dtype)
        txt_concepts = self.txt_concept_pre_ln(txt_concepts)
        img_dustbins = self.img_dustbin_ln(self.img_dustbins.repeat([batch_size, 1, 1]))
        txt_dustbins = self.txt_dustbin_ln(self.txt_dustbins.repeat([batch_size, 1, 1]))

        for i in range(self.num_cross_layers):

            img_concepts = self.i2i_layers[i](src=img_nodes_emb, tgt=img_concepts, key_padding_mask=None)
            txt_concepts = self.t2t_layers[i](src=txt_nodes_emb, tgt=txt_concepts, key_padding_mask=mask)

            img_concepts = torch.cat([img_concepts, self.img_dustbin_projs[i](img_dustbins)], dim=1)
            txt_concepts = torch.cat([txt_concepts, self.txt_dustbin_projs[i](txt_dustbins)], dim=1)

            txt_nodes_emb = self.i2t_layers[i](src=img_concepts, tgt=txt_nodes_emb, key_padding_mask=None)
            img_nodes_emb = self.t2i_layers[i](src=txt_concepts, tgt=img_nodes_emb, key_padding_mask=None)
            img_concepts = img_concepts[:, 0:self.num_concepts]
            txt_concepts = txt_concepts[:, 0:self.num_concepts]


        img_nodes_emb = self.img_out_proj(self.ln3(img_nodes_emb))
        txt_nodes_emb = self.txt_out_proj(self.ln4(txt_nodes_emb))

        # node masker
        img_query = self.img_query(img_global_emb)
        img_key = self.img_key(img_nodes_emb)
        img_nodes_scores = torch.matmul(img_query.unsqueeze(1), img_key.permute(0,2,1)).squeeze(1)
        img_nodes_scores = img_nodes_scores / (self.hid_dim ** 0.5)
        img_nodes_scores = self.softmax(img_nodes_scores)
        
        txt_query = self.txt_query(txt_global_emb)
        txt_key = self.txt_key(txt_nodes_emb)
        txt_nodes_scores = torch.matmul(txt_query.unsqueeze(1), txt_key.permute(0,2,1)).squeeze(1)
        txt_nodes_scores = txt_nodes_scores / (self.hid_dim ** 0.5)
        txt_nodes_scores[mask] = -float('inf')
        txt_nodes_scores = self.softmax(txt_nodes_scores)

        # inconsistency-aware graph matching

        # dustbins
        img_dustbins = self.img_dustbin_proj(img_dustbins)
        txt_dustbins = self.txt_dustbin_proj(txt_dustbins)
        img_nodes_emb = torch.cat([img_nodes_emb, img_dustbins], dim=1)
        txt_nodes_emb = torch.cat([txt_nodes_emb, txt_dustbins], dim=1)

        img_norm_nodes_emb = F.normalize(img_nodes_emb, p=2, dim=-1)
        txt_norm_nodes_emb = F.normalize(txt_nodes_emb, p=2, dim=-1)
        sim_matrix = torch.matmul(img_norm_nodes_emb, txt_norm_nodes_emb.permute(0,2,1)) # (B, N, M)
        cost_matrix = 1 - sim_matrix

        dustbin_scores = torch.ones([batch_size, self.num_dustbins],device=device) / self.num_dustbins
        img_nodes_scores = torch.cat([img_nodes_scores, dustbin_scores], dim=-1)
        txt_nodes_scores = torch.cat([txt_nodes_scores, dustbin_scores], dim=-1)
        
        T = self.sinkhorn(img_nodes_scores, txt_nodes_scores, cost_matrix)

        nodes_s = torch.sum(T[:, 0:num_img_nodes, 0:num_txt_nodes] * sim_matrix[:, 0:num_img_nodes, 0:num_txt_nodes], dim=[-1,-2])

        s = nodes_s.reshape(-1)
                
        return s

        
    def visualize_forward(self, img_global_emb, img_nodes_emb, txt_global_emb, txt_nodes_emb, lengths,
                          img_token_idx=None, txt_token_idx=None):
        batch_size, num_img_nodes, _ = img_nodes_emb.shape
        batch_size, num_txt_nodes, _ = txt_nodes_emb.shape
        dtype = img_global_emb.dtype
        device = img_global_emb.device
        lengths = lengths.to(device=device)

        img_nodes_emb = self.img_in_proj(img_nodes_emb)
        txt_nodes_emb = self.txt_in_proj(txt_nodes_emb)

        mask = torch.zeros(batch_size, num_txt_nodes, dtype=torch.bool).to(txt_nodes_emb.device)
        for i, length in enumerate(lengths):
            mask[i, 0:length] = False
            mask[i, length:] = True

        # concept-based fusion
        img_nodes_emb = self.ln1(img_nodes_emb)
        txt_nodes_emb = self.ln2(txt_nodes_emb)

        img_concepts = self.img_concepts.repeat([batch_size, 1, 1]).to(dtype=dtype)
        img_concepts = self.img_concept_pre_ln(img_concepts)
        txt_concepts = self.txt_concepts.repeat([batch_size, 1, 1]).to(dtype=dtype)
        txt_concepts = self.txt_concept_pre_ln(txt_concepts)
        img_dustbins = self.img_dustbin_ln(self.img_dustbins.repeat([batch_size, 1, 1]))
        txt_dustbins = self.txt_dustbin_ln(self.txt_dustbins.repeat([batch_size, 1, 1]))

        for i in range(self.num_cross_layers):

            img_concepts = self.i2i_layers[i](src=img_nodes_emb, tgt=img_concepts, key_padding_mask=None)
            txt_concepts = self.t2t_layers[i](src=txt_nodes_emb, tgt=txt_concepts, key_padding_mask=mask)

            img_concepts = torch.cat([img_concepts, self.img_dustbin_projs[i](img_dustbins)], dim=1)
            txt_concepts = torch.cat([txt_concepts, self.txt_dustbin_projs[i](txt_dustbins)], dim=1)

            txt_nodes_emb = self.i2t_layers[i](src=img_concepts, tgt=txt_nodes_emb, key_padding_mask=None)
            img_nodes_emb = self.t2i_layers[i](src=txt_concepts, tgt=img_nodes_emb, key_padding_mask=None)
            img_concepts = img_concepts[:, 0:self.num_concepts]
            txt_concepts = txt_concepts[:, 0:self.num_concepts]

            if i == self.num_cross_layers - 2:
                concept_img_nodes_emb = img_nodes_emb.clone()
                concept_txt_nodes_emb = txt_nodes_emb.clone()

        img_nodes_emb = self.img_out_proj(self.ln3(img_nodes_emb))
        txt_nodes_emb = self.txt_out_proj(self.ln4(txt_nodes_emb))

        # node masker
        img_query = self.img_query(img_global_emb)
        img_key = self.img_key(img_nodes_emb)
        img_nodes_scores = torch.matmul(img_query.unsqueeze(1), img_key.permute(0,2,1)).squeeze(1)
        img_nodes_scores = img_nodes_scores / (self.hid_dim ** 0.5)
        img_nodes_scores = self.softmax(img_nodes_scores)
        
        txt_query = self.txt_query(txt_global_emb)
        txt_key = self.txt_key(txt_nodes_emb)
        txt_nodes_scores = torch.matmul(txt_query.unsqueeze(1), txt_key.permute(0,2,1)).squeeze(1)
        txt_nodes_scores = txt_nodes_scores / (self.hid_dim ** 0.5)
        txt_nodes_scores[mask] = -float('inf')
        txt_nodes_scores = self.softmax(txt_nodes_scores)
        
        # inconsistency-aware graph matching

        # dustbins
        img_dustbins = self.img_dustbin_proj(img_dustbins)
        txt_dustbins = self.txt_dustbin_proj(txt_dustbins)
        img_nodes_emb = torch.cat([img_nodes_emb, img_dustbins], dim=1)
        txt_nodes_emb = torch.cat([txt_nodes_emb, txt_dustbins], dim=1)

        img_norm_nodes_emb = F.normalize(img_nodes_emb, p=2, dim=-1)
        img_norm_nodes_emb.register_hook(self.img_hook)
        txt_norm_nodes_emb = F.normalize(txt_nodes_emb, p=2, dim=-1)
        txt_norm_nodes_emb.register_hook(self.txt_hook)
        sim_matrix = torch.matmul(img_norm_nodes_emb, txt_norm_nodes_emb.permute(0,2,1)) # (B, N, M)
        cost_matrix = 1 - sim_matrix

        dustbin_scores = torch.ones([batch_size, self.num_dustbins],device=device) / self.num_dustbins
        img_nodes_scores = torch.cat([img_nodes_scores, dustbin_scores], dim=-1)
        txt_nodes_scores = torch.cat([txt_nodes_scores, dustbin_scores], dim=-1)
        

        T = self.sinkhorn(img_nodes_scores, txt_nodes_scores, cost_matrix)


        assert img_token_idx is None or txt_token_idx is None

        # alignment
        if img_token_idx is not None:
            s = -torch.sum(T[:,img_token_idx,0:num_txt_nodes] * sim_matrix[:,img_token_idx,0:num_txt_nodes])
            s.backward()
            txt_nodes_grad = self.txt_grad # (B,M,D)
            txt_nodes_grad = torch.mean(txt_nodes_grad[~mask.unsqueeze(0).repeat([1,1,self.hid_dim])], dim=1, keepdim=True) # (B,1,D)
            txt_scores = F.relu(torch.sum(txt_norm_nodes_emb * txt_nodes_grad, dim=-1)) # (B,M)
            return txt_scores, None
        elif txt_token_idx is not None:
            # s = torch.sum(T[:,0:num_img_nodes,txt_token_idx] * sim_matrix[:,0:num_img_nodes,txt_token_idx])
            # s.backward()
            # img_nodes_grad = self.img_grad # (B,N,D)
            # img_nodes_grad = torch.mean(img_nodes_grad, dim=1, keepdim=True)
            # img_scores = F.relu(torch.sum(img_norm_nodes_emb * img_nodes_grad, dim=-1))
            # return img_scores, None

            ## masker
            #return img_nodes_scores[:, 0:num_img_nodes], txt_nodes_scores[:, 0:num_txt_nodes]

            ## dustbin
            img_scores = T[:,0:num_img_nodes,txt_token_idx]
            txt_scores = T[:, txt_token_idx, 0:num_txt_nodes]
            return img_scores, txt_scores

            ## concepts
            norm_image_concepts = F.normalize(img_concepts,p=2,dim=-1)
            norm_image_concept_nodes = F.normalize(concept_img_nodes_emb,p=2,dim=-1)
            sim = norm_image_concepts @ norm_image_concept_nodes.permute(0,2,1) # (B,C,N)
            img_sim = sim[:, txt_token_idx]
            norm_text_concepts = F.normalize(txt_concepts, p=2, dim=-1)
            norm_text_concept_nodes = F.normalize(concept_txt_nodes_emb, p=2, dim=-1)
            sim = norm_text_concepts @ norm_text_concept_nodes.permute(0,2,1)
            txt_sim = sim[:, txt_token_idx]
            return img_sim, txt_sim