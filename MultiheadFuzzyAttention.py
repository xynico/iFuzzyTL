from typing import Optional, Tuple
import torch
from torch import fft
from torch import nn, Tensor
from einops import rearrange
# from .FuzzyTransformer import FuzzyMultiheadAttention
from functools import partial

class NormalizeLayer(nn.Module):
    def __init__(self, min_val, max_val):
        super(NormalizeLayer, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        # Normalize to [0, 1]
        x_normalized = (x - self.min_val) / (self.max_val - self.min_val)
        # Scale to [-1, 1]
        x_scaled = x_normalized * 2 - 1
        return x_scaled

class fakedata(object):
    is_cuda = False
    device='cca'
    weight=torch.rand(1,1)
    bias=torch.rand(1,1)

class FuzzyMultiheadAttention(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 num_heads, 
                 num_rules,
                 seq_len, 
                 dropout=0.,
                 use_projection=True,
                 norm=False, 
                 high_dim=False,
                 softmax = 'softmax',
                 method='l1',
                 ):
        super().__init__()
        self.batch_first = True
        self._qkv_same_embed_dim = True
        # Ensure embed_dim is divisible by num_heads
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.num_rules = num_rules
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.high_dim = high_dim
        self.norm=None if not norm else NormalizeLayer(0, 1)
        self.method = method

        # Initialize rules (keys and values) as parameters
        if self.high_dim:
            self.rules_keys = nn.Parameter(torch.Tensor(num_rules, num_heads, seq_len, self.head_dim))
            self.rules_widths = nn.Parameter(torch.ones(num_rules, num_heads, seq_len, self.head_dim))
            self.value_proj = nn.Linear(embed_dim, embed_dim*num_rules)
        else:
            self.rules_keys = nn.Parameter(torch.Tensor(self.num_heads, self.num_rules, self.head_dim))
            self.rules_widths = nn.Parameter(torch.ones(self.num_heads, self.num_rules, self.head_dim))
            self.value_proj = nn.Linear(embed_dim, embed_dim*num_rules)

        if use_projection:
            self.query_proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.query_proj = nn.Identity()
            # raise NotImplementedError("No query projection is not supported")
  

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = lambda x: getattr(nn.functional, softmax)(x, dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.in_proj_weight=fakedata()
        self.in_proj_bias=fakedata()
        if self.num_rules != 2:
            nn.init.normal_(self.rules_keys, mean=0.0, std=0.02)
        else:
            nn.init.normal_(self.rules_keys, mean=0.0, std=0.02)
            # (B, n_rule, embed_dim)
            # the first rule is high in the 1:embed_dim/2, low in the embed_dim/2:embed_dim
            # the second rule is low in the 1:embed_dim/2, high in the embed_dim/2:embed_dim

            nn.init.normal_(self.rules_keys[:,0, 0:embed_dim//2], mean=1, std=0.02)
            nn.init.normal_(self.rules_keys[:,1, embed_dim//2:], mean=-1, std=0.02)



    def custom_distance(self, query, method='l1'):
        if self.norm:
            query = self.norm(query)
        if self.high_dim:
            key = self.rules_keys.unsqueeze(0).repeat(query.shape[0], 1, 1, 1, 1)
        else:
            key = self.rules_keys.unsqueeze(0).unsqueeze(0)
            key = key.repeat(query.shape[0], query.shape[2], 1, 1, 1).permute(0, 2, 1, 3, 4)
        
        query = query.unsqueeze(-2).repeat(1, 1, 1, self.num_rules, 1)
        
        if method == 'l1':
            distance = torch.abs(query - key)
        
        elif method == 'l2':
            distance = torch.square(query - key)
        
        elif method == 'cos':
            distance = 1 - torch.cosine_similarity(query, key, dim=-1).unsqueeze(-1)
        
        elif method == 'plv':
            phase_query = torch.angle(fft.fft(query, dim=-1))
            phase_key = torch.angle(fft.fft(key, dim=-1))
            phase_diff = torch.exp(1j * (phase_query - phase_key))
            distance = 1 - torch.abs(torch.mean(phase_diff, dim=-1)).unsqueeze(-1)
        
        elif method == 'coh':
            
            f_query = fft.fft(query, dim=-1)
            f_key = fft.fft(key, dim=-1)
            cross_spectrum = torch.mean(f_query * torch.conj(f_key), dim=-1)
            power_query = torch.mean(torch.abs(f_query) ** 2, dim=-1)
            power_key = torch.mean(torch.abs(f_key) ** 2, dim=-1)
            distance = 1 - torch.abs(cross_spectrum) ** 2 / (power_query * power_key)
            distance = distance.unsqueeze(-1)
            
        elif method == 'corr':
            mean_query = torch.mean(query, dim=-1, keepdim=True)
            mean_key = torch.mean(key, dim=-1, keepdim=True)
            std_dev_query = torch.std(query, dim=-1, keepdim=True)
            std_dev_key = torch.std(key, dim=-1, keepdim=True)
            normalized_query = (query - mean_query) / std_dev_query
            normalized_key = (key - mean_key) / std_dev_key
            distance = 1 - torch.mean(normalized_query * normalized_key, dim=-1)

        return distance

    
    def get_z_values(self, query_key_distance):
        # Calculate z values from query-key distance
        if self.high_dim:  
            prot=torch.div(query_key_distance, self.rules_widths)
            root=-torch.square(prot)*0.5
            z_values =root.mean(-1) # HTSK
            return z_values.permute(0, 2, 3, 1)
        else:
            prot=torch.div(query_key_distance.permute(0,2,1,3,4), self.rules_widths.unsqueeze(0).unsqueeze(0))
            root=-torch.square(prot)*0.5
            z_values =root.mean(-1) # HTSK
            return z_values

    def preprocessing_query(self, query):
        batch_size, seq_length, _ = query.size()
        # Project query
        query = self.query_proj(query) * self.scale # [B, seq_len, embed_dim]
        # query = query.reshape(batch_size, seq_length, self.num_heads, -1) # [B, seq_len, head, head_dim]
        # query = query.transpose(1, 2) # [B, head, seq_len, head_dim]
        query = rearrange(query, 'b s (h d) -> b h s d', h=self.num_heads, d=self.head_dim)
        return query

    def preprocessing_value(self, value):
        batch_size, seq_length, _ = value.size()
        value = self.value_proj(value) * self.scale # [B, seq_len, embed_dim]
        # rewrite it by rearrange value = value.reshape(batch_size, seq_length, self.num_heads,  -1, self.num_rules) # [B, seq_len, head, head_dim]
        # value = value.transpose(1, 2) # [B, head, seq_len, head_dim]
        value = rearrange(value, 'b s (h d r) -> b h s d r', h=self.num_heads, d=self.head_dim, r=self.num_rules)
        
        return value
    
    def get_atten(self, query, method='l1'): 
        query_key_distance = self.custom_distance(query, method = method)
        z_values = self.get_z_values(query_key_distance)
        attn_weights = self.softmax(z_values)
        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)
        return attn_weights
    
        
    def forward(self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False,
            return_meta: bool = False,
            
            ) -> Tuple[Tensor, Optional[Tensor]]:
        
        query = self.preprocessing_query(query)

        attn_weights = self.get_atten(query, method=self.method)
        value = self.preprocessing_value(value)

        if self.high_dim:
            batch_size, num_heads, seq_len_q, head_dim = query.shape
            value= value.reshape(batch_size, num_heads, seq_len_q, self.num_rules, head_dim)
            # value = rearrange(value, 'b h s d r -> b h s d r')
        else:
            value= value.permute(0,2,1,4,3)
     
        Fnn_output = (attn_weights.unsqueeze(-1) *value)
        output = Fnn_output.sum(dim=-2)
        output = output.transpose(1, 2).contiguous().view(output.size(0), -1, self.num_heads * self.head_dim)
        output = self.out_proj(output)

        if return_meta:
            return output, {
                            'attn': attn_weights,
                            'center': self.rules_keys, 
                            'width': self.rules_widths
                            }
        else:
            return output, None
        
# class FuzzyCNNAtt(FuzzyMultiheadAttention):
#     def __init__(self, 
#                  embed_dim, 
#                  num_heads, 
#                  num_rules,
#                  seq_len, 
#                  dropout=0.,
#                  kernel_t = 32,
#                  use_projection=True,
#                  norm=True, 
#                  high_dim=False,
#                  softmax = 'softmax',
#                  method='l1',
#                  batch_first = True,
#                  _qkv_same_embed_dim = True,
#                  ):
#         super().__init__()
#         self.__dict__.update(locals())

#         self.norm = None if not norm else NormalizeLayer(0, 1)

#         self.rules_kernel = nn.Parameter(torch.Tensor(self.num_rules, self.kernel_t))
#         self.rules_widths = nn.Parameter(torch.ones(self.num_rules, self.kernel_t))
#         self.query_proj = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#         self.softmax = lambda x: getattr(nn.functional, softmax)(x, dim=1)
#         self.dropout = nn.Dropout(dropout)
#         self.in_proj_weight=fakedata()
#         self.in_proj_bias=fakedata()

#         nn.init.normal_(self.rules_keys, mean=0.0, std=0.02)
    
#     def forward(self, value, return_meta=False):
        
#         query = self.query_proj(value) # [B, seq_len, embed_dim]

#         query = self.norm(query)
#         key = self.rules_keys # [num_rules, kernel_t]

#         # sliding window to get the distance
#         query = query.unfold(1, self.kernel_t, 1) # [B, seq_len-kernel_t+1, kernel_t, embed_dim]
#         query = query.permute(0, 2, 1, 3) # [B, kernel_t, seq_len-kernel_t+1, embed_dim]
#         query = query.unsqueeze(1) # [B, 1, kernel_t, seq_len-kernel_t+1, embed_dim]
#         key = key.unsqueeze(0).unsqueeze(0) # [1, 1, num_rules, kernel_t, embed_dim]
#         query_key_distance = torch.abs(query - key) # [B, num_rules, kernel_t, seq_len-kernel_t+1, embed_dim]

#         # average the kernel_t dimension
#         query_key_distance = query_key_distance.mean(dim=2) # [B, num_rules, seq_len-kernel_t+1, embed_dim]
#         z_values = self.get_z_values(query_key_distance)
#         attn_weights = self.softmax(z_values)
#         if self.dropout is not None:
#             attn_weights = self.dropout(attn_weights)
        



        

    # def get_z_values(self, query_key_distance):
    #     # query_key_distance: [B, num_rules, seq_len-kernel_t+1, embed_dim]
    #     # Calculate z values from query-key distance in num_rules dimension and apply softmax

    #     prot=torch.div(query_key_distance, self.rules_widths.unsqueeze(0).unsqueeze(0))
    #     root=-torch.square(prot)*0.5
    #     z_values =root.mean(-1) # HTSK
    #     return z_values.permute(0, 2, 3, 1) # [B, seq_len-kernel_t+1, embed_dim, num_rules]
        





class FuzzyDualAttention(nn.Module):
    def __init__(self, pretrained = True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),  
                seq_len = 300,
                embed_dim = 64,
                num_heads = 'auto', # auto or int
                num_rules = 10,
                dropout = 0.1,
                softmax = 'softmax',
                use_projection=True,
                norm=True,
                methods=['l1', 'l1'],
                layer_sort = ['s', 'e'], # the last dimension of the input tensor (B, seq_len, embed_dim)
            ):
        super().__init__()
        self.__dict__.update(locals())


        for ilayer, layer_type in enumerate(layer_sort):

            if layer_type == 's':
                fea_size = (embed_dim, seq_len)
            elif layer_type == 'e':
                fea_size = (seq_len, embed_dim)

            if num_heads == 'auto':
                esti_num_heads = 8
                while fea_size[1] % esti_num_heads != 0:
                    esti_num_heads -= 1
                print(f'Estimated num_heads for {layer_type}: {esti_num_heads}')
                num_head = esti_num_heads
            elif type(num_heads) == list:
                num_head = num_heads[ilayer]
            else:
                raise NotImplementedError

            setattr(self, f'feature_extractor_{ilayer}', FuzzyMultiheadAttention(
                embed_dim = fea_size[1],
                num_heads = num_head,
                num_rules = num_rules,
                seq_len = fea_size[0],
                dropout = dropout,
                use_projection = use_projection,
                norm = norm,
                softmax = softmax,
                method = methods[ilayer],
            ))

    def forward(self, x, return_meta=False):

        '''
        X: [B, seq_len, embed_dim]
        '''
        metas = {}
        for ilayer, layer_type in enumerate(self.layer_sort):
            
            if layer_type == 's':
                x = x.permute(0, 2, 1)
            
            # print(f'Layer {ilayer} - {layer_type}: {x.shape}')
            x, meta = getattr(self, f'feature_extractor_{ilayer}')(x, x, x, return_meta=return_meta)
            metas[f'meta_{ilayer}'] = meta
            
            if layer_type == 's':
                x = x.permute(0, 2, 1)
        
        # if len(self.layer_sort) % 2 != 0:
        #     x = x.permute(0, 2, 1)  
        
        if return_meta:
            return x, metas
        else:
            return x, None