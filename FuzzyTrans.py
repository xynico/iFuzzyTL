
import torch
import torch.nn as nn
from .base import BaseArch
from einops.layers.torch import Rearrange
import sys
from functools import partial
import numpy as np
from ..builder import ARCHS, build_backbone, build_head, build_arch
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F
from torch.nn import Linear, Dropout, Softmax
from .MultiheadFuzzyAttention import FuzzyMultiheadAttention, FuzzyDualAttention

@ARCHS.register_module()
class FuzzyTrans(BaseArch):
    def __init__(self, pretrained = True, model_ckpt = None, fixed = True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),  
                seq_len = 40,
                embed_dim = 33,
                num_rules=4,
                num_heads = 'auto',
                softmax = 'softmax',
                layer_sort = ['s', 'e'],
                dropout=0.3,
                use_projection=True,
                norm=True,
                num_classes=12,
                methods = ['l1', 'l1'],
                classifier_direction = 's2c', #s2c
                
            ):
        super().__init__()
        self.__dict__.update(locals())

        # feature extractor
        self.feature_extractor = FuzzyDualAttention(
            seq_len = seq_len,
            embed_dim = embed_dim,
            num_heads = num_heads,
            num_rules = num_rules,
            dropout = dropout,
            softmax = softmax,
            use_projection = use_projection,
            norm = norm,
            layer_sort = layer_sort,
            methods = methods,
        )
       
        # adapted average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        if self.classifier_direction == 's2c':
            self.classification_head = nn.Sequential(
                nn.Linear(seq_len, seq_len),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(seq_len, num_classes)
            )
        elif self.classifier_direction == 'e2c':
            self.classification_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes)
            )


        # --------------------------------------------------------------------------
        # CLS specifics
        self.cls_loss = nn.CrossEntropyLoss()
        # --------------------------------------------------------------------------
        self.initialize_weights()
        self.setup()
    
    def setup(self, stage=None):

        if self.model_ckpt is not None:
            ckpt = torch.load(self.model_ckpt, map_location='cpu')
            ckpt["state_dict"] = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
            ckpt["state_dict"] = {k: v for k, v in ckpt["state_dict"].items() if "encoder" in k}            
            # copy the weights from pre_train_model.encoder to self.encoder
            self.encoder.load_state_dict(ckpt["state_dict"], strict=False)
            if self.fixed == True:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
        

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, data, label, return_meta=False):
        # Encoder
        latent, meta = self.forward_encoder(data['seq'],return_meta)
        latent = self.classification_head(latent)
        pred = latent.squeeze(1)
        # print('pred:',pred.shape)
        # raise
        loss = self.cls_loss(pred, label)
        return loss, pred, meta
    
    def forward_encoder(self, x, return_meta=False):
        '''
        x: [B, TP, CH]
        '''
        x, meta = self.feature_extractor(x, return_meta=return_meta) # [B, seq_len, embed_dim]
        if self.classifier_direction == 's2c':
            x = x
        elif self.classifier_direction == 'e2c':
            x = Rearrange('b s e -> b e s')(x)
        x = self.avgpool(x).squeeze(-1)

        return x, meta
    
    def forward_train(self, x, label):
        loss, pred, meta = self.forward(x, label)
        # pack the output and losses
        return {'loss': loss}

    def forward_test(self, x, label=None):
        loss, pred, meta = self.forward(x, label, return_meta=True)
        # pred = pred.argmax(dim=1)
        pred = torch.argmax(pred, dim=1)
        # print('pred:',pred.shape, pred.min(), pred.max())
        return {'loss': loss, 'output': pred, 'meta_data': meta, 'label': label}
    