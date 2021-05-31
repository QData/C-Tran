 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from pdb import set_trace as stop
import math
from .transformer_layers import SelfAttnLayer
from .backbone import Backbone,InceptionBackbone
from .utils import custom_replace,weights_init
from .position_enc import PositionEmbeddingSine,positionalencoding2d


 
class CTranModelCub(nn.Module):
    def __init__(self,num_labels,use_lmt,pos_emb=False,layers=3,heads=4,dropout=0.1,int_loss=0,no_x_features=False):
        super(CTranModelCub, self).__init__()
        self.use_lmt = use_lmt
        embedding_dim = 2048

        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(2048,embedding_dim,(1,1))

        self.backbone = InceptionBackbone()

        
        self.no_x_features = no_x_features #for no image features
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1,-1).long()
        self.label_lt = torch.nn.Embedding(num_labels, embedding_dim, padding_idx=None)
        self.known_label_lt = torch.nn.Embedding(3, embedding_dim, padding_idx=0)

        # State Embeddings
        self.type_lt = torch.nn.Embedding(2, embedding_dim, padding_idx=None)

        self.use_pos_enc = pos_emb
        if self.use_pos_enc:
            # self.position_encoding = PositionEmbeddingSine(int(embedding_dim/2), normalize=True)
            self.position_encoding = positionalencoding2d(embedding_dim, 18, 18).unsqueeze(0)

        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(embedding_dim,heads,dropout) for _ in range(layers)])
        self.output_linear = torch.nn.Linear(embedding_dim,num_labels)

        self.LayerNorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.label_lt.apply(weights_init)
        self.type_lt.apply(weights_init)
        self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)


    def forward(self,images,mask):
        const_label_input = self.label_input.repeat(images.size(0),1).cuda()
        init_label_embeddings = self.label_lt(const_label_input)

        x = self.backbone(images)

        out_aux,features = x[0],x[1]
        out_aux,features = self.dropout(out_aux),self.dropout(features)
        
        if self.downsample:
            features = self.conv_downsample(features)
        if self.use_pos_enc:
            pos_encoding = self.position_encoding(features,torch.zeros(features.size(0),18,18, dtype=torch.bool).cuda())
            features = features + pos_encoding

        features = features.view(features.size(0),features.size(1),-1).permute(0,2,1) 

    
        if self.use_lmt:
            label_feat_vec = custom_replace(mask,0,1,2).long()
            known_label_embs = self.known_label_lt(label_feat_vec)
            init_label_embeddings += known_label_embs
        
        if self.no_x_features:
            embeddings = init_label_embeddings 
        else:
            embeddings = torch.cat((features,init_label_embeddings),1)

        embeddings = self.LayerNorm(embeddings)

        attns = []
        for layer in self.self_attn_layers:
            embeddings,attn = layer(embeddings,mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:,-init_label_embeddings.size(1):,:]

        output = self.output_linear(label_embeddings) 
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0),1,1).cuda()

        output = (output*diag_mask).sum(-1)

        return output,out_aux,attns


