from ..build import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from .mamba_layer import MambaBlock
from .PCM_utils import MLP, serialization, _init_weights, index_points
from .PointMLP_layers import ConvBNReLU1D, LocalGrouper, PreExtraction, PreExtraction_Replace,\
    PosExtraction, get_activation
from typing import List
from ..layers import furthest_point_sample
import pdb

class MLP2(nn.Module):
    def __init__(self, dim=192):
        super(MLP2, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(self.dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.dim)

    def forward(self, x):
        batch_size, num_points, input_dim = x.shape
        x = x.reshape(-1, input_dim)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        x = x.view(batch_size, num_points, input_dim)
        return x
    

class MaskLearningModule(nn.Module):
    def __init__(self, num_features=1024, feature_dim=192, temperature=0.5, gamma=0.1):
        super(MaskLearningModule, self).__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.gamma = gamma
        
        self.prob_matrix = nn.Parameter(torch.rand(num_features, feature_dim) * 2 * gamma - gamma)
    
    def forward(self, x):
        batch_size, num_features, feature_dim = x.size()
        assert num_features == self.num_features and feature_dim == self.feature_dim
        
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.prob_matrix)))
        
        logits = self.prob_matrix + gumbel_noise 
        mask = F.softmax(logits / self.temperature, dim=-1)

        masked_x = x * mask
        
        return masked_x, mask


@MODELS.register_module()
class PointMambaEncoder(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=False, use_xyz=False, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[24, 24, 24, 24], k_strides=[1, 1, 1, 1], reducers=[2, 2, 2, 2],
                 mamba_blocks=[1, 1, 1, 1],
                 mamba_layers_orders='z',
                 use_order_prompt=False, prompt_num_per_order=1,
                 rms_norm=True, fused_add_norm=False, residual_in_fp32=False,
                 bimamba_type="none", drop_path_rate=0.1,
                 mamba_pos=False, pos_type='share', pos_proj_type="linear",
                 grid_size=0.02, combine_pos=False, block_residual=True,
                 use_windows=False, windows_size=1200,
                 cls_pooling="max",
                 **kwargs):
        super(PointMambaEncoder, self).__init__()

        assert len(pre_blocks) == len(k_neighbors) == len(reducers) ==\
               len(pos_blocks) == len(dim_expansion) == len(mamba_blocks), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."

        # local windows for mamba, this is only used in large point-cloud, such as 100k+ points (S3DIS)
        self.use_windows = use_windows
        self.windows_size = windows_size
        if not isinstance(self.windows_size, list):
            self.windows_size = [windows_size] * len(mamba_blocks)

        self.combine_pos = combine_pos
        self.local_rank = None
        self.block_residual = block_residual
        
        self.stages = len(pre_blocks)
        self.embedding = ConvBNReLU1D(in_channels, embed_dim, bias=bias, activation=activation)

        # assign serialization order for per mamba layer
        if not isinstance(mamba_layers_orders, list):
            mamba_layers_orders = [mamba_layers_orders for i in range(sum(mamba_blocks))]
        else:
            assert len(mamba_layers_orders) == sum(mamba_blocks)
        self.mamba_layers_orders = mamba_layers_orders

        # indicate the current sequence order
        self.order = 'original'

        # order prompts
        self.use_order_prompt = use_order_prompt
        self.promot_num_per_order = prompt_num_per_order

        if use_order_prompt:
            # learnable embeddings for per order, channel is 384
            unique_order = list(set(mamba_layers_orders))
            overall_prompt_nums = len(unique_order) * prompt_num_per_order
            self.order_prompt = nn.Embedding(overall_prompt_nums, 384)
            order2idx = {order: [i * prompt_num_per_order, (i + 1) * prompt_num_per_order]
                         for i, order in enumerate(unique_order)}
            self.per_layer_prompt_indexe = []
            for order in mamba_layers_orders:
                self.per_layer_prompt_indexe.append(order2idx[order])

        # point positional embedding
        self.pos_type = pos_type
        self.mamba_pos = mamba_pos
        if pos_proj_type == "linear":
            self.pos_proj_type = nn.Linear
        else:
            self.pos_proj_type = MLP
        if self.mamba_pos:
            if self.pos_type == "share":
                if sum(dim_expansion) == len(dim_expansion):
                    # if the channel size of all stages is equal, only init one proj function
                    self.pos_proj = self.pos_proj_type(3, embed_dim * dim_expansion[0], bias=False)
                    self.block_pos_share = True
                else:
                    # init proj functions for per stage
                    self.pos_proj = nn.ModuleList()
                    cur_dim = embed_dim
                    for ratio in dim_expansion:
                        cur_dim = ratio * cur_dim
                        self.pos_proj.append(self.pos_proj_type(3, cur_dim, bias=False))
                    self.block_pos_share = False
            else:
                # init proj functions for per mamba layer, will add at below
                self.pos_proj = nn.ModuleList()

        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.mamba_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        self.residual_proj_blocks_list = nn.ModuleList()

        last_layer_channel = embed_dim
        for ratio in dim_expansion:
            last_layer_channel *= ratio

        if use_order_prompt:
            self.order_prompt_proj = nn.ModuleList()
    
        last_channel = embed_dim
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm, eps=1e-5,
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(mamba_blocks))]
        inter_dpr = [0.0] + dpr
        mamba_layer_idx = 0

        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            mamba_block_num = mamba_blocks[i]
            kneighbor = k_neighbors[i]
            kstride = k_strides[i]
            reduce = reducers[i]
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, reduce, kneighbor, use_xyz, normalize,
                                         k_stride=kstride)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            if pre_block_num == 0:
                # only max pooling
                pre_block_module = PreExtraction_Replace(last_channel, out_channel, pre_block_num,
                                                         groups=groups,
                                                         res_expansion=res_expansion,
                                                         bias=bias, activation=activation, use_xyz=use_xyz)
            else:
                pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num,
                                                 groups=groups,
                                                 res_expansion=res_expansion,
                                                 bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)

            # append mamba block
            if last_channel == out_channel or i == 0 or not self.block_residual:
                self.residual_proj_blocks_list.append(nn.Identity())
            else:
                self.residual_proj_blocks_list.append(nn.Linear(last_channel, out_channel, bias=False))
            mamba_block = nn.Sequential()
            if use_order_prompt:
                self.order_prompt_proj.append(nn.Linear(384, out_channel, bias=False))
                
            for n_mamba in range(mamba_block_num):
                mamba_block_module = MambaBlock(dim=out_channel, layer_idx=mamba_layer_idx,
                                                bimamba_type=bimamba_type,
                                                norm_cls=norm_cls, fused_add_norm=fused_add_norm,
                                                residual_in_fp32=residual_in_fp32,
                                                drop_path=inter_dpr[mamba_layer_idx])
                mamba_block.append(mamba_block_module)
                if self.mamba_pos and self.pos_type != "share":
                    self.pos_proj.append(self.pos_proj_type(3, out_channel, bias=False))
                mamba_layer_idx += 1
            self.mamba_blocks_list.append(mamba_block)

            # append pos_block_list
            if pos_block_num == 0:
                self.pos_blocks_list.append(nn.Identity())
            else:
                pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                                 res_expansion=res_expansion, bias=bias,
                                                 activation=activation)
                self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.out_channels = last_channel
        self.act = get_activation(activation)

        self.mamba_blocks_list.apply(
            partial(_init_weights, n_layer=mamba_layer_idx, )
        )

        self.grid_size = grid_size
        self.cls_pooling = cls_pooling
        
        self.global_prompt = nn.Parameter(torch.randn(1024, embed_dim))
        self.MLP_1 = MLP2(embed_dim)
        self.MLP_2 = MLP2(embed_dim)
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=1, bias=bias),
            nn.BatchNorm1d(1024),
            self.act
        )
        self.reduce_dim_conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1024*3, out_channels=1024, kernel_size=1, bias=bias),
            nn.BatchNorm1d(1024),
            self.act
        )
        self.cross_domain_scan  = MambaBlock(dim=embed_dim, layer_idx=mamba_layer_idx,
                                        bimamba_type=bimamba_type,
                                        norm_cls=norm_cls, 
                                        fused_add_norm=fused_add_norm,
                                        residual_in_fp32=residual_in_fp32,
                                        drop_path=inter_dpr[mamba_layer_idx],
                                        cross_domain_and_within_domain_scan=True)
        
        self.within_domain_scan = MambaBlock(dim=embed_dim, layer_idx=mamba_layer_idx,
                                        bimamba_type=bimamba_type,
                                        norm_cls=norm_cls, 
                                        fused_add_norm=fused_add_norm,
                                        residual_in_fp32=residual_in_fp32,
                                        drop_path=inter_dpr[mamba_layer_idx],
                                        cross_domain_and_within_domain_scan=True)

        tau = 0.6
        self.mask_learning_module = MaskLearningModule(num_features=1024, feature_dim=embed_dim)


    def forward(self, x1,x2, f0=None ,dataset_divide = 'train'):
        return self.forward_cls_feat(x1, x2,f0, dataset_divide = dataset_divide)

    def serialization_func(self, p, x, x_res, order, layers_outputs=[]):
        if order == self.order:
            return p, x, x_res
        else:
            p, x, x_res = serialization(p, x, x_res=x_res, order=order,
                                        layers_outputs=layers_outputs,
                                        grid_size=self.grid_size)
            self.order = order
            return p, x, x_res
        
    def forward_cls_feat_Channel_and_Spatial(self, p, x=None, dataset_divide = 'train'):
        self.order = "original"


        if isinstance(p, dict):
            p, x = p['pos'], p.get('x', None) 
        if x is None:
            x = p.transpose(1, 2).contiguous()
        else:
            if self.combine_pos:
                x = torch.cat([x, p.transpose(1, 2)], dim=1).contiguous()

        batch_size, _, _ = x.size()
        x = self.embedding(x)  # B,D,N
        x_res = None

        pos_proj_idx = 0
        mamba_layer_idx = 0
        for i in range(self.stages):
            # GAM forward
            p, x, x_res = self.local_grouper_list[i](p, x.permute(0, 2, 1), x_res)  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = x.permute(0, 2, 1).contiguous()
            if not self.block_residual:
                x_res = None
            x_res = self.residual_proj_blocks_list[i](x_res)

            # mamba forward
            
            for layer in self.mamba_blocks_list[i]:
                p, x, x_res = self.serialization_func(p, x, x_res, self.mamba_layers_orders[mamba_layer_idx])

                if self.use_windows:
                    p, x, x_res, n_windows = self.pre_split_windows(p, x, x_res, windows_size=self.windows_size[i])

                if self.mamba_pos:
                    if self.pos_type == 'share':
                        if self.block_pos_share:
                            x = x + self.pos_proj(p)
                        else:
                            x = x + self.pos_proj[i](p)
                    else:
                        x = x + self.pos_proj[pos_proj_idx](p)
                        pos_proj_idx += 1
                if self.use_order_prompt:
                    layer_order_prompt_indexes = self.per_layer_prompt_indexe[mamba_layer_idx]

                    layer_order_prompt = self.order_prompt.weight[
                                         layer_order_prompt_indexes[0]: layer_order_prompt_indexes[1]]
                    
                    layer_order_prompt = self.order_prompt_proj[i](layer_order_prompt)
                    layer_order_prompt = layer_order_prompt.unsqueeze(0).repeat(p.shape[0], 1, 1)
                    x = torch.cat([layer_order_prompt, x, layer_order_prompt], dim=1)
                    if x_res is not None:
                        x_res = torch.cat([layer_order_prompt, x_res, layer_order_prompt], dim=1)

                    x, x_res = layer(x, x_res) # out, residual


                    x = x[:, self.promot_num_per_order:-self.promot_num_per_order]
                    x_res = x_res[:, self.promot_num_per_order:-self.promot_num_per_order]
                else:
                    x, x_res = layer(x, x_res)
                
                if self.use_windows:
                    p, x, x_res = self.post_split_windows(p, x, x_res, n_windows)
                mamba_layer_idx += 1
            

            x = x.permute(0, 2, 1).contiguous()
            
            x = self.pos_blocks_list[i](x)  # [b,d,g]

        # Max Pooling 
        if self.cls_pooling == "max":
            x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        elif self.cls_pooling == "mean":
            x = x.mean(dim=-1, keepdim=False)
        else:
            x_max = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
            x_mean = x.mean(dim=-1, keepdim=False)
            x = x_max + x_mean
        return x
    
    def forward_cls_feat_Channel_and_Spatial(self, p1, p2,x=None, dataset_divide = 'train'):
        self.order = "original"


        if isinstance(p1, dict):
            p1, x1 = p1['pos'], p1.get('x', None)
        if x is None:
            x1 = p1.transpose(1, 2).contiguous()
        else:
            if self.combine_pos:
                x1 = torch.cat([x, p1.transpose(1, 2)], dim=1).contiguous()
        pdb.set_trace()
        batch_size1, _, _ = x1.size()
        x1 = self.embedding(x1)  # B,D,N 
        x_res1 = None

        if isinstance(p2, dict):
            p2, x2 = p2['pos'], p2.get('x', None)
        if x is None:
            x2 = p2.transpose(1, 2).contiguous()
        else:
            if self.combine_pos:
                x2 = torch.cat([x, p2.transpose(1, 2)], dim=1).contiguous()

        batch_size2, _, _ = x2.size()
        x2 = self.embedding(x2)  # B,D,N 
        x_res2 = None

        pos_proj_idx1 = 0
        pos_proj_idx2 = 0
        mamba_layer_idx = 0
        for i in range(self.stages):
            # GAM forward

            p1, x1, x_res1 = self.local_grouper_list[i](p1, x1.permute(0, 2, 1), x_res1)  # [b,g,3]  [b,g,k,d]
            x1 = self.pre_blocks_list[i](x1)  # [b,d,g] 

            x1 = x1.permute(0, 2, 1).contiguous()
            if not self.block_residual:
                x_res1 = None
            x_res1 = self.residual_proj_blocks_list[i](x_res1)

            p2, x2, x_res2 = self.local_grouper_list[i](p2, x2.permute(0, 2, 1), x_res2)  # [b,g,3]  [b,g,k,d]
            x2 = self.pre_blocks_list[i](x2)  # [b,d,g] 

            x2 = x2.permute(0, 2, 1).contiguous()
            if not self.block_residual:
                x_res2 = None
            x_res2 = self.residual_proj_blocks_list[i](x_res2)

            # mamba forward
            
            for layer in self.mamba_blocks_list[i]:
                p1, x1, x_res1 = self.serialization_func(p1, x1, x_res1, self.mamba_layers_orders[mamba_layer_idx])
                p2, x2, x_res2 = self.serialization_func(p2, x2, x_res2, self.mamba_layers_orders[mamba_layer_idx])
                # mamba forward

                if self.use_windows:
                    p1, x1, x_res1, n_windows = self.pre_split_windows(p1, x1, x_res1, windows_size=self.windows_size[i])
                if self.use_windows:
                    p2, x2, x_res2, n_windows = self.pre_split_windows(p2, x2, x_res2, windows_size=self.windows_size[i])

                if self.mamba_pos:
                    if self.pos_type == 'share':
                        if self.block_pos_share:
                            x1 = x1 + self.pos_proj(p1)
                        else:
                            x1 = x1 + self.pos_proj[i](p1) 
                    else:
                        x1 = x1 + self.pos_proj[pos_proj_idx1](p1)
                        pos_proj_idx1 += 1
                if self.mamba_pos:
                    if self.pos_type == 'share':
                        if self.block_pos_share:
                            x2 = x2 + self.pos_proj(p2)
                        else:
                            x2 = x2 + self.pos_proj[i](p2)
                    else:
                        x2 = x2 + self.pos_proj[pos_proj_idx2](p2)
                        pos_proj_idx2 += 1
                    
                if self.use_order_prompt:
                    layer_order_prompt_indexes = self.per_layer_prompt_indexe[mamba_layer_idx]

                    layer_order_prompt = self.order_prompt.weight[
                                         layer_order_prompt_indexes[0]: layer_order_prompt_indexes[1]]
                    
                    layer_order_prompt = self.order_prompt_proj[i](layer_order_prompt)
                    layer_order_prompt = layer_order_prompt.unsqueeze(0).repeat(p1.shape[0], 1, 1)
                    x1 = torch.cat([layer_order_prompt, x1, layer_order_prompt], dim=1)
                    x2 = torch.cat([layer_order_prompt, x2, layer_order_prompt], dim=1)
                    if x_res1 is not None:
                        x_res1 = torch.cat([layer_order_prompt, x_res1, layer_order_prompt], dim=1)
                    if x_res2 is not None:
                        x_res2 = torch.cat([layer_order_prompt, x_res2, layer_order_prompt], dim=1)

                    x1, x_res1 = layer(x1, x_res1)
                    x2, x_res2 = layer(x2, x_res2) 


                    x1 = x1[:, self.promot_num_per_order:-self.promot_num_per_order]
                    x_res1 = x_res1[:, self.promot_num_per_order:-self.promot_num_per_order]

                    x2 = x2[:, self.promot_num_per_order:-self.promot_num_per_order]
                    x_res2 = x_res2[:, self.promot_num_per_order:-self.promot_num_per_order]
                else:
                    x1, x_res1 = layer(x1, x_res1) 
                    x2, x_res2 = layer(x2, x_res2)
                
                if self.use_windows:
                    p1, x1, x_res1 = self.post_split_windows(p1, x1, x_res1, n_windows)
                    p2, x2, x_res2 = self.post_split_windows(p2, x2, x_res2, n_windows)
                mamba_layer_idx += 1
            mid = x1 * x2
            x1 = torch.concat([x1, mid,layer_order_prompt],dim = 1)
            x2 = torch.concat([x2, mid,layer_order_prompt],dim = 1)
            
            x1 = self.pos_blocks_list[i](x1)  # [b,d,g]
            x2 = self.pos_blocks_list[i](x2)  # [b,d,g]

        # Max Pooling 
        if self.cls_pooling == "max":
            x1 = F.adaptive_max_pool1d(x1, 1).squeeze(dim=-1)
        elif self.cls_pooling == "mean":
            x1 = x1.mean(dim=-1, keepdim=False)
        else:
            x_max1 = F.adaptive_max_pool1d(x1, 1).squeeze(dim=-1)
            x_mean1 = x1.mean(dim=-1, keepdim=False)
            x1 = x_max1 + x_mean1

        if self.cls_pooling == "max":
            x2 = F.adaptive_max_pool1d(x2, 1).squeeze(dim=-1)
        elif self.cls_pooling == "mean":
            x2 = x2.mean(dim=-1, keepdim=False)
        else:
            x_max2 = F.adaptive_max_pool1d(x2, 1).squeeze(dim=-1)
            x_mean2 = x2.mean(dim=-1, keepdim=False)
            x2 = x_max2 + x_mean2
        return x1, x2
    
    def forward_cls_feat(self, p1,p2, x=None, dataset_divide = 'train'):
        self.order = "original"

        if isinstance(p1, dict):
            p1, x1 = p1['pos'], p1.get('x', None)
        if x1 is None:
            x1 = p1.transpose(1, 2).contiguous()
        else:
            if self.combine_pos:
                x1 = torch.cat([x1, p1.transpose(1, 2)], dim=1).contiguous()
       
        x1 = self.embedding(x1)  # B,D,N
        x_res1 = None

        if isinstance(p2, dict):
            p2, x2 = p2['pos'], p2.get('x', None)  
        if x2 is None:
            x2 = p2.transpose(1, 2).contiguous()
        else:
            if self.combine_pos:
                x2 = torch.cat([x2, p2.transpose(1, 2)], dim=1).contiguous()

        x2 = self.embedding(x2)  # B,D,N 
        x_res2 = None

        pos_proj_idx1 = 0
        pos_proj_idx2 = 0
        mamba_layer_idx = 0
        contrast_feature = list()
        for i in range(self.stages):
            # GAM forward

            p1, x1, x_res1 = self.local_grouper_list[i](p1, x1.permute(0, 2, 1), x_res1)  # [b,g,3]  [b,g,k,d]
            x1 = self.pre_blocks_list[i](x1)  

            x1 = x1.permute(0, 2, 1).contiguous()
            if not self.block_residual:
                x_res1 = None
            x_res1 = self.residual_proj_blocks_list[i](x_res1)

            p2, x2, x_res2 = self.local_grouper_list[i](p2, x2.permute(0, 2, 1), x_res2)
            x2 = self.pre_blocks_list[i](x2) 

            x2 = x2.permute(0, 2, 1).contiguous()
            if not self.block_residual:
                x_res2 = None
            x_res2 = self.residual_proj_blocks_list[i](x_res2)

            # mamba forward
            
            for layer in self.mamba_blocks_list[i]:

                p1, x1, x_res1 = self.serialization_func(p1, x1, x_res1, self.mamba_layers_orders[mamba_layer_idx])

                p2, x2, x_res2 = self.serialization_func(p2, x2, x_res2, self.mamba_layers_orders[mamba_layer_idx])
                # mamba forward

                if self.use_windows:
                    p1, x1, x_res1, n_windows = self.pre_split_windows(p1, x1, x_res1, windows_size=self.windows_size[i])

                if self.use_windows:
                    p2, x2, x_res2, n_windows = self.pre_split_windows(p2, x2, x_res2, windows_size=self.windows_size[i])

                if self.mamba_pos:
                    if self.pos_type == 'share':
                        if self.block_pos_share:
                            x1 = x1 + self.pos_proj(p1)
                        else:
                            x1 = x1 + self.pos_proj[i](p1)
                    else:
                        x1 = x1 + self.pos_proj[pos_proj_idx1](p1)
                        pos_proj_idx1 += 1

                if self.mamba_pos:
                    if self.pos_type == 'share':
                        if self.block_pos_share:
                            x2 = x2 + self.pos_proj(p2)
                        else:
                            x2 = x2 + self.pos_proj[i](p2)
                    else:
                        x2 = x2 + self.pos_proj[pos_proj_idx2](p2)
                        pos_proj_idx2 += 1
                    
                if self.use_order_prompt:
                    layer_order_prompt_indexes = self.per_layer_prompt_indexe[mamba_layer_idx]

                    layer_order_prompt = self.order_prompt.weight[
                                         layer_order_prompt_indexes[0]: layer_order_prompt_indexes[1]]
                    
                    layer_order_prompt = self.order_prompt_proj[i](layer_order_prompt)
                    layer_order_prompt1 = layer_order_prompt.unsqueeze(0).repeat(p1.shape[0], 1, 1)
                    layer_order_prompt2 = layer_order_prompt.unsqueeze(0).repeat(p2.shape[0], 1, 1)
                    x1 = torch.cat([layer_order_prompt1, x1, layer_order_prompt1], dim=1)
                    x2 = torch.cat([layer_order_prompt2, x2, layer_order_prompt2], dim=1)
                    if x_res1 is not None:
                        x_res1 = torch.cat([layer_order_prompt1, x_res1, layer_order_prompt1], dim=1)
                    if x_res2 is not None:
                        x_res2 = torch.cat([layer_order_prompt2, x_res2, layer_order_prompt2], dim=1)

                    x1, x_res1 = layer(x1, x_res1)
                    x2, x_res2 = layer(x2, x_res2)


                    x1 = x1[:, self.promot_num_per_order:-self.promot_num_per_order]
                    x_res1 = x_res1[:, self.promot_num_per_order:-self.promot_num_per_order]

                    x2 = x2[:, self.promot_num_per_order:-self.promot_num_per_order]
                    x_res2 = x_res2[:, self.promot_num_per_order:-self.promot_num_per_order]
                else:
                    x1, x_res1 = layer(x1, x_res1)
                    x2, x_res2 = layer(x2, x_res2)
                
                if self.use_windows:
                    p1, x1, x_res1 = self.post_split_windows(p1, x1, x_res1, n_windows)
                    p2, x2, x_res2 = self.post_split_windows(p2, x2, x_res2, n_windows)
                
                if mamba_layer_idx == 0:
                    contrast_feature.append(x1)
                    contrast_feature.append(x2)

                    # Mask
                    x1, mask_1 = self.mask_learning_module(x1)
                    x2, mask_2 = self.mask_learning_module(x2)

                    # SCFA+DDS
                    data1 = self.MLP_1(x1)
                    data2 = self.MLP_2(x2)
                    mid = self.conv1d(data1 * data2)

                    global_prompt_ = self.global_prompt.unsqueeze(0).repeat(x1.shape[0],1,1)
                    contrast_feature.append(global_prompt_)

                    x1 = torch.concat([x1, mid, global_prompt_],dim = 1)
                    x2 = torch.concat([x2, mid, global_prompt_],dim = 1)

                    # cross domain scan, within domain scan
                    x1_, _ = self.within_domain_scan(x1,x1)
                    x2_, _ = self.within_domain_scan(x2,x2)
                    x1 = x1 + x1_
                    x2 = x2 + x2_

                    stride = mid.shape[1]
                    x1_temp = torch.zeros_like(x1)
                    x2_temp = torch.zeros_like(x2)

                    x1_temp[:, ::3, :]  = x1[:, 0:stride, :]
                    x1_temp[:, 1::3, :] = x1[:, stride:2*stride, :]
                    x1_temp[:, 2::3, :] = x1[:, 2*stride:, :]
                    x2_temp[:, ::3, :]  = x2[:, 0:stride, :]
                    x2_temp[:, 1::3, :] = x2[:, stride:2*stride, :]
                    x2_temp[:, 2::3, :] = x2[:, 2*stride:, :]
                    x1 = x1_temp
                    x2 = x2_temp
                    del x1_temp, x2_temp, stride

                    x1__, _ = self.cross_domain_scan(x1,x1)
                    x2__, _ = self.cross_domain_scan(x2,x2)
                    x1 = x1 + x1__
                    x2 = x2 + x2__

                    x1 = self.reduce_dim_conv1d(x1)
                    x2 = self.reduce_dim_conv1d(x2)

                mamba_layer_idx += 1

            x1 = x1.permute(0,2,1)
            x2 = x2.permute(0,2,1)

            x1 = self.pos_blocks_list[i](x1)  # [b,d,g]
            x2 = self.pos_blocks_list[i](x2) 

        # Max Pooling 
        if self.cls_pooling == "max":
            x1 = F.adaptive_max_pool1d(x1, 1).squeeze(dim=-1)
        elif self.cls_pooling == "mean":
            x1 = x1.mean(dim=-1, keepdim=False)
        else:
            x_max1 = F.adaptive_max_pool1d(x1, 1).squeeze(dim=-1)
            x_mean1 = x1.mean(dim=-1, keepdim=False)
            x1 = x_max1 + x_mean1

        if self.cls_pooling == "max":
            x2 = F.adaptive_max_pool1d(x2, 1).squeeze(dim=-1)
        elif self.cls_pooling == "mean":
            x2 = x2.mean(dim=-1, keepdim=False)
        else:
            x_max2 = F.adaptive_max_pool1d(x2, 1).squeeze(dim=-1)
            x_mean2 = x2.mean(dim=-1, keepdim=False)
            x2 = x_max2 + x_mean2


        return x1, x2, contrast_feature


    def forward_seg_feat(self, p, x=None):
        self.order = "original"
        if isinstance(p, dict):
            p, x = p['pos'], p.get('x', None)
        if x is None:
            x = p.transpose(1, 2).contiguous()
        else:
            if self.combine_pos:
                x = torch.cat([x, p.transpose(1, 2)], dim=1).contiguous()
        batch_size, _, _ = x.size()
        x = self.embedding(x)  # B,D,N

        p_list, x_list = [p], [x]

        x_res = None

        pos_proj_idx = 0
        mamba_layer_idx = 0
        for i in range(self.stages):
            # Give p[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            p, x, x_res = self.local_grouper_list[i](p, x.permute(0, 2, 1), x_res)  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]

            x = x.permute(0, 2, 1).contiguous()
            if not self.block_residual:
                x_res = None
            x_res = self.residual_proj_blocks_list[i](x_res)
            for layer in self.mamba_blocks_list[i]:
                p, x, x_res = self.serialization_func(p, x, x_res, self.mamba_layers_orders[mamba_layer_idx])
                if self.use_windows:
                    p, x, x_res, n_windows = self.pre_split_windows(
                        p, x, x_res, windows_size=self.windows_size[i])
                if self.mamba_pos:
                    if self.pos_type == 'share':
                        if self.block_pos_share:
                            x = x + self.pos_proj(p)
                        else:
                            x = x + self.pos_proj[i](p)
                    else:
                        x = x + self.pos_proj[pos_proj_idx](p)
                        pos_proj_idx += 1
                if self.use_order_prompt:
                    layer_order_prompt_indexes = self.per_layer_prompt_indexe[mamba_layer_idx]
                    layer_order_prompt = self.order_prompt.weight[
                                         layer_order_prompt_indexes[0]: layer_order_prompt_indexes[1]]
                    layer_order_prompt = self.order_prompt_proj[i](layer_order_prompt)
                    layer_order_prompt = layer_order_prompt.unsqueeze(0).repeat(p.shape[0], 1, 1)
                    x = torch.cat([layer_order_prompt, x, layer_order_prompt], dim=1)
                    if x_res is not None:
                        x_res = torch.cat([layer_order_prompt, x_res, layer_order_prompt], dim=1)
                    x, x_res = layer(x, x_res)
                    x = x[:, self.promot_num_per_order:-self.promot_num_per_order]
                    x_res = x_res[:, self.promot_num_per_order:-self.promot_num_per_order]
                else:
                    x, x_res = layer(x, x_res)
                if self.use_windows:
                    p, x, x_res = self.post_split_windows(p, x, x_res, n_windows)
                mamba_layer_idx += 1
            x = x.permute(0, 2, 1).contiguous()
            x = self.pos_blocks_list[i](x)  # [b,d,g]

            p_list.append(p)
            x_list.append(x)

        return p_list, x_list

    def pre_split_windows(self, p, x, x_res, windows_size=1024):
        # x (bs, n, c), p (bs, n, 3)
        bs, n, c = x.shape
        if n <= windows_size:
            return p, x, x_res, 1

        # fps sample
        n_sample = n // windows_size * windows_size
        fps_idx = furthest_point_sample(p, n_sample).long()  # [B, n_windows]
        fps_idx = torch.sort(fps_idx, dim=-1)[0]
        new_p = index_points(p, fps_idx)  # [B, n_windows, 3]
        new_x = index_points(x, fps_idx)
        if x_res is not None:
            new_x_res = index_points(x_res, fps_idx)
        else:
            new_x_res = None

        # split windows
        bs, n, c = new_x.shape
        n_splits = n // windows_size
        new_p = new_p.reshape(bs, n_splits, windows_size, -1).flatten(0, 1)
        new_x = new_x.reshape(bs, n_splits, windows_size, -1).flatten(0, 1)
        if new_x_res is not None:
            new_x_res = new_x_res.reshape(bs, n_splits, windows_size, -1).flatten(0, 1).contiguous()
        return new_p.contiguous(), new_x.contiguous(), new_x_res, n_splits

    def post_split_windows(self, p, x, x_res, n_windos):
        if n_windos == 1:
            return p, x, x_res
        bs_nw, window_size, c = x.shape
        bs = bs_nw // n_windos

        p = p.reshape(bs, n_windos, window_size, -1).flatten(1, 2)
        x = x.reshape(bs, n_windos, window_size, -1).flatten(1, 2)
        if x_res is not None:
            x_res = x_res.reshape(bs, n_windos, window_size, -1).flatten(1, 2)
            return p.contiguous(), x.contiguous(), x_res.contiguous()
        else:
            return p.contiguous(), x.contiguous(), x_res
