import torch
import torch.nn as nn
import logging
from typing import List
from ..layers import create_linearblock
from ...utils import get_missing_parameters_message, get_unexpected_parameters_message
from ..build import MODELS, build_model_from_cfg
from ...loss import build_criterion_from_cfg
from ...utils import load_checkpoint
import pdb


def domain_aware_contrastive_loss(f, f_tr, label, domain_label, glaobal_feature=None):
    batch_size = f.shape[0]
    # 计算特征的相似度
    # Concat N originals and N stylized ones -> 2N samples
    f_list = torch.cat((f.view(batch_size, -1), f_tr.view(batch_size, -1)), dim=0).clone()
    # Normalize feature vectors
    f_list = f_list / torch.norm(f_list, p=2, dim=1).unsqueeze(1)
    # 2N X 2N similarity matrix
    sim_matrix = torch.matmul(f_list, torch.transpose(f_list, 0, 1))
    # exp
    sim_matrix = torch.exp(sim_matrix)
    # set diagonal elements to zero
    sim_matrix = sim_matrix.clone().fill_diagonal_(0)

    # 生成跨域的mask矩阵
    # Reshape labels N -> N * 1
    cls_label = label.contiguous().view(-1, 1)
    # N * N matrix ,比较 cls_label和其转置cls_label.T，返回一个布尔型张量，其中相同类别的样本对应位置为 True
    cls_mask = torch.eq(cls_label, cls_label.T).float().cuda()

    # 拉近
    pull_1 = torch.cat((cls_mask.clone().fill_diagonal_(0),cls_mask),dim=1)
    pull_2 = torch.cat((cls_mask,cls_mask.clone().fill_diagonal_(0)),dim=1)
    pull_mask = torch.cat((pull_1,pull_2),dim=0)

    # 推远
    push_1 = torch.cat((1-cls_mask,torch.zeros_like(cls_mask)),dim=1)
    push_2 = torch.cat((torch.zeros_like(cls_mask),1-cls_mask),dim=1)
    push_mask = torch.cat((push_1,push_2),dim=0)

    pull = (sim_matrix * pull_mask).sum()
    push = (sim_matrix * push_mask).sum()

    scores = pull / (pull + push)
    loss_contrast = - torch.log(scores)

    return loss_contrast


@MODELS.register_module()
class BaseCls(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 cls_args=None,
                 criterion_args=None,
                 **kwargs):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)

        if cls_args is not None:
            in_channels = self.encoder.out_channels if hasattr(self.encoder, 'out_channels') else cls_args.get('in_channels', None)
            cls_args.in_channels = in_channels
            self.prediction = build_model_from_cfg(cls_args)
        else:
            self.prediction = nn.Identity()
        self.criterion = build_criterion_from_cfg(criterion_args) if criterion_args is not None else None

    def forward(self, data1, data2,dataset = 'train'):
        feat1,feat2,contrast_feature = self.encoder.forward_cls_feat(data1, data2,dataset_divide = dataset)
        feat = torch.cat((feat1, feat2), dim=0)
        return self.prediction(feat), feat1, feat2, contrast_feature

    def get_logits_loss(self, data1, gt1, data2, gt2, is_for_contrastive_learning):
        logits, feat1, feat2, contrast_feature= self.forward(data1, data2,dataset='train')
        gt = torch.cat([gt1, gt2], dim=0)
        
        class_loss = self.criterion(logits, gt.long())
        
        return logits, class_loss
    
    def get_loss(self, pred, gt, inputs=None):
        return self.criterion(pred, gt.long())


@MODELS.register_module()
class DistillCls(BaseCls):
    def __init__(self,
                 encoder_args=None,
                 cls_args=None,
                 distill_args=None,
                 criterion_args=None,
                 **kwargs):
        super().__init__(encoder_args, cls_args, criterion_args)
        self.distill = encoder_args.get('distill', True)
        in_channels = self.encoder.distill_channels
        distill_args.distill_head_args.in_channels = in_channels
        self.dist_head = build_model_from_cfg(distill_args.distill_head_args)
        self.dist_model = build_model_from_cfg(distill_args).cuda()
        load_checkpoint(self.dist_model, distill_args.pretrained_path)
        self.dist_model.eval()

    def forward(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0['x']
        if self.distill and self.training:
            global_feat, distill_feature = self.encoder.forward_cls_feat(p0, f0)
            return self.prediction(global_feat), self.dist_head(distill_feature)
        else:
            global_feat = self.encoder.forward_cls_feat(p0, f0)
            return self.prediction(global_feat)

    def get_loss(self, pred, gt, inputs):
        return self.criterion(inputs, pred, gt.long(), self.dist_model)

    def get_logits_loss(self, data, gt):
        logits, dist_logits = self.forward(data)
        return logits, self.criterion(data, [logits, dist_logits], gt.long(), self.dist_model)


@MODELS.register_module()
class ClsHead(nn.Module):
    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 mlps: List[int]=[256],
                 norm_args: dict=None,
                 act_args: dict={'act': 'relu'},
                 dropout: float=0.5,
                 global_feat: str=None,
                 point_dim: int=2,
                 **kwargs
                 ):
        """A general classification head. supports global pooling and [CLS] token
        Args:
            num_classes (int): class num
            in_channels (int): input channels size
            mlps (List[int], optional): channel sizes for hidden layers. Defaults to [256].
            norm_args (dict, optional): dict of configuration for normalization. Defaults to None.
            act_args (_type_, optional): dict of configuration for activation. Defaults to {'act': 'relu'}.
            dropout (float, optional): use dropout when larger than 0. Defaults to 0.5.
            cls_feat (str, optional): preprocessing input features to obtain global feature.
                                      $\eg$ cls_feat='max,avg' means use the concatenateion of maxpooled and avgpooled features.
                                      Defaults to None, which means the input feautre is the global feature
        Returns:
            logits: (B, num_classes, N)
        """
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.global_feat = global_feat.split(',') if global_feat is not None else None
        self.point_dim = point_dim
        in_channels = len(self.global_feat) * in_channels if global_feat is not None else in_channels
        if mlps is not None:
            mlps = [in_channels] + mlps + [num_classes]
        else:
            mlps = [in_channels, num_classes]

        heads = []
        for i in range(len(mlps) - 2):
            heads.append(create_linearblock(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))
        heads.append(create_linearblock(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)


    def forward(self, end_points):
        if self.global_feat is not None:
            global_feats = []
            for preprocess in self.global_feat:
                if 'max' in preprocess:
                    global_feats.append(torch.max(end_points, dim=self.point_dim, keepdim=False)[0])
                elif preprocess in ['avg', 'mean']:
                    global_feats.append(torch.mean(end_points, dim=self.point_dim, keepdim=False))
            end_points = torch.cat(global_feats, dim=1)
        logits = self.head(end_points)
        return logits
