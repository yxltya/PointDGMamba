import os, logging, csv, numpy as np
#import os, logging, csv, numpy as np, wandb
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist
from torch.utils.tensorboard import SummaryWriter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, load_checkpoint_inv, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
from openpoints.dataset import build_dataloader_from_cfg
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
# from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.models.layers import furthest_point_sample, fps
from thop import profile
import pdb,wandb
import thop,time

def get_features_by_keys(input_features_dim, data):
    if input_features_dim == 3:
        features = data['pos']
    elif input_features_dim == 4:
        features = torch.cat(
            (data['pos'], data['heights']), dim=-1)
        raise NotImplementedError("error")
    return features.transpose(1, 2).contiguous()


def write_to_csv(oa, macc, accs, best_epoch, cfg, write_header=True):
    accs_table = [f'{item:.2f}' for item in accs]
    header = ['method', 'OA', 'mAcc'] + \
        cfg.classes + ['best_epoch', 'log_path', 'wandb link']
    data = [cfg.exp_name, f'{oa:.3f}', f'{macc:.2f}'] + accs_table + [
        str(best_epoch), cfg.run_dir, wandb.run.get_url() if cfg.wandb.use_wandb else '-']
    with open(cfg.csv_path, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(data)
        f.close()


def print_cls_results(oa, macc, accs, epoch, cfg):
    s = f'\nClasses\tAcc\n'
    for name, acc_tmp in zip(cfg.classes, accs):
        s += '{:10}: {:3.2f}%\n'.format(name, acc_tmp)
    s += f'E@{epoch}\tOA: {oa:3.2f}\tmAcc: {macc:3.2f}\n'
    logging.info(s)


def main(gpu, cfg, profile=False):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()
    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    if cfg.rank == 0 :
        Wandb.launch(cfg, cfg.wandb.use_wandb)
        writer = SummaryWriter(log_dir=cfg.run_dir)
    else:
        writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    
    torch.backends.cudnn.enabled = True
    logging.info(cfg)

    if not cfg.model.get('criterion_args', False):
        cfg.model.criterion_args = cfg.criterion_args
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))
    # criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    # pdb.set_trace()
    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logging.info('Using Synchronized BatchNorm ...')
    
    # pdb.set_trace()
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        print(f"using gpu {gpu}")
        # model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        model = nn.DataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        print(f"using gpu {gpu} for model")
        logging.info('Using Distributed Data parallel ...')
    
    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
    scheduler = build_scheduler_from_cfg(cfg, optimizer)

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
                                           )
    logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    test_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='test',
                                            distributed=cfg.distributed
                                            )
    test_loader1 = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                            cfg.testdataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='test',
                                            distributed=cfg.distributed
                                            )
    
    num_classes = val_loader.dataset.num_classes if hasattr(
        val_loader.dataset, 'num_classes') else None
    num_points = val_loader.dataset.num_points if hasattr(
        val_loader.dataset, 'num_points') else None
    # pdb.set_trace()
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    logging.info(f"number of classes of the dataset: {num_classes}, "
                 f"number of points sampled from dataset: {num_points}, "
                 f"number of points as model input: {cfg.num_points}")
    cfg.classes = cfg.get('classes', None) or val_loader.dataset.classes if hasattr(
        val_loader.dataset, 'classes') else None or np.range(num_classes)
    # pdb.set_trace()
    validate_fn = eval(cfg.get('val_fn', 'validate'))

    # optionally resume from a checkpoint
    if cfg.pretrained_path is not None:
        if cfg.mode == 'testDG':
            # test mode
            epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
            # pdb.set_trace()
            macc, oa, accs, cm = validate_fn(model, test_loader1, cfg)
            print_cls_results(oa, macc, accs, epoch, cfg)
            return True
        elif cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler,
                              pretrained_path=cfg.pretrained_path)
            macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
            print_cls_results(oa, macc, accs, cfg.start_epoch, cfg)
        else:
            if cfg.mode == 'test':
                # test mode
                epoch, best_val = load_checkpoint(model, pretrained_path=cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, test_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'val':
                # validation mode
                epoch, best_val = load_checkpoint(model, cfg.pretrained_path)
                macc, oa, accs, cm = validate_fn(model, val_loader, cfg)
                print_cls_results(oa, macc, accs, epoch, cfg)
                return True
            elif cfg.mode == 'finetune':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint(model.encoder, cfg.pretrained_path)
            elif cfg.mode == 'finetune_encoder_inv':
                # finetune the whole model
                logging.info(f'Finetuning from {cfg.pretrained_path}')
                load_checkpoint_inv(model.encoder, cfg.pretrained_path)
    else:
        logging.info('Training from scratch')
    
    

    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )

    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    # ===> start training
    val_macc, val_oa, val_accs, best_val, macc_when_best, best_epoch,val_oa1,best_oa1, best_epochontest1= 0., 0., [], 0., 0., 0 , 0 , 0 ,0
    model.zero_grad()
    
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):

        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):
            train_loader.dataset.epoch = epoch - 1
        
        train_loss, train_macc, train_oa, _, _ = train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg)
        # pdb.set_trace()
        is_best = False
        is_best_ontest1 = False

        if epoch % cfg.val_freq == 0:
            val_macc, val_oa, val_accs, val_cm = validate_fn(model, val_loader, cfg)
            
            '''if is_best:
                best_val = val_oa
                macc_when_best = val_macc
                best_epoch = epoch
                logging.info(f'Find a better ckpt @E{epoch}')'''
            print_cls_results(val_oa, val_macc, val_accs, epoch, cfg)
            #if epoch % 1 == 0:
            logging.info(f"Testing on @E{cfg.testdataset.common.NAME}")
            val_macc1, val_oa1, val_accs1, val_cm1= validate_fn(model, test_loader1, cfg)
            '''is_best = val_oa1 > best_val1
            if is_best:
            best_val1 = val_oa1
            macc_when_best1 = val_macc1
            best_epoch = epoch'''
            # logging.info(f'Find a better ckpt @E{epoch}')
            print_cls_results(val_oa1, val_macc1, val_accs1, epoch, cfg)

            is_best = val_oa1 > best_oa1
            is_best_ontest1 = val_oa1 > best_oa1
            
            if is_best_ontest1:
                best_oa1 = val_oa1
                best_epochontest1 = epoch
                logging.info(f'Find a better ckpt on {cfg.testdataset.common.NAME}')

        lr = optimizer.param_groups[0]['lr']
        logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train_oa {train_oa:.2f}, val_oa {val_oa:.2f}, best_oa1 {best_oa1:.2f}\n\n')
        if writer is not None:
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_oa', train_macc, epoch)
            writer.add_scalar('lr', lr, epoch)
            writer.add_scalar('val_oa', val_oa, epoch)
            writer.add_scalar('mAcc_when_best', macc_when_best, epoch)
            writer.add_scalar('best_val', best_val, epoch)
            writer.add_scalar('epoch', epoch, epoch)

        if cfg.sched_on_epoch:
            scheduler.step(epoch)
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best': best_oa1},
                            is_best=is_best
                            )
    # test the last epoch
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, epoch)
        writer.add_scalar('test_macc', test_macc, epoch)

    # test the best validataion model
    best_epoch, _ = load_checkpoint(model, pretrained_path=os.path.join(
        cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    test_macc, test_oa, test_accs, test_cm = validate(model, test_loader, cfg)
    test_macc1, test_oa1, test_accs1, test_cm1 = validate(model, test_loader1, cfg)
    if writer is not None:
        writer.add_scalar('test_oa', test_oa, best_epoch)
        writer.add_scalar('test_macc', test_macc, best_epoch)
        writer.add_scalar('test_oa', test_oa1, best_epoch)
        writer.add_scalar('test_macc', test_macc1, best_epoch)
    print_cls_results(test_oa, test_macc, test_accs, best_epoch, cfg)
    print_cls_results(test_oa1, test_macc1, test_accs1, best_epoch, cfg)

    if writer is not None:
        writer.close()
    dist.destroy_process_group()

def train_one_epoch(model, train_loader, optimizer, scheduler, epoch, cfg):
    loss_meter = AverageMeter()
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points

    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())
    num_iter = 0
    for idx, [data1,data2,is_for_contrastive_learning] in pbar:
        for key in data1.keys():
            data1[key] = data1[key].cuda(non_blocking=True)
            data2[key] = data2[key].cuda(non_blocking=True)
        num_iter += 1
        points1 = data1['x']
        target1 = data1['y']
        points2 = data2['x']
        target2 = data2['y']
        """ bebug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0])
        """
        num_curr_pts = points1.shape[1]
        if num_curr_pts > npoints:  # point resampling strategy
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
            if  points1.size(1) < point_all:
                point_all = points1.size(1)
            fps_idx1 = furthest_point_sample(points1[:, :, :3].contiguous(), point_all)
            fps_idx1 = fps_idx1[:, np.random.choice(point_all, npoints, False)]
            points1 = torch.gather(points1, 1, fps_idx1.unsqueeze(-1).long().expand(-1, -1, points1.shape[-1]))

            fps_idx2 = furthest_point_sample(points2[:, :, :3].contiguous(), point_all)
            fps_idx2 = fps_idx2[:, np.random.choice(point_all, npoints, False)]
            points2  = torch.gather(points2, 1, fps_idx2.unsqueeze(-1).long().expand(-1, -1, points2.shape[-1]))        

        data1['pos'] = points1[:, :, :3].contiguous()
        data1['x']   = points1[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()

        data2['pos'] = points2[:, :, :3].contiguous()
        data2['x']   = points2[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()

        logits, loss = model.get_logits_loss(data1, target1, data2, target2, is_for_contrastive_learning) if not hasattr(model, 'module') else model.module.get_logits_loss(data1, target1, data2, target2)

        loss.backward()

        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0
            optimizer.step()
            model.zero_grad()
            if not cfg.sched_on_epoch:
                scheduler.step(epoch)

        # update confusion matrix
        cm.update(logits.argmax(dim=1), torch.cat((target1,target2),dim=0))
        loss_meter.update(loss.item())
        if idx % cfg.print_freq == 0:
            pbar.set_description(f"Train Epoch [{epoch}/{cfg.epochs}] "
                                 f"Loss {loss_meter.val:.3f} Acc {cm.overall_accuray:.2f}")
    macc, overallacc, accs = cm.all_acc()
    return loss_meter.avg, macc, overallacc, accs, cm


@torch.no_grad()
def validate(model, val_loader, cfg):
    model.eval()  # set model to eval mode
    cm = ConfusionMatrix(num_classes=cfg.num_classes)
    npoints = cfg.num_points
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__())

    vis_features = []
    vis_labels = []

    for idx, [data1,data2,is_for_contrastive_learning] in pbar:
        for key in data1.keys():
            data1[key] = data1[key].cuda(non_blocking=True)
            data2[key] = data2[key].cuda(non_blocking=True)
        target1 = data1['y']
        points1 = data1['x']
        points2 = data2['x']
        target2 = data2['y']
        points1 = points1[:, :npoints]
        points2 = points2[:, :npoints]
        data1['pos'] = points1[:, :, :3].contiguous()
        data1['x'] = points1[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()

        data2['pos'] = points2[:, :, :3].contiguous()
        data2['x'] = points2[:, :, :cfg.model.in_channels].transpose(1, 2).contiguous()

        logits, feat1, feat2, _ = model(data1, data2,dataset = 'test')

        cm.update(logits.argmax(dim=1), torch.cat((target1,target2),dim=0))

    tp, count = cm.tp, cm.count
    if cfg.distributed:
        dist.all_reduce(tp), dist.all_reduce(count)
    macc, overallacc, accs = cm.cal_acc(tp, count)
    return macc, overallacc, accs, cm