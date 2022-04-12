import argparse
import logging
import math
import os
import random
import time
import torch
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path

# from models.models import models
from models.models_mae import mae_maskedViT as models

from utils.datasets import create_dataloader
from utils.plots import plot_labels, plot_images
from utils.torch_utils import torch_distributed_zero_first, ModelEMA, select_device
from utils.general import init_seeds, check_img_size, check_dataset, check_file, increment_path, set_logging, strip_optimizer, get_latest_run
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import torch.optim as optim
from torch.cuda import amp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler

import test
import timm.optim.optim_factory as optim_factory

logger = logging.getLogger(__name__)


from torch.distributed.elastic.multiprocessing.errors import record

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

@record
def train(hyp, opt, device, tb_writer=None, wandb=None):
    logger.info(f'Hyperparameters {hyp}')
    save_dir ,epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    wdir = save_dir /'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir/ 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # plots = not opt.evolve
    plots = False
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    # with torch_distributed_zero_first(rank):
    #     check_dataset(data_dict)
    train_path = data_dict['train']
    test_path= data_dict['val']

    pretrained = weights.endswith('.pt')
    if pretrained:
        ckpt = torch.load(weights, map_location=device)
        model = models(opt.cfg).to(device)  # create
        state_dict = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(state_dict, strict=False)
        print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report
    else:
        model = models(opt.cfg).to(device)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay

    # pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    # for k, v in dict(model.named_parameters()).items():
    #     if '.bias' in k:
    #         pg2.append(v)  # biases
    #     elif 'm.weight' in k:
    #         pg1.append(v)  # apply weight_decay
    #     elif 'w.weight' in k:
    #         pg1.append(v)  # apply weight_decay
    #     else:
    #         pg0.append(v)  # all else
    #
    # if opt.adam:
    #     optimizer = optim.AdamW(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    # else:
    #     optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    #
    # optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    # optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    gs = 64
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]

    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    model_without_ddp = model
    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = optim_factory.add_weight_decay(model_without_ddp, opt.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=hyp['lr0'], betas=(0.9, 0.95))




    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp['lrf']) + hyp['lrf']  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # Results
        if ckpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(ckpt['training_results'])  # write results.txt

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if opt.resume:
            assert start_epoch > 0, '%s training to %g epochs is finished, nothing to resume.' % (weights, epochs)
        if epochs < start_epoch:
            logger.info('%s has been trained for %g epochs. Fine-tuning for %g additional epochs.' %
                        (weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, state_dict



    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, hyp,
                                            augment=True, cache=opt.cache_images, rect=opt.rect,
                                            rank=rank, world_size=opt.world_size, workers=opt.workers)
    nb = len(dataloader)  # number of batches

    # Process 0
    if rank in [-1, 0]:
        ema.updates = start_epoch * nb // accumulate  # set EMA updates
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,
                                       cache=opt.cache_images and not opt.notest, rect=True,
                                       rank=-1, world_size=opt.world_size, workers=opt.workers)[0]  # testloader

    model.hyp = hyp
    model.gr = 1.0

    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    results = (0)
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)

    # Logging
    if wandb and wandb.run is None:
        opt.hyp = hyp  # add hyperparameters
        wandb_run = wandb.init(config=opt, resume="allow",
                               project='MAE' if opt.project == 'runs/train' else Path(opt.project).stem,
                               name=save_dir.stem,
                               id=ckpt.get('wandb_id') if 'ckpt' in locals() else None)

    logger.info('Image sizes %g train, %g test\n'
                'using %g dataloader worker\nLogging results to %s\n'
                'Starting training for %g epochs' % (imgsz, imgsz_test, dataloader.num_workers, save_dir, epochs))
    torch.save(model, wdir/'init.pt')

    for epoch in range(start_epoch, epochs):
        model.train()

        mloss = torch.zeros(1, device=device)  # mean losses
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)

        logger.info(('\n' + '%10s' * 3) % ('Epoch', 'gpu_mem', 'mse'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)
        optimizer.zero_grad()
        for i, (imgs, paths) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j,x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # if opt.mutil_scale:
            #     sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
            #     sf = sz / max(imgs.shape[2:])
            #     if sf != 1:
            #         ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
            #         imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            with amp.autocast(enabled=cuda):
                loss, pred, mask = model(imgs)
                # loss = loss.item()
                # if rank != -1:
                #     losses *= opt.world_size

            scaler.scale(loss).backward()

            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            if rank in [-1, 0]:
                loss_item = loss.item()
                mloss = (mloss * i + loss_item) / (i + 1)
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                s = ('%10s ' * 2 + ' %20s' * 1) % ('%g / %g' % (epoch, epochs-1), mem, mloss)
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}_sample{i}.jpg'  # filename
                    plot_images(imgs, imgs, fname=f)

                elif plots and ni == 3 and wandb:
                    wandb.log({"Mosaics": [wandb.Image(str(x), caption=x.name) for x in save_dir.glob('train*.jpg')]})

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for tensorboard
        scheduler.step()

        if rank in [-1, 0]:
            if ema:
                ema.update_attr(model)
            final_epoch = epoch + 1 == epochs
            if not opt.notest or final_epoch:
                if epoch >= 0:
                    results, maps, times = test.test(opt.data,
                                                     batch_size=batch_size * 2,
                                                     imgsz=imgsz_test,
                                                     model=ema.ema.module if hasattr(ema.ema, 'module') else ema.ema,
                                                     dataloader=testloader,
                                                     save_dir=save_dir,
                                                     plots=plots and final_epoch,
                                                     log_imgs=opt.log_imgs if wandb else 0)

            # Write
            with open(results_file, 'a') as f:
                f.write(s + '%10.4g' * 1 % results + '\n')  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp %s gs://%s/results/results%s.txt' % (results_file, opt.bucket, opt.name))

            # Log
            tags = ['train/loss',   # train loss
                    # 'metrics/precision',
                    'val/loss',   # val loss
                    'x/lr0', 'x/lr1', 'x/lr2']  # params
            # for x, tag in zip(list(mloss[:-1]) + list(results) + lr, tags):
            for x, tag in zip([mloss] + [results] + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)  # tensorboard
                if wandb:
                    wandb.log({tag: x})  # W&B

            # Update best mAP
            fi = results

            if fi > best_fitness:
                best_fitness = fi

            # Save model
            save = (not opt.nosave) or (final_epoch and not opt.evolve)
            if save:
                with open(results_file, 'r') as f:
                    ckpt = {'epoch':epoch,
                            'best_fitness': best_fitness,
                            'training_results':f.read(),
                            'model':ema.ema.module.state_dict() if hasattr(ema, 'module') else ema.ema.state_dict(),
                            'optimizer':None if final_epoch else optimizer.state_dict(),
                            'wandb_id':wandb_run.id if wandb else None
                            }
                torch.save(ckpt, last)

                if best_fitness == fi:
                    torch.save(ckpt, best)

                if epoch == 0:
                    torch.save(ckpt, wdir / 'epoch_{}:03d.pt'.format(epoch))
                del ckpt

    if rank in [-1, 0]:
        n = opt.name if opt.name.isnumeric else ''
        fresults, flast, fbest = save_dir/ f'results{n}.txt', wdir/f'last{n}.pt', wdir/f'best{n}.pt'

        for f1, f2 in zip([wdir/'last.pt', wdir/'best.pt', results_file], [flast, fbest, fresults]):
            if f1.exists():
                os.rename(f1, f2)
                if str(f2).endswith('.pt'):
                    strip_optimizer(f2)
                    os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket else None
        if plots:
            if wandb:
                wandb.log({"Results":[wandb.Image(str(save_dir/ x), caption=x) for x in ['results.png', 'precesion-recall_curve.png']]})
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    else:
        dist.destroy_process_group()

    wandb.run.finish() if wandb and wandb.sun else None
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='cfg/mae.cfg', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/data.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.1280.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--sync_bn', action='store_true', help='rectangular training')
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--img_size', nargs='*', type=int, default=[1280, 1280], help='[train, test] image size')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--log-imgs', type=int, default=16, help='number of images for W&B logging, max 100')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

    opt = parser.parse_args()

    # set DDP variables
    opt.total_batch_size = opt.batch_size
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK'in os.environ else -1
    set_logging(opt.global_rank)

    if opt.resume:
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert os.path.isfile(ckpt)
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.FullLoader))
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        logger.info('Resuming training from %s') % ckpt
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
        opt.name = 'evole' if opt.evolve else opt.name
        opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)

    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    # Train
    logger.info(opt)
    if not opt.evolve:
        tb_writer = None  # init loggers
        if opt.global_rank in [-1, 0]:
            logger.info(f'Start Tensorboard with "tensorboard --logdir {opt.project}", view at http://localhost:6006/')
            tb_writer = SummaryWriter(opt.save_dir)  # Tensorboard
        train(hyp, opt, device, tb_writer, wandb)










