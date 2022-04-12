import os
import argparse
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path


from models.models import MAE, models
from utils.torch_utils import select_device, time_synchronized
from utils.general import set_logging, check_file, increment_path, check_img_size
from utils.datasets import create_dataloader
from utils.plots import patch2img, plot_images


def patchify(imgs, patch_size=32):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x


def compute_pp(imgs, pred, mask, norm_pix_loss=True):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove,
    """
    target = patchify(imgs)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5

    pp = abs(pred - target)
    pp = pp.mean(dim=-1)  # [N, L], mean loss per patch

    pp = (pp * mask).sum() / mask.sum()  # mean loss on removed patches
    return pp


def test(data,
         weights=None,
         batch_size=16,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_conf=False,
         plots=True,
         log_imgs=0):  # number of logged images

    training = model is not None
    seen = 0
    if training:
        device = next(model.parameters()).device
    else:
        set_logging()
        device = select_device(opt.device, batch_size=batch_size)
        save_txt = opt.save_txt  # save *.txt labels

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = models(opt.cfg).to(device)

        # load model
        try:
            ckpt = torch.load(weights[0], map_location=device)  # load checkpoint
            ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(ckpt['model'], strict=False)
        except:
            print('Error!')
        imgsz = check_img_size(imgsz, s=64)  # check img_size

    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    model.eval()

    # Dataloader
    if not training:
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        path = data['test'] if opt.task == 'test' else data['val']  # path to val/test images
        dataloader = create_dataloader(path, imgsz, batch_size, 64, opt, pad=0.5, rect=True)[0]

    loss = torch.zeros(1, device=device)
    s = ('%20s' + '%12s' * 3) % ('Class', 'Images', 'Targets', 'mae')
    t0, t1 = 0, 0
    mloss = torch.zeros(1, device=device)
    mpp = torch.zeros(1, device=device)

    pbar = tqdm(dataloader)


    for batch_i, (imgs, paths) in enumerate(pbar):
        img = imgs.to(device, non_blocking=True)
        img = img.half() if half else img.float()
        img /= 255.0
        nb, _, height, width = img.shape

        with torch.no_grad():
            t = time_synchronized()
            loss, pred, mask = model(img)
            t0 += time_synchronized() - t

            mloss = (mloss * batch_i + loss.item()) / (batch_i + 1)
            pp = compute_pp(img, pred, mask)
            mpp = (mpp * batch_i + pp) / (batch_i + 1)

        for si, (p,m, path_) in enumerate(zip(pred, mask, paths)):
            seen += 1
            if plots:
                # pre_img = patch2img(pre_value, masked_indice, num_patches)
                im = img[si]
                f = save_dir / f'test_batch{batch_i}_sample{si}_preds.jpg'
                plot_images(pred, im, f)
        pbar.set_description()

        s = ('%10s ' * 2) % (mloss.tolist(), mpp.tolist())
        pbar.set_description(s)
    t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple



    return mloss, mpp, t

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_p6.pt', help='model.pt path(s)')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='*.data path')
    parser.add_argument('--batch-size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help="'val', 'test', 'study'")
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/coco.names', help='*.cfg path')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data = check_file(opt.data)  # check file
    print(opt)

    if opt.task in ['val', 'test']:  # run normally
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt,
             save_conf=opt.save_conf,
             )

    elif opt.task == 'study':  # run over a range of settings and save/plot
        for weights in ['yolor_p6.pt', 'yolor_w6.pt']:
            f = 'study_%s_%s.txt' % (Path(opt.data).stem, Path(weights).stem)  # filename to save to
            x = list(range(320, 800, 64))  # x axis
            y = []  # y axis
            for i in x:  # img-size
                print('\nRunning %s point %s...' % (f, i))
                r, _, t = test(opt.data, weights, opt.batch_size, i, opt.conf_thres, opt.iou_thres, opt.save_json)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        # utils.general.plot_study_txt(f, x)  # plot











