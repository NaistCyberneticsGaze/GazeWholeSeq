"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

End-to-end inference codes for 
3D human body mesh reconstruction from an image

python ./metro/tools/end2end_inference_bodymesh.py 
       --resume_checkpoint ./models/metro_release/metro_3dpw_state_dict.bin
       --image_file_or_path ./samples/human-body
"""

from __future__ import absolute_import, division, print_function
import argparse
import os
import os.path as op
import code
import copy
import time
import random
import datetime
import torch
import torchvision.models as models
from torchvision.utils import make_grid
import numpy as np
import cv2
from torch.utils.data import DataLoader, random_split
from models.bert.modeling_bert import BertConfig
from models.bert.modeling_metro import METRO_Body_Network as METRO_Network
from models.bert.modeling_metro import METRO
from models.bert.modeling_gabert import GAZEFROMBODY
from models.smpl._smpl import SMPL, Mesh
from models.hrnet.hrnet_cls_net_featmaps import get_cls_net
from models.hrnet.config import config as hrnet_config
from models.hrnet.config import update_config as hrnet_update_config
from models.dataloader.gafa_loader import create_gafa_dataset
from models.utils.logger import setup_logger
from models.utils.metric_logger import AverageMeter
from models.utils.miscellaneous import mkdir, set_seed

from PIL import Image
from torchvision import transforms

transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])


def save_checkpoint(model, args, epoch, iteration, num_trial=10):
    checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(
        epoch, iteration))

    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    for i in range(num_trial):
        try:
            torch.save(model_to_save, op.join(checkpoint_dir, 'model.bin'))
            torch.save(model_to_save.state_dict(), op.join(checkpoint_dir, 'state_dict.bin'))
            torch.save(args, op.join(checkpoint_dir, 'training_args.bin'))
            logger.info("Save checkpoint to {}".format(checkpoint_dir))
            break
        except:
            pass
    else:
        logger.info("Failed to save checkpoint after {} trails.".format(num_trial))
    return checkpoint_dir




class CosLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        l2 = torch.linalg.norm(outputs, ord=2, axis=1)
        outputs = outputs/l2[:,None]
        outputs = outputs.reshape(-1, outputs.shape[-1])
        l2 = torch.linalg.norm(targets, ord=2, axis=1)
        targets = targets/l2[:,None]
        targets = targets.reshape(-1, targets.shape[-1])
        cos =  torch.sum(outputs*targets,dim=-1)
        #cos[cos != cos] = 0
        cos[cos > 999/1000] = 999/1000
        cos[cos < -999/1000] = -999/1000
        rad = torch.acos(cos)
        loss = torch.rad2deg(rad)#0.5*(1-cos)#criterion(pred_gaze,gaze_dir)

        return loss


def run(args, train_dataloader, val_dataloader, _gaze_network, smpl, mesh_sampler):

    max_iter = len(train_dataloader)
    print("len of dataset:",max_iter)


    epochs = args.num_train_epochs

    optimizer = torch.optim.Adam(params=list(_gaze_network.parameters()),lr=args.lr,
                                            betas=(0.9, 0.999), weight_decay=0) 

    logger.info(
        ", lr:{:.6f}".format( optimizer.param_groups[0]["lr"])
    )

    start_training_time = time.time()
    end = time.time()
    _gaze_network.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    log_losses = AverageMeter()
    log_cos = AverageMeter()
    log_head = AverageMeter()

    criterion_cos = CosLoss().cuda(args.device)
    criterion_head = CosLoss().cuda(args.device)

    for epoch in range(args.num_init_epoch, epochs):
        for iteration, batch in enumerate(train_dataloader):

            iteration += 1
            #epoch = iteration
            _gaze_network.train()

            image = batch["image"].cuda(args.device)
            gaze_dir = batch["gaze_dir"].cuda(args.device)
            head_dir = batch["head_dir"].cuda(args.device)

            batch_imgs = image
            batch_size = image.size(0)

            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
            data_time.update(time.time() - end)

            # forward-pass
            direction, mdirection = _gaze_network(batch_imgs, smpl, mesh_sampler, is_train=True)

            # loss
            loss_cos = criterion_cos(direction,gaze_dir[:,(args.n_frames-1)//2]).mean()
            loss_head = criterion_head(mdirection,head_dir[:,(args.n_frames-1)//2]).mean()

            #loss = loss_cos + loss_bcos + loss_mse*40
            a = 0.7
            loss = (a)*loss_cos + (1-a)*loss_head



            # update logs
            log_losses.update(loss.item(), batch_size)
            log_cos.update(loss_cos.item(), batch_size)
            log_head.update(loss_head.item(), batch_size)
            #log_body.update(loss_body.item(), batch_size)

            # back prop
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if(iteration%args.logging_steps==0):
                eta_seconds = batch_time.avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                logger.info(
                    ' '.join(
                    ['eta: {eta}', 'epoch: {ep}', 'iter: {iter}',]
                    ).format(eta=eta_string, ep=epoch, iter=iteration) 
                    + ", loss:{:.4f}, cos:{:.2f}".format(log_losses.avg,log_cos.avg)
                    + ", head:{:.3f}".format(log_head.avg)
                )

            #if(iteration%int((max_iter+10)/3)==0):
            if(iteration*5==0):

                checkpoint_dir = save_checkpoint(_gaze_network, args, epoch, iteration)
                print("save trained model at ", checkpoint_dir)

        checkpoint_dir = save_checkpoint(_gaze_network, args, epoch, iteration)
        print("save trained model at ", checkpoint_dir)

        val = run_validate(args, val_dataloader, 
                            _gaze_network, 
                            criterion_cos,
                            smpl,
                            mesh_sampler,
                            )
        print("val:", val)

def run_validate(args, val_dataloader, gaze_network, criterion_cos, smpl,mesh_sampler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mse = AverageMeter()

    gaze_network.eval()
    smpl.eval()

    criterion = CosLoss().cuda(args.device)

    with torch.no_grad():        
        for iteration, batch in enumerate(val_dataloader):
            iteration += 1
            epoch = iteration

            image = batch["image"].cuda(args.device)
            gaze_dir = batch["gaze_dir"].cuda(args.device)

            batch_imgs = image
            batch_size = image.size(0)

            # forward-pass
            direction = gaze_network(batch_imgs, smpl, mesh_sampler)
            #print(direction.shape)

            loss = criterion(direction,gaze_dir).mean()

            # update logs
            mse.update(loss.item(), batch_size)

    return mse.avg


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--image_file_or_path", default='./test_images/human-body', type=str, 
                        help="test data")
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='models/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default="models/weights/metro/metro_3dpw_state_dict.bin", type=str, required=False,
                        help="Path to specific checkpoint for inference.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--model_checkpoint", default='output/checkpoint-6-54572/state_dict.bin', type=str, required=False,
                        help="Path to wholebodygaze checkpoint for inference.")
    #########################################################
    # Training parameters
    #########################################################
    parser.add_argument("--per_gpu_train_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=30, type=int, 
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--lr', "--learning_rate", default=1e-4, type=float, 
                        help="The initial lr.")
    parser.add_argument("--num_train_epochs", default=10, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--drop_out", default=0.1, type=float, 
                        help="Drop out ratio in BERT.")
    parser.add_argument("--num_init_epoch", default=0, type=int, 
                        help="initial epoch number.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,128', type=str, 
                        help="The Image Feature Dimension.")   
    parser.add_argument("--legacy_setting", default=True, action='store_true',)
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--run_eval_only", default=False, action='store_true',) 
    parser.add_argument('--logging_steps', type=int, default=10000, 
                        help="Log every X steps.")
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")
    parser.add_argument('--is_GAFA', type=bool, default=False,
                        help="use GAFA dataset or not, default is False")
    parser.add_argument("--n_frames", type=int, default=7)

    args = parser.parse_args()
    return args

# 最初はここから
def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    # 並列処理の設定
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)

    # default='output/'
    #mkdir(args.output_dir)
    logger = setup_logger("gaze", args.output_dir, 0)
    # randomのシード
    # default=88
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and SMPL utils
    # from metro.modeling._smpl import SMPL, Mesh
    mesh_smpl = SMPL().to(args.device)
    mesh_sampler = Mesh()


    logger.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))

    if args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        # この中っぽいとは思う。
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _metro_network = torch.load(args.resume_checkpoint)
    else:
        # どうやらこっち側みたい
        # Build model from scratch, and load weights from state_dict.bin
        trans_encoder = []
        # input_feat_dim default='2051,512,128'
        input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
        # hidden_feat_dim default='1024,256,128'
        hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
        output_feat_dim = input_feat_dim[1:] + [3]
        # init three transformer encoders in a loop
        # transformerの初期化
        # output_feat_dim = [512, 128, 3]
        for i in range(len(output_feat_dim)):
            # from metro.modeling.bert import BertConfig, METRO
            config_class, model_class = BertConfig, METRO
            # default='metro/modeling/bert/bert-base-uncased/'
            config = config_class.from_pretrained(args.model_name_or_path)

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]

            if args.legacy_setting==True:
                # During our paper submission, we were using the original intermediate size, which is 3072 fixed
                # We keep our legacy setting here 
                args.intermediate_size = -1
            else:
                # We have recently tried to use an updated intermediate size, which is 4*hidden-size.
                # But we didn't find significant performance changes on Human3.6M (~36.7 PA-MPJPE)
                args.intermediate_size = int(args.hidden_size*4)

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    #logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            # model_class = METRO
            model = model_class(config=config) 
            #logger.info("Init model from scratch.")
            trans_encoder.append(model)

        # for ここまで
        # init ImageNet pre-trained backbone model
        # arch default='hrnet-w64'
        if args.arch=='hrnet':
            hrnet_yml = 'models/hrnet/weights/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = './models/hrnet/weights/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/weights/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/weights/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Transformers total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
        # from metro.modeling.bert import METRO_Body_Network as METRO_Network
        # ここでモデルの初期化
        _metro_network = METRO_Network(args, config, backbone, trans_encoder, mesh_sampler)

        logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
        cpu_device = torch.device('cpu')
        state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
        _metro_network.load_state_dict(state_dict, strict=False)
        del state_dict
    # model の構築ここまで

    # update configs to enable attention outputs
    setattr(_metro_network.trans_encoder[-1].config,'output_attentions', True)
    setattr(_metro_network.trans_encoder[-1].config,'output_hidden_states', True)
    _metro_network.trans_encoder[-1].bert.encoder.output_attentions = True
    _metro_network.trans_encoder[-1].bert.encoder.output_hidden_states =  True
    for iter_layer in range(4):
        _metro_network.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
    for inter_block in range(3):
        setattr(_metro_network.trans_encoder[-1].config,'device', args.device)

    _metro_network.to(args.device)
    logger.info("Run inference")

    _gaze_network = GAZEFROMBODY(args, _metro_network)
    _gaze_network.to(args.device)

    if not args.num_init_epoch == 0:
        state_dict = torch.load(args.model_checkpoint)
        _gaze_network.load_state_dict(state_dict)
        del state_dict


    print(args.device)
    if args.device == 'cuda':
        print("distribution")
        _gaze_network = torch.nn.DataParallel(_gaze_network) # make parallel
        torch.backends.cudnn.benchmark = True


    #logger.info("Run train without lab")


    
    if not args.is_GAFA:
        exp_names = [
            'data20',
            'data23',
            'data25',
            'data29_0',
            'data29_1',
            'data29_2',
        ]
        random.shuffle(exp_names)
        # Ryukoku dataset
        dset = create_gafa_dataset(exp_names=exp_names, root_dir='data/GoTK', n_frames=args.n_frames)

    if args.is_GAFA:
        exp_names = [
        'living_room/005',
        'living_room/004',
        'kitchen/1015_4',
        'kitchen/1022_4',
        'library/1028_2',
        'library/1028_5',
        'library/1026_3',
        'courtyard/004',
        'courtyard/005',
        'lab/1013_1',
        'lab/1014_1',
                    ]
        random.shuffle(exp_names)
        # GAFA dataset
        dset = create_gafa_dataset(exp_names=exp_names, n_frames=args.n_frames)

    #train_idx, val_idx = np.arange(0, 800), np.arange(int(len(dset)*0.9), len(dset))
    train_idx, val_idx = np.arange(0, int(len(dset)*0.9)), np.arange(int(len(dset)*0.9), len(dset))
    train_dset, val_dset = random_split(dset, [len(train_idx), len(val_idx)])

    train_dataloader = DataLoader(
        train_dset, batch_size=1, num_workers=16, pin_memory=True, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True
    )
    # Training
    run(args, train_dataloader, val_dataloader, _gaze_network, mesh_smpl, mesh_sampler)

if __name__ == "__main__":
    args = parse_args()
    main(args)
