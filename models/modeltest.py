

import argparse
import os
import os.path as op
import torch
from bert.model_bert import GAZEBERT_Network as GAZEBERT
from hrnet.config import config as hrnet_config
from hrnet.config import update_config as hrnet_update_config
from hrnet.hrnet_cls_net_featmaps import get_cls_net
from utils.logger import setup_logger
from utils.miscellaneous import mkdir

from PIL import Image
from torchvision import transforms

#from dataloader.gafa_loader import create_gafa_dataset

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


def run_inference(args, image_list, _gaze_bert):

    _gaze_bert.eval()

    for image_file in image_list:
        if "pred" not in image_file:
            print(image_file)

            img = Image.open(image_file)
            # from torchvision import transforms
            img_tensor = transform(img)
            #img_visual = transform_visualize(img)

            batch_imgs = torch.unsqueeze(img_tensor, 0)#.cuda()
            #batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()

            output = _gaze_bert(batch_imgs)
            #output = backbone(batch_imgs)
            #print(image_file)
            #print(output)
            print(output.size())




    return



def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--image_file_or_path", default='./data/sample', type=str, 
                        help="test data")

    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')

    #########################################################
    # Others
    #########################################################
    parser.add_argument("--device", type=str, default='cuda',
                        help="cuda or cpu")

    args = parser.parse_args()
    return args

def main(args):

    print("in main")
    global logger
    mkdir(args.output_dir)
    logger = setup_logger("model Test", args.output_dir, 0)


    args.device = torch.device(args.device)

    # init ImageNet pre-trained backbone model
    if args.arch=='hrnet':
        hrnet_yml = 'models/hrnet/weights/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        hrnet_checkpoint = './models/hrnet/weights/hrnetv2_w40_imagenet_pretrained.pth'
        hrnet_update_config(hrnet_config, hrnet_yml)
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

    backbone_total_param = sum(p.numel() for p in backbone.parameters())
    logger.info("Backbone total parameters: {}".format(backbone_total_param))


    # Initialize GAZEBERT model 
    _gaze_bert = GAZEBERT(args, backbone)



    image_list = []

    if not args.image_file_or_path:
        raise ValueError("image_file_or_path not specified")
    elif op.isdir(args.image_file_or_path):
        for filename in os.listdir(args.image_file_or_path):
            if filename.endswith(".png") or filename.endswith(".jpg") and "pred" not in filename:
                image_list.append(args.image_file_or_path+"/"+filename)
    else:
        raise ValueError("Cannot find images at {}".format(args.image_file_or_path))

    run_inference(args, image_list, _gaze_bert)



if __name__ == "__main__":
    args = parse_args()
    main(args)
