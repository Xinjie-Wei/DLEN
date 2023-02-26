import argparse
import logging
import os
import torch

import cv2
import numpy as np
from src import utils
from models.DLEN_arch import DLEN
from src.dataset import LoadImages_LOL
from torch.utils.data import DataLoader

def get_args():
    parser = argparse.ArgumentParser(
        description='Converting from DLEN.')
    parser.add_argument('--model_dir', '-m',
                        default='./pretrained_models/LOLv2_best.pth',
                        help="Specify the directory of the trained model.",
                        dest='model_dir')
    parser.add_argument('--input_dir', '-i', help='Input image directory',
                        dest='input_dir',
                        default='G:/Dataset/eval15/low')
    parser.add_argument('--device', '-d', default='cuda',
                        help="Device: cuda or cpu.", dest='device')
    parser.add_argument('--output_dir',
                        default='./result/LOLv1')
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    if args.device.lower() == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device {device}')

    net = DLEN().to(device)

    net.eval()


    # data_type=1 denotes test on LOLv1, data_type=2 denotes test on LOLv2
    test = LoadImages_LOL(args.input_dir, img_size=(400, 592), data_type=1, augment=False,
                          normalize=False, is_train=False)
    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    total_params = utils.calc_para(net)
    logging.info('Total number of parameters: {}'.format(total_params))

    with torch.no_grad():
        checkpoint = torch.load(args.model_dir, map_location=device)
        net.load_state_dict(checkpoint['model'], strict=False)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for batch in test_loader:
            i = 0
            low_sRGB = batch['low_sRGB']
            low_sRGB_files = batch['low_sRGB_files']

            low_sRGB = low_sRGB.to(device=device, dtype=torch.float32)
            low_sRGB = torch.clamp(low_sRGB, 0, 1)
            output = net(low_sRGB)
            output = torch.clamp(output, 0, 1)

            output = utils.from_tensor_to_image(output, device=device)
            output = utils.outOfGamutClipping(output)

            in_dir, fn = os.path.split(low_sRGB_files[0])
            name, _ = os.path.splitext(fn)
            outsrgb_name = os.path.join(args.output_dir, name + '.png')
            img_name = name + '.png'
            output = output * 255
            cv2.imwrite(outsrgb_name, output.astype(np.uint8))
    print("Test image is done")


