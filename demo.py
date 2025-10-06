'''
Author: Hui Liu (base test flow), adapted for single-image demo
'''

import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from models import build_model
from main import get_args_parser


def preprocess_bgr(img_bgr, width: int, height: int):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if width > 0 and height > 0:
        img_rgb = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_CUBIC)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    tensor = normalize(to_tensor(img_rgb))  # 3xHxW in [-1,1]
    return tensor.unsqueeze(0)  # 1x3xHxW


def run_single_image_demo():
    # Hardcoded configuration: change these to your paths
    IMAGE_PATH = '/absolute/path/to/your.jpg'
    CHECKPOINT = './checkpoints/weights/checkpoint_TUT/checkpoint_TUT.pth'
    SAVE_DIR = './results/demo'
    DEVICE = 'cuda'  # or 'cpu'
    LOAD_WIDTH, LOAD_HEIGHT = 512, 512

    if not os.path.isfile(IMAGE_PATH):
        raise FileNotFoundError(f'Image not found: {IMAGE_PATH}')

    # Build default args without reading CLI, then override what we need
    parser = get_args_parser()
    args = parser.parse_args([])
    args.phase = 'test'
    args.device = DEVICE
    args.load_width = LOAD_WIDTH
    args.load_height = LOAD_HEIGHT

    device = torch.device(args.device)
    model, _ = build_model(args)

    state_dict = torch.load(CHECKPOINT, map_location='cpu')
    if isinstance(state_dict, dict) and 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    img_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f'Failed to read image: {IMAGE_PATH}')
    inp = preprocess_bgr(img_bgr, args.load_width, args.load_height).to(device)

    with torch.no_grad():
        logits = model(inp)
        prob = torch.sigmoid(logits)[0, 0, ...].cpu().numpy()

    mask_vis = (prob * 255.0).astype(np.uint8)
    mask_vis = cv2.resize(mask_vis, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    base = Path(IMAGE_PATH).stem
    out_pre = save_dir / f'{base}_pre.png'
    cv2.imwrite(str(out_pre), mask_vis)

    print('----------------------------------------------------------------------------------------------')
    print(f'Saved prediction: {out_pre}')
    print('----------------------------------------------------------------------------------------------')
    print('Finished!')


if __name__ == '__main__':
    run_single_image_demo()



