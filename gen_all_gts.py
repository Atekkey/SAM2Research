import os
from gen_gt0 import parse_custom_rle_line as parse_rle
from gen_gt0 import decode_rle_to_mask as rle_to_mask
from gen_gt0 import mask_to_img

import cv2
import numpy as np

path = "/work/nvme/bdnb/atekkey/didi_data"
gt_path = "/work/nvme/bdnb/atekkey/didi_data_gts"
folders = sorted([name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))])
i = 0
for folder in folders:
    i+=1
    os.makedirs(os.path.join(gt_path, folder), exist_ok=True)
    img_name = sorted(os.listdir(os.path.join(path, folder, "color")))[0]
    gt_img_path = os.path.join(gt_path, folder, img_name)

    rle_path = os.path.join(path, folder, "first_frame_segm.txt")
    rle = ""
    with open(rle_path, "r") as f:
        rle = f.readline().strip()
    rle_info = parse_rle(rle)
    mask = rle_to_mask(rle_info['rle'], rle_info['width'], rle_info['height'])
    img = mask_to_img(mask)
    cv2.imwrite(gt_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("Processed:", i, "/", len(folders))

    
