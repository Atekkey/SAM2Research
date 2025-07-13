from os import name
import cv2
import numpy as np
import matplotlib.pyplot as plt

def parse_custom_rle_line(line):
    parts = line.strip().split(',')
    assert parts[0].startswith('m')
    
    # masklet_id = (parts[0][1])
    x, y = int((parts[0])[1:]), int(parts[1])
    width, height = int(parts[2]), int(parts[3])
    
    
    rle = list(map(int, parts[4:]))
    return {
        # 'masklet_id': masklet_id,
        'x': x,
        'y': y,
        'width': width,
        'height': height,
        'rle': rle
    }

def decode_rle_to_mask(rle, width, height):
    mask_flat = []
    val = 0
    for run in rle:
        mask_flat.extend([val] * run)
        val = 1 - val
    
    # Adjust length
    expected_len = width * height
    if len(mask_flat) < expected_len:
        mask_flat.extend([0] * (expected_len - len(mask_flat)))
    else:
        mask_flat = mask_flat[:expected_len]
    
    return np.array(mask_flat, dtype=np.uint8).reshape((height, width))

def compute_bounding_box(mask):
    """Returns (x, y, w, h) bounding box of the mask."""
    ys, xs = np.where(mask == 1)
    if len(xs) == 0 or len(ys) == 0:
        return None  # Empty mask
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

def mask_to_img(mask):
    """Converts a binary mask to an RGB image."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    
    img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    img[mask == 1] = [255, 0, 0]  # Red for the mask
    return img

# if name == "__main__":
    # folder = "/work/nvme/bdnb/atekkey/didi_data/GOT-10k_GOT-10k_Val_000014"
    # img_path = f"{folder}/color/00000001.jpg"
    # image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

#     # Step 1: Decode the RLE
#     rle_line = ""
#     with open(f"{folder}/first_frame_segm.txt", "r") as f:
#     rle_line = f.readline().strip()

#     info = parse_custom_rle_line(rle_line)
#     mask = decode_rle_to_mask(info['rle'], info['width'], info['height'])
#     # bbox = compute_bounding_box(mask)

#     # Step 2: Create a blank RGB image (you can load your own with cv2.imread if needed)
#     image = np.ones((info['height'], info['width'], 3), dtype=np.uint8)

#     # Draw bbox
#     # if bbox:
#     #     x, y, w, h = bbox
#     #     cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)  # green


#     # Step 3: Overlay mask as red (optional: set alpha blending)
#     overlay = image.copy()
#     overlay[mask == 1] = [255, 0, 0]  # red mask

#     # Optional: blend original and mask overlay
#     # alpha = 0.5
#     # blended = ((1 - alpha) * image + alpha * overlay).astype(np.uint8)
#     blended = overlay

#     # Step 4: Display
#     # plt.figure(figsize=(10, 6))
#     # plt.imshow(blended)
#     # plt.title(f"Mask for Frame {info['frame_id']}")
#     # plt.axis('off')
#     # plt.show()
#     cv2.imwrite("/work/nvme/bdnb/atekkey/trash/0.png", cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)) 
