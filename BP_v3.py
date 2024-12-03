# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from collections import defaultdict

import cv2 # AJT

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor


# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"

print("In on BP_v1")

def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name, "masks"), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, "masks", f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, "masks", object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, "masks", object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)


def save_overlay(frame_idx, frame_name, video_dir, masks, save_dir, color_map, video_name="Err"):
    frame_name += ".jpg"
    frame_path = os.path.join(video_dir, video_name, frame_name) # annot + jksdflsd + 000.jpg
    frame = cv2.imread(frame_path)
    if frame is None:
        raise ValueError(f"The frame at {frame_path} could not be loaded.")

    # combined_mask = np.zeros_like(frame[:,:,0], dtype=np.uint8)
    # Overlay each mask with a unique color for each object ID
    i = 0 # Iter for map
    for obj_id, mask in masks.items():
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        color = color_map[i]
        mask_indices = np.where(mask == 1)
        # combined_mask[mask_indices] = 255
        frame[mask_indices[0], mask_indices[1], :] = 0.4 * frame[mask_indices[0], mask_indices[1], :] + 0.6 * np.array(color)
        i += 1 # Iter for map

    save_path = os.path.join(save_dir, video_name, frame_name) 
    # save_folder = os.path.join(save_dir, video_name, 'masks')
    # save_path2 = os.path.join(save_folder, frame_name)

    # if not os.path.exists(save_folder):
    #     os.makedirs(save_folder)

    cv2.imwrite(save_path, frame)
    # cv2.imwrite(save_path2, combined_mask)


def maskExists(tensorIn): # My fxn
    return len(tensorIn[tensorIn > 0]) != 0

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_inference(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
    doBP=False,
):
    """Run VOS inference on a single video with the given predictor."""
    # load the video frames and initialize the inference state on this video
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    # fetch mask inputs from input_mask_dir (either only mask for the first frame, or all available masks)
    if not use_all_masks:
        # use only the first video's ground-truth mask as the input mask
        input_frame_inds = [0]
    else:
        # use all mask files available in the input_mask_dir as the input masks
        if not per_obj_png_file:
            input_frame_inds = [
                idx
                for idx, name in enumerate(frame_names)
                if os.path.exists(
                    os.path.join(input_mask_dir, video_name, f"{name}.png")
                )
            ]
        else:
            input_frame_inds = [
                idx
                for object_name in os.listdir(os.path.join(input_mask_dir, video_name))
                for idx, name in enumerate(frame_names)
                if os.path.exists(
                    os.path.join(input_mask_dir, video_name, object_name, f"{name}.png")
                )
            ]
        # check and make sure we got at least one input frame
        if len(input_frame_inds) == 0:
            raise RuntimeError(
                f"In {video_name=}, got no input masks in {input_mask_dir=}. "
                "Please make sure the input masks are available in the correct format."
            )
        input_frame_inds = sorted(set(input_frame_inds))

    # add those input masks to SAM 2 inference state before propagation
    object_ids_set = None
    for input_frame_idx in input_frame_inds:
        try:
            per_obj_input_mask, input_palette = load_masks_from_dir(
                input_mask_dir=input_mask_dir,
                video_name=video_name,
                frame_name=frame_names[input_frame_idx],
                per_obj_png_file=per_obj_png_file,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"In {video_name=}, failed to load input mask for frame {input_frame_idx=}. "
                "Please add the `--track_object_appearing_later_in_video` flag "
                "for VOS datasets that don't have all objects to track appearing "
                "in the first frame (such as LVOS or YouTube-VOS)."
            ) from e
        # get the list of object ids to track from the first input frame
        if object_ids_set is None:
            object_ids_set = set(per_obj_input_mask)
        for object_id, object_mask in per_obj_input_mask.items():
            # check and make sure no new object ids appear only in later frames
            if object_id not in object_ids_set:
                raise RuntimeError(
                    f"In {video_name=}, got a new {object_id=} appearing only in a "
                    f"later {input_frame_idx=} (but not appearing in the first frame). "
                    "Please add the `--track_object_appearing_later_in_video` flag "
                    "for VOS datasets that don't have all objects to track appearing "
                    "in the first frame (such as LVOS or YouTube-VOS)."
                )
            predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=object_mask,
            )

    # check and make sure we have at least one object to track
    if object_ids_set is None or len(object_ids_set) == 0:
        raise RuntimeError(
            f"In {video_name=}, got no object ids on {input_frame_inds=}. "
            "Please add the `--track_object_appearing_later_in_video` flag "
            "for VOS datasets that don't have all objects to track appearing "
            "in the first frame (such as LVOS or YouTube-VOS)."
        )
    # run propagation throughout the video and collect the results in a dict
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}  # video_segments contains the per-frame segmentation results

    numObjs = len(object_ids_set)

    if not doBP: # Prop like normal
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            per_obj_output_mask = {
                out_obj_id: (out_mask_logits[i] > score_thresh).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            video_segments[out_frame_idx] = per_obj_output_mask
    
    if doBP: # Do imperfect BP
        video_segments = {}  # per-frame 
        lostAtIdx = [0 for i in range(numObjs)]
        maskLost = [False for i in range(numObjs)]
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            for i, out_obj_id in enumerate(out_obj_ids):
                curTensor = out_mask_logits[i][0]
                if not maskExists(curTensor) and not maskLost[i]: # If all mask is lost, record frame index
                    lostAtIdx[i] = out_frame_idx
                    maskLost[i] = True
                if maskExists(curTensor) and maskLost[i]:
                    # Mask was regained, backPropogate
                    print("Begin BackProp for ", out_frame_idx-lostAtIdx[i], " frames")
                    for out_frame_idx2, out_obj_id2, out_mask_logits in predictor.propagate_in_video(inference_state, out_frame_idx,  max_frame_num_to_track = (out_frame_idx-lostAtIdx[i]), reverse=True):
                        if(out_obj_id != out_obj_id2): # BUG FIX
                            continue
                        video_segments[out_frame_idx2][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
                    
                    maskLost[i] = False
                    lostAtIdx[i] = out_frame_idx + 1
            
            
    
    color_map = {i: np.random.randint(30, 220, size=(3,)).tolist() for i in range(numObjs+1)}
    # Write the overlayed images for visual comparison
    for out_frame_idx, masks in video_segments.items():
        save_overlay(out_frame_idx, frame_names[out_frame_idx], base_video_dir, masks, save_dir=output_mask_dir, color_map=color_map, video_name=video_name)

    ## write the output masks as palette PNG files to output_mask_dir
    for out_frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[out_frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
            output_palette=output_palette,
        )


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_separate_inference_per_object(
    predictor,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
    doBP=False,
):
    """
    Run VOS inference on a single video with the given predictor.

    Unlike `vos_inference`, this function run inference separately for each object
    in a video, which could be applied to datasets like LVOS or YouTube-VOS that
    don't have all objects to track appearing in the first frame (i.e. some objects
    might appear only later in the video).
    """
    # load the video frames and initialize the inference state on this video
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.init_state(
        video_path=video_dir, async_loading_frames=False
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    initMasks = {} # My add
    # collect all the object ids and their input masks
    inputs_per_object = defaultdict(dict)
    for idx, name in enumerate(frame_names):
        if per_obj_png_file or os.path.exists(
            os.path.join(input_mask_dir, video_name, f"{name}.png")
        ):
            per_obj_input_mask, input_palette = load_masks_from_dir(
                input_mask_dir=input_mask_dir,
                video_name=video_name,
                frame_name=frame_names[idx],
                per_obj_png_file=per_obj_png_file,
                allow_missing=True,
            )
            for object_id, object_mask in per_obj_input_mask.items():
                # skip empty masks
                if not np.any(object_mask):
                    continue
                # if `use_all_masks=False`, we only use the first mask for each object
                if len(inputs_per_object[object_id]) > 0 and not use_all_masks:
                    continue
                print(f"adding mask from frame {idx} as input for {object_id=}")
                inputs_per_object[object_id][idx] = object_mask
                initMasks[object_id] = object_mask # My add
                
    def myIOU(mask1, mask2):
        m1, m2 = mask1.copy() > 0, mask2.copy() > 0
        return np.sum(np.logical_and(m1, m2)) / np.sum(np.logical_or(m1, m2)) # IOU
    
    numberOfBPs = 0

    if not doBP: # Prop like normal
        # run inference separately for each object in the video
        object_ids = sorted(inputs_per_object)
        output_scores_per_object = defaultdict(dict)
        for object_id in object_ids:
            # add those input masks to SAM 2 inference state before propagation
            input_frame_inds = sorted(inputs_per_object[object_id])
            predictor.reset_state(inference_state)
            for input_frame_idx in input_frame_inds:
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    mask=inputs_per_object[object_id][input_frame_idx],
                )

            # run propagation throughout the video and collect the results in a dict
            for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=min(input_frame_inds),
                reverse=False,
            ):
                obj_scores = out_mask_logits.cpu().numpy()
                output_scores_per_object[object_id][out_frame_idx] = obj_scores
    
    if doBP:
        # run inference separately for each object in the video
        object_ids = sorted(inputs_per_object)
        output_scores_per_object = defaultdict(dict)
        for object_id in object_ids:
            # add those input masks to SAM 2 inference state before propagation
            input_frame_inds = sorted(inputs_per_object[object_id])
            predictor.reset_state(inference_state)
            for input_frame_idx in input_frame_inds:
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=input_frame_idx,
                    obj_id=object_id,
                    mask=inputs_per_object[object_id][input_frame_idx],
                )
            
            # run propagation throughout the video and collect the results in a dict
            curMaskFound = True

            existBoolMap = {0: True} # Map each index to whether we found an object mask
            # initMask = initMasks[object_id]
            tupList = []
            lastFrame = 0
            for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video( # Default propagation
                inference_state,
                start_frame_idx=min(input_frame_inds),
                reverse=False,
            ):
                lastFrame = out_frame_idx
                obj_scores = out_mask_logits.cpu().numpy()
                output_scores_per_object[object_id][out_frame_idx] = obj_scores
                curMaskFound = maskExists(obj_scores)

                if not curMaskFound:
                    if(tupList == []):
                        tupList.append((out_frame_idx, out_frame_idx))
                    elif(tupList[-1][1] == out_frame_idx - 1):
                        tupList[-1] = (tupList[-1][0], out_frame_idx) # adv by 1
                    else:
                        tupList.append((out_frame_idx, out_frame_idx))
                    existBoolMap[out_frame_idx] = False
                    continue # Nothing more to do

                existBoolMap[out_frame_idx] = True
            print("BP tupleList: ", tupList)
            for tup in tupList:
                l,r = tup
                if(r+10 >= lastFrame): # cant bp
                    continue
                predictor.reset_state(inference_state)
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx= r + 10,
                    obj_id=object_id,
                    mask=output_scores_per_object[object_id][input_frame_idx+10],
                )
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx= r + 5,
                    obj_id=object_id,
                    mask=output_scores_per_object[object_id][input_frame_idx+5],
                )
            
                for out_frame_idx2, _2, out_mask_logits2 in predictor.propagate_in_video( # Backpropagation
                            inference_state,
                            start_frame_idx=r+5,
                            max_frame_num_to_track = r - l - 5,
                            reverse=True,
                        ):
                            obj_scores2 = out_mask_logits2.cpu().numpy()
                            if(existBoolMap[out_frame_idx2]): # If mask alr exists
                                # MERGE RULE = IGNORE 
                                continue
                            output_scores_per_object[object_id][out_frame_idx2] = obj_scores2


    # post-processing: consolidate the per-object scores into per-frame masks
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}  # video_segments contains the per-frame segmentation results

    for frame_idx in range(len(frame_names)):
        scores = torch.full(
            size=(len(object_ids), 1, height, width),
            fill_value=-1024.0,
            dtype=torch.float32,
        )
        for i, object_id in enumerate(object_ids):
            if frame_idx in output_scores_per_object[object_id]:
                scores[i] = torch.from_numpy(
                    output_scores_per_object[object_id][frame_idx]
                )

        if not per_obj_png_file:
            scores = predictor._apply_non_overlapping_constraints(scores)
        per_obj_output_mask = {
            object_id: (scores[i] > score_thresh).cpu().numpy()
            for i, object_id in enumerate(object_ids)
        }
        video_segments[frame_idx] = per_obj_output_mask

    
    
    # Write the overlayed images for visual comparison
    numObjs = len(object_ids)
    color_map = {i: np.random.randint(30, 220, size=(3,)).tolist() for i in range(numObjs+1)}
    for out_frame_idx, masks in video_segments.items():
        save_overlay(out_frame_idx, frame_names[out_frame_idx], base_video_dir, masks, save_dir=output_mask_dir, color_map=color_map, video_name=video_name)

    # write the output masks as palette PNG files to output_mask_dir
    for frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
            output_palette=output_palette,
        )
    if(doBP):
        print("Number of BackPropogations: ", numberOfBPs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM 2 model configuration file",
    )
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="../checkpoints/sam2.1_hiera_base_plus.pt",
        help="path to the SAM 2 model checkpoint",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        required=False, ##
        default = "/scratch/bdnb/vpurushotham/datasets/LVOS/valid/JPEGImages", ##
        help="directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument(
        "--input_mask_dir",
        type=str,
        required=False, ##
        default = "/scratch/bdnb/vpurushotham/datasets/LVOS/valid/Annotations", ##
        help="directory containing input masks (as PNG files) of each video",
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="text file containing the list of video names to run VOS prediction on",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=False, # AJT
        default = "/work/hdd/bdnb/atekkey/sam2/notebooks/results/bp_v3",
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--use_all_masks",
        action="store_true",
        help="whether to use all available PNG files in input_mask_dir "
        "(default without this flag: just the first PNG file as input to the SAM 2 model; "
        "usually we don't need this flag, since semi-supervised VOS evaluation usually takes input from the first frame only)",
    )
    parser.add_argument(
        "--per_obj_png_file",
        action="store_true",
        help="whether use separate per-object PNG files for input and output masks "
        "(default without this flag: all object masks are packed into a single PNG file on each frame following DAVIS format; "
        "note that the SA-V dataset stores each object mask as an individual PNG file and requires this flag)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    parser.add_argument(
        "--track_object_appearing_later_in_video",
        action="store_true",
        default=True, ## AJT FOR LVOS
        help="whether to track objects that appear later in the video (i.e. not on the first frame; "
        "some VOS datasets like LVOS or YouTube-VOS don't have all objects appearing in the first frame)",
    )
    print("late track = true")
    args = parser.parse_args()

    # if we use per-object PNG files, they could possibly overlap in inputs and outputs
    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false" if args.per_obj_png_file else "true")
    ]
    predictor = build_sam2_video_predictor(
        config_file=args.sam2_cfg,
        ckpt_path=args.sam2_checkpoint,
        apply_postprocessing=args.apply_postprocessing,
        hydra_overrides_extra=hydra_overrides_extra,
    )

    if args.use_all_masks:
        print("using all available masks in input_mask_dir as input to the SAM 2 model")
    else:
        print(
            "using only the first frame's mask in input_mask_dir as input to the SAM 2 model"
        )
    # if a video list file is provided, read the video names from the file

    # AJT: If Set directories


    # (otherwise, we use all subdirectories in base_video_dir)
    if args.video_list_file is not None:
        with open(args.video_list_file, "r") as f:
            video_names = [v.strip() for v in f.readlines()]
    else:
        video_names = [
            p
            for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        ]
    
    print("Cutting to 1 video")
    video_names = ["NFbsxmYE"] 

    print(f"running VOS prediction on {len(video_names)} videos:\n{video_names}")

    for n_video, video_name in enumerate(video_names):

        if os.path.exists(os.path.join(args.output_mask_dir, video_name)):
            print("Already done ", video_name)
            continue
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        if not args.track_object_appearing_later_in_video: # NEVER RUNS
            print("-----Together------- Err?")
            vos_inference(
                predictor=predictor,
                base_video_dir=args.base_video_dir,
                input_mask_dir=args.input_mask_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                score_thresh=args.score_thresh,
                use_all_masks=args.use_all_masks,
                per_obj_png_file=args.per_obj_png_file,
                doBP=True,
            )
        else:
            vos_separate_inference_per_object(
                predictor=predictor,
                base_video_dir=args.base_video_dir,
                input_mask_dir=args.input_mask_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                score_thresh=args.score_thresh,
                use_all_masks=args.use_all_masks,
                per_obj_png_file=args.per_obj_png_file,
                doBP=True,
            )
        # print("Break to only do 1 video")
        # break

    print(
        f"completed VOS prediction on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )


if __name__ == "__main__":
    main()
