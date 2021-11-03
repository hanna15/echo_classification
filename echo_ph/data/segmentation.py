import os
import numpy as np
import bz2
import json
import pickle
from collections import defaultdict


class SegmentationAnalyser:
    def __init__(self, sample_name, segm_res_dir, model_view='psax'):
        dir = os.path.join(segm_res_dir, sample_name, model_view)
        segm_mask_path = os.path.join(dir, sample_name + '-frames.segment_pbz2')
        label_path = os.path.join(dir, sample_name + '-segmentation_label.json')
        label_path = os.path.join(os.path.dirname(label_path), sample_name + '-segmentation_label.json')
        with open(label_path, 'r') as file:
            self.labels = json.load(file)
        data = bz2.BZ2File(segm_mask_path, 'rb')
        self.segm_mask = np.asarray(pickle.load(data))
        self.w, self.h = self.segm_mask.shape[1:3]

    # def extraxt_max_frames(self, num_max_frames=5):
    #     volume_to_frame_nr = defaultdict(list)  # initialise the dict with an empty list (to add frame ids)
    #     for frame_nr, segm_mask_frame in enumerate(self.segm_mask):
    #         rv_vol = np.count_nonzero(segm_mask_frame == self.labels['rv'])
    #         lv_vol = np.count_nonzero(segm_mask_frame == self.labels['lv'])
    #         volume_to_frame_nr[lv_vol + rv_vol].append(frame_nr)
    #     sorted_volumes = sorted(volume_to_frame_nr)
    #     max_expansion_volumes = sorted_volumes[-num_max_frames:]  # get largest volumes
    #     max_expansion_frames = []
    #     cnt = 0
    #     for max_vol in max_expansion_volumes:
    #         for frame_nr in volume_to_frame_nr[max_vol]:
    #             cnt += 1
    #             max_expansion_frames.append(frame_nr)
    #             if cnt >= num_max_frames:
    #                 break
    #     return max_expansion_frames

    def extract_max_percentile_frames(self, percentile=90, maxp=True):
        volume_to_frame_nr = defaultdict(list)  # initialise the dict with an empty list (to add frame ids)
        for frame_nr, segm_mask_frame in enumerate(self.segm_mask):
            rv_vol = np.count_nonzero(segm_mask_frame == self.labels['rv'])
            lv_vol = np.count_nonzero(segm_mask_frame == self.labels['lv'])
            volume_to_frame_nr[lv_vol + rv_vol].append(frame_nr)

        volume_list = np.asarray(list(volume_to_frame_nr))
        p = percentile if maxp else (100-percentile)  # if find minp, reverse
        percentile = np.percentile(volume_list, p, interpolation="nearest")
        if maxp:
            top_percentile_volumes = volume_list[volume_list >= percentile]
        else:
            top_percentile_volumes = volume_list[volume_list <= percentile]
        max_expansion_frames = []
        for top_p in top_percentile_volumes:
            top_frame = volume_to_frame_nr[top_p]
            max_expansion_frames.extend(top_frame)
        return max_expansion_frames