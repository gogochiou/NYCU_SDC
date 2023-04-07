# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import argparse
import copy
import os
import sys
import time
import json
import math

import numpy as np
import torch

from nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import transform_matrix
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from tqdm import tqdm
from copy import deepcopy

from scipy.optimize import linear_sum_assignment as linear_assignment
from tracker import greedy_assignment


NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck',
]

# 99.9 percentile of the l2 velocity error distribution (per class / 0.5 second)
# This is an earlier statistics and I didn't spend much time tuning it.
# Tune this for your model should provide some considerable AMOTA improvement
NUSCENE_CLS_VELOCITY_ERROR = {
    'car': 4,
    'truck': 4,
    'bus': 5.5,
    'trailer': 3,
    'pedestrian': 1,
    'motorcycle': 13,
    'bicycle': 3,
    'construction_vehicle': 1,
    'barrier': 1,
    'traffic_cone': 1,
}

# compute distance matrix
def create_distance_matrix(tracks1, tracks2):
    # initialize distance matrix
    distances = np.ndarray(shape=(len(tracks1), len(tracks2)))

    # for every tracklet of both tracks lists
    for row in range(len(tracks1)):
        for col in range(len(tracks2)):

            # check if potential match has same class (else: invalid match)
            if (tracks1[row]['tracking_name'] == tracks2[col]['tracking_name']):

                # compute pure distance (eucl. distance)
                dist = math.sqrt((tracks1[row]['translation'][0] - tracks2[col]['translation'][0]) ** 2 + \
                                 (tracks1[row]['translation'][1] - tracks2[col]['translation'][1]) ** 2)

                # determine whether distance is close enough to count as a match
                if (dist <= NUSCENE_CLS_VELOCITY_ERROR[tracks1[row]['tracking_name']]):
                    distances[row][col] = dist

                # else: invalid match
                else:
                    distances[row][col] = 1e18

            # set distance to infinite if match is invalid
            else:
                distances[row][col] = 1e18

    return distances

def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = os.path.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--out_dir", default="data/track_results", help="the dir to save logs and tracking results")
    parser.add_argument("--ct_result", type=str, default=None)
    parser.add_argument("--frames_meta_path", type=str, default='data/frames_meta.json')

    parser.add_argument("--evaluate", type=int, default=0)
    parser.add_argument("--dataroot", type=str, default='data/nuscenes')
    args,opts = parser.parse_known_args()

    # load CenterTrack trackings
    with open(args.ct_result, 'rb') as f:
        ct_tracking = json.load(f)['results']
    
    # read frame meta
    with open(args.frames_meta_path, 'rb') as f:
        frames=json.load(f)['frames']

    # prepare writen output file
    nusc_annos = {
        "results": {},
        "meta": None,
    }

    # start tracking id rearrangement ************************
    print("Begin Tracking id rearrangment\n")
    start = time.time()
    tracking_id_counter = 1
    previous_token = -1
    len_frames = len(frames)

    # for each frame
    for i in tqdm(range(len(ct_tracking))):
        token = frames[i]['token']  # get frameID (=token)

        # reset tracking after one video sequence
        if frames[i]['first']:
            # tracker.reset()
            tracking_id_counter = 1

            # in the first frame, simply give every tracklet a unique tracking id in ascending order
            for tracklet in ct_tracking[token]:
                tracklet['tracking_id'] = tracking_id_counter
                tracking_id_counter += 1

        # for all subsequent frames, do matching with its previous frame
        else:
            # calculate distance matrix between current tracklets and tracklets from previous frame
            track_distances = create_distance_matrix(ct_tracking[token], ct_tracking[previous_token])

            # find best matching of the two tracking results:
            # use greedy algorithm
            matched_ids = greedy_assignment(copy.deepcopy(track_distances))

            # assign new tracking ids:
            for i in range(len(ct_tracking[token])):

                # for all matched ids, assign previous tracking id
                if i in matched_ids[:,0]:
                    matching_id = matched_ids[[item[0] == i for item in matched_ids].index(True)][1]
                    ct_tracking[token][i]['tracking_id'] = ct_tracking[previous_token][matching_id]['tracking_id']

                # for all unmatched tracklets, give new tracking ids
                else:
                    ct_tracking[token][i]['tracking_id'] = tracking_id_counter
                    tracking_id_counter += 1

        previous_token = token

        # prepare writen results file
        annos = []
        for item in ct_tracking[token]:
            nusc_anno = {
                "sample_token": token,
                "translation": item['translation'],
                "size": item['size'],
                "rotation": item['rotation'],
                "velocity": item['velocity'],
                "detection_name": item['detection_name'],
                "attribute_name": item['attribute_name'],
                "detection_score": item['detection_score'],
                "tracking_name": item['tracking_name'],
                "tracking_score": item['tracking_score'],
                "tracking_id": item['tracking_id'],
            }
            annos.append(nusc_anno)
        nusc_annos["results"].update({token: annos})
    
    # calculate computation time
    end = time.time()
    second = (end - start)
    speed = len_frames / second
    print("======")
    print("The speed is {} FPS".format(speed))
    print("tracking results have {} frames".format(len(nusc_annos["results"])))

    # add meta info to writen result file
    nusc_annos["meta"] = {
        "use_camera": True,
        "use_lidar": False,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }

    # write result file
    root_path = args.out_dir
    track_res_path = os.path.join(root_path, 'centertrack_rearrange.json')
    mkdir_or_exist(os.path.dirname(track_res_path))
    with open(track_res_path, "w") as f:
        json.dump(nusc_annos, f)
    print(f"tracking results write to {track_res_path}\n")

    # Evaluation
    if args.evaluate:
        print("======")
        print("Start evaluating tracking results")
        output_dir = os.path.join(root_path, 'eval')
        eval(
            track_res_path,
            "val",
            output_dir,  # instead of args.work_dir,
            args.dataroot
        )

    return speed

def eval(res_path, eval_set="val", output_dir=None, data_path=None):
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory as track_configs

    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=data_path,
    )
    metrics_summary = nusc_eval.main()


if __name__ == '__main__':
    main()