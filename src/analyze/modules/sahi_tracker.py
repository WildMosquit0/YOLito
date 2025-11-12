#!/usr/bin/env python3
"""
Modified SORT tracker example to read detections from a CSV file,
compute object trajectories, and plot the (x, y) centers as lines.
"""

from __future__ import print_function

import numpy as np
import pandas as pd
from typing import Dict
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

np.random.seed(0)

# ------------------ SORT helper functions and classes ------------------

def linear_assignment(cost_matrix):
    """Solve the linear assignment problem using scipy."""
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    """
    Computes IOU between two bboxes in the form [x1,y1,x2,y2].
    Returns an array of IOU values.
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / (
        ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])) +
        ((bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])) - wh
    )
    return o

def convert_bbox_to_z(bbox):
    """
    Converts a bounding box in [x1,y1,x2,y2] form to [x, y, s, r] where
    x, y is the centre, s is scale (area) and r is aspect ratio.
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h    # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x, score=None):
    """
    Converts a bounding box from [x,y,s,r] (centre form) to [x1,y1,x2,y2].
    If score is provided, returns [x1,y1,x2,y2,score].
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))

class KalmanBoxTracker(object):
    """
    Represents the internal state of individual tracked objects observed as bounding boxes.
    """
    count = 0
    def __init__(self, bbox):
        """
        Initializes a tracker using an initial bounding box.
        """
        # Define a constant velocity model.
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000.  # High uncertainty for initial velocities.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with an observed bounding box.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked objects (both in [x1,y1,x2,y2] form).
    Returns:
      - matches (each row is [detection_index, tracker_index])
      - unmatched detections indices
      - unmatched trackers indices
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 2), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class sahi_tracker(object):
    """
    SORT tracker: creates and updates tracklets for object detections.
    """
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [x1, y1, x2, y2, score]
        Returns:
          An array with each row in the format [x1, y1, x2, y2, track_id]
        """
        self.frame_count += 1

        # Get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        # Remove any trackers with NaN predictions.
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update matched trackers with assigned detections.
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Create and initialize new trackers for unmatched detections.
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 so that IDs are positive
            i -= 1
            # Remove dead tracklets.
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

# ------------------ Main routine to load CSV, track, and plot ------------------

def _detections_from_frame(frame_df: pd.DataFrame) -> np.ndarray:
    """
    Convert a frame slice to the [x1, y1, x2, y2, score] matrix expected by the tracker.
    """
    if frame_df.empty:
        return np.empty((0, 5))

    dets = frame_df[['x', 'y', 'w', 'h', 'confidence']].to_numpy(dtype=float, copy=True)
    dets[:, 2] = dets[:, 0] + dets[:, 2]
    dets[:, 3] = dets[:, 1] + dets[:, 3]
    return dets


def _match_tracks_to_detections(detections: np.ndarray, tracks: np.ndarray, threshold: float = 0.05) -> Dict[int, int]:
    """
    Match each detection row to the most likely tracker id using IoU to keep the association
    consistent with the Kalman filter updates.
    Returns a mapping of {detection_row_index: track_id}.
    """
    assignments: Dict[int, int] = {}
    if detections.size == 0 or tracks.size == 0:
        return assignments

    iou_matrix = iou_batch(detections[:, :4], tracks[:, :4])
    if iou_matrix.size == 0:
        return assignments

    while True:
        max_index = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        max_iou = iou_matrix[max_index]
        if max_iou < threshold:
            break
        det_idx, track_idx = max_index
        assignments[det_idx] = int(tracks[track_idx, 4])
        iou_matrix[det_idx, :] = -1
        iou_matrix[:, track_idx] = -1

    return assignments


def _fill_missing_track_ids(track_ids: np.ndarray) -> np.ndarray:
    """
    Replace any -1 placeholders with unique positive ids so downstream aggregation
    never encounters NaNs when SAHI tracking cannot associate a detection.
    """
    missing = np.where(track_ids < 0)[0]
    if not len(missing):
        return track_ids

    next_id = int(track_ids.max() if track_ids.max() > 0 else 0) + 1
    for idx in missing:
        track_ids[idx] = next_id
        next_id += 1
    return track_ids


def main(csv_file='results.csv', output_dir='.'):
    try:
        df = pd.read_csv(csv_file, sep=',')
        # Remove extra whitespace from column names (if any)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print("Error reading CSV file:", e)
        return

    df['frame'] = df['image_idx'].astype(int) + 1
    df.sort_values(['frame', 'image_idx', 'box_idx'], inplace=True)
    tracker = sahi_tracker(max_age=10000, min_hits=0, iou_threshold=0.001)

    track_ids = np.full(len(df), -1, dtype=int)

    for frame_value, frame_data in df.groupby('frame', sort=True):
        dets = _detections_from_frame(frame_data)
        tracks = tracker.update(dets)
        assignments = _match_tracks_to_detections(dets, tracks)
        frame_indices = frame_data.index.to_numpy()
        for det_idx, track_id in assignments.items():
            track_ids[frame_indices[det_idx]] = track_id

    df['track_id'] = _fill_missing_track_ids(track_ids)
    df.to_csv(csv_file, index=False)
    print(f"Updated SAHI slicing CSV with Kalman-filtered track IDs: {csv_file}")
