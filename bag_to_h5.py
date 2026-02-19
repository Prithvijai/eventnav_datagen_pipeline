#!/usr/bin/env python3
"""
bag_to_h5.py  –  Convert ROS2 bags to HDF5 for NN training.
Precise Timestamp Sync + Fast Voxelization.
"""

import argparse
import bisect
import sys
import os

import cv2
import h5py
import numpy as np
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation as R
from event_camera_py import Decoder

# ── constants ──────────────────────────────────────────────────────────────────
VOXEL_WINDOW_US = 250_000   # 250 ms windows
NUM_BINS        = 5
HEIGHT, WIDTH   = 720, 1280
RGB_HEIGHT, RGB_WIDTH = 1024, 1280
ACTION_CHUNK    = 8
MIN_TRAJ_LEN    = 10

DEFAULT_EVENT_TOPIC = "/event_camera/events"
DEFAULT_RGB_TOPIC   = "/cam_sync/cam0/image_raw"
DEFAULT_ODOM_TOPIC  = "/odom"

def get_ts_ns(msg):
    """Safe extraction of header timestamp in nanoseconds."""
    if hasattr(msg, 'header'):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    return None

def decode_events(msg_obj, decoder):
    try:
        evs = decoder.decode(msg_obj)
        if evs is None or len(evs) == 0:
            evs = decoder.get_cd_events()
        if evs is not None and len(evs) > 0:
            return np.column_stack([evs['x'], evs['y'], evs['t'], evs['p']])
    except Exception: pass
    return None

def events_to_voxel_fast(events, num_bins, height, width, t_start, t_end):
    """Optimized voxelizer using bincount."""
    voxel = np.zeros(num_bins * height * width, dtype=np.float32)
    if events is None or len(events) == 0:
        return voxel.reshape(num_bins, height, width)

    t = events[:, 2].astype(np.float64)
    dt = float(t_end - t_start)
    if dt <= 0: return voxel.reshape(num_bins, height, width)

    bin_norm = (t - t_start) / dt * (num_bins - 1)
    bin_low  = np.floor(bin_norm).astype(np.int32)
    bin_high = bin_low + 1
    w_high   = (bin_norm - bin_low).astype(np.float32)
    w_low    = (1.0 - w_high).astype(np.float32)

    x, y = events[:, 0].astype(np.int32), events[:, 1].astype(np.int32)
    pol = np.where(events[:, 3] > 0, 1.0, -1.0).astype(np.float32)

    base_idx = y * width + x
    idx_low  = bin_low * (height * width) + base_idx
    idx_high = bin_high * (height * width) + base_idx

    mask_l = (bin_low >= 0) & (bin_low < num_bins) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    mask_h = (bin_high >= 0) & (bin_high < num_bins) & (x >= 0) & (x < width) & (y >= 0) & (y < height)

    voxel += np.bincount(idx_low[mask_l], weights=(pol * w_low)[mask_l], minlength=len(voxel))
    voxel += np.bincount(idx_high[mask_h], weights=(pol * w_high)[mask_h], minlength=len(voxel))

    voxel = voxel.reshape(num_bins, height, width)
    for b in range(num_bins):
        v_max = np.abs(voxel[b]).max()
        if v_max > 0: voxel[b] /= v_max
    return voxel

def interpolate_pose(odom_ts, poses, query_ts):
    idx = bisect.bisect_left(odom_ts, query_ts)
    if idx == 0: return poses[0]
    if idx >= len(odom_ts): return poses[-1]
    t0, t1 = odom_ts[idx-1], odom_ts[idx]
    alpha = (query_ts - t0) / (t1 - t0)
    p0, p1 = poses[idx-1], poses[idx]
    x = p0[0] + alpha * (p1[0] - p0[0])
    y = p0[1] + alpha * (p1[1] - p0[1])
    dyaw = np.arctan2(np.sin(p1[2] - p0[2]), np.cos(p1[2] - p0[2]))
    return np.array([x, y, p0[2] + alpha * dyaw])

def compute_action_chunk(odom_ts, poses, current_ts_ns):
    ref = interpolate_pose(odom_ts, poses, current_ts_ns)
    chunk = []
    win_ns = VOXEL_WINDOW_US * 1000
    for k in range(1, ACTION_CHUNK + 1):
        fut = interpolate_pose(odom_ts, poses, current_ts_ns + k * win_ns)
        dx, dy = fut[0] - ref[0], fut[1] - ref[1]
        dx_l = dx * np.cos(ref[2]) + dy * np.sin(ref[2])
        dy_l = -dx * np.sin(ref[2]) + dy * np.cos(ref[2])
        chunk.append([dx_l, dy_l, np.arctan2(np.sin(fut[2] - ref[2]), np.cos(fut[2] - ref[2]))])
    return np.array(chunk, dtype=np.float32)

def decode_rgb(msg_obj):
    h, w = msg_obj.height, msg_obj.width
    data = np.frombuffer(bytes(msg_obj.data), dtype=np.uint8)
    img = data.reshape((h, w, -1))
    enc = msg_obj.encoding.lower()
    if 'rgb' in enc: img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif 'bayer_rggb' in enc: img = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
    elif 'bayer' in enc: img = cv2.cvtColor(img, cv2.COLOR_BayerBG2BGR)
    return cv2.flip(img, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag', required=True)
    parser.add_argument('--h5', required=True)
    parser.add_argument('--storage', default='mcap')
    args = parser.parse_args()

    reader = SequentialReader()
    reader.open(StorageOptions(uri=args.bag, storage_id=args.storage), ConverterOptions('cdr', 'cdr'))
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_types = {t: get_message(topics[t]) for t in [DEFAULT_EVENT_TOPIC, DEFAULT_RGB_TOPIC, DEFAULT_ODOM_TOPIC] if t in topics}

    print("Pass 1: Pre-scanning Odom and RGB mapping...")
    odom_ts, odom_poses, rgb_ts, rgb_data = [], [], [], []
    bag_to_sensor_offset_ns = None

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == DEFAULT_ODOM_TOPIC:
            msg = deserialize_message(data, msg_types[topic])
            ts = get_ts_ns(msg)
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
            odom_ts.append(ts); odom_poses.append(np.array([p.x, p.y, yaw]))
        elif topic == DEFAULT_RGB_TOPIC:
            msg = deserialize_message(data, msg_types[topic])
            rgb_ts.append(get_ts_ns(msg)); rgb_data.append(data)
        elif topic == DEFAULT_EVENT_TOPIC and bag_to_sensor_offset_ns is None:
            msg = deserialize_message(data, msg_types[topic])
            bag_to_sensor_offset_ns = get_ts_ns(msg) - (msg.time_base * 1000)

    if not odom_ts or bag_to_sensor_offset_ns is None:
        sys.exit("[error] Missing Odom or Event sync data.")

    print(f"Pass 2: Processing with Bag-to-Sensor Offset: {bag_to_sensor_offset_ns/1e6:.2f} ms")
    reader = SequentialReader()
    reader.open(StorageOptions(uri=args.bag, storage_id=args.storage), ConverterOptions('cdr', 'cdr'))
    decoder = Decoder()
    win_ns = VOXEL_WINDOW_US * 1000
    
    curr_win_start = max(odom_ts[0], rgb_ts[0], bag_to_sensor_offset_ns)
    last_safe = odom_ts[-1] - ACTION_CHUNK * win_ns
    
    event_buffer = []
    step_idx, rgb_stored_count = 0, 0

    with h5py.File(args.h5, 'w') as f:
        d_vox = f.create_dataset('voxels', (0, NUM_BINS, HEIGHT, WIDTH), maxshape=(None, NUM_BINS, HEIGHT, WIDTH), dtype='float32', compression='gzip', chunks=(1, NUM_BINS, HEIGHT, WIDTH))
        d_act = f.create_dataset('actions', (0, ACTION_CHUNK, 3), maxshape=(None, ACTION_CHUNK, 3), dtype='float32')
        d_ts  = f.create_dataset('timestamps_ns', (0,), maxshape=(None,), dtype='int64')
        d_msk = f.create_dataset('rgb_mask', (0,), maxshape=(None,), dtype='bool')
        d_rgb = f.create_dataset('rgb_images', (0, RGB_HEIGHT, RGB_WIDTH, 3), maxshape=(None, RGB_HEIGHT, RGB_WIDTH, 3), dtype='uint8', compression='gzip', chunks=(1, RGB_HEIGHT, RGB_WIDTH, 3))
        d_rid = f.create_dataset('rgb_indices', (0,), maxshape=(None,), dtype='int32')

        while reader.has_next() and curr_win_start + win_ns <= last_safe:
            topic, data, _ = reader.read_next()
            if topic == DEFAULT_EVENT_TOPIC:
                evs = decode_events(deserialize_message(data, msg_types[topic]), decoder)
                if evs is not None:
                    event_buffer.append(evs)
                    t_sensor_query = (curr_win_start + win_ns - bag_to_sensor_offset_ns) // 1000
                    
                    if evs[-1, 2] >= t_sensor_query:
                        all_evs = np.concatenate(event_buffer)
                        while all_evs[-1, 2] >= t_sensor_query:
                            t_sensor_start = (curr_win_start - bag_to_sensor_offset_ns) // 1000
                            split = bisect.bisect_left(all_evs[:, 2], t_sensor_query)
                            
                            voxel = events_to_voxel_fast(all_evs[:split], NUM_BINS, HEIGHT, WIDTH, t_sensor_start, t_sensor_query)
                            
                            # Efficient RGB Matching
                            win_center = curr_win_start + win_ns // 2
                            r_idx = bisect.bisect_left(rgb_ts, win_center)
                            has_rgb = False
                            if r_idx < len(rgb_ts) and abs(rgb_ts[r_idx] - win_center) < win_ns // 2:
                                img = decode_rgb(deserialize_message(rgb_data[r_idx], msg_types[DEFAULT_RGB_TOPIC]))
                                d_rgb.resize(rgb_stored_count + 1, axis=0); d_rid.resize(rgb_stored_count + 1, axis=0)
                                d_rgb[rgb_stored_count], d_rid[rgb_stored_count] = img, step_idx
                                rgb_stored_count += 1
                                has_rgb = True

                            for d in [d_vox, d_act, d_ts, d_msk]: d.resize(step_idx + 1, axis=0)
                            d_vox[step_idx], d_act[step_idx], d_ts[step_idx], d_msk[step_idx] = voxel, compute_action_chunk(odom_ts, odom_poses, curr_win_start), curr_win_start, has_rgb
                            
                            step_idx += 1
                            curr_win_start += win_ns
                            t_sensor_query = (curr_win_start + win_ns - bag_to_sensor_offset_ns) // 1000
                            all_evs = all_evs[split:]
                            if step_idx % 20 == 0: print(f"  Steps: {step_idx} | RGB: {rgb_stored_count}", end='\r')
                            if curr_win_start + win_ns > last_safe: break
                        event_buffer = [all_evs]

    print(f"\nDone. {step_idx} steps saved to {args.h5}")

if __name__ == '__main__': main()
