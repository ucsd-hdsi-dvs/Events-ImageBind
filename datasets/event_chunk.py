import os
import argparse
import pickle as pkl
import numpy as np
import os.path as op
from dv import AedatFile
from numpy.lib import recfunctions as rfn
from numpy_groupies import aggregate
from tqdm import tqdm

def background_filter(events, frame_size=(260, 346), dt=70000):
    """ Filter out the background.
        Works by looking at the last shooting time of a certain event's location.
        If the last event at the same position is too long ago, discard it. 
    Args:
        events: ndarray, shape:(ta, 4), the events stream
        frame_size: tuple/list, (h, w), the frame size
        dt: int, threshold of 'long time' in us.
    Returns:
        events_filtered: ndarray, shape: (?, 4), filtered events.
    """
    h, w = frame_size
    ta = events.shape[0]
    lastTimesMap = np.zeros((w, h))
    index = np.zeros(ta)
    for i in tqdm(range(ta), total=ta):
        ts, xs, ys, ps = events[i]
        deltaT = ts-lastTimesMap[xs, ys]
        if deltaT > dt:
            index[i] = np.nan
        lastTimesMap[max(0, xs-1):min(w, xs+1), max(0, ys-1):min(h, ys+1)] = ts

    events_filtered = events[~(np.isnan(index))]
    return events_filtered

def hot_pixel_filter(events, frame_size=(260, 346), thres_percentile=99):
    """ Filter out the Hot Pixels.
        Hot pixels are defined as the pixels that record a number of event
        bigger than threventhotpixel.
    Args:
        events: ndarray, shape:(ta, 4), the events stream
        frame_size: tuple/list, (h, w), the frame size
        thres_percentile: float, 0~100, threshold percentile of hot pixel.
    Returns:
        events_filtered: ndarray, shape: (?, 4), filtered events.
    """
    h, w = frame_size

    hotpixelarray = aggregate(events[:, 1:3].T, np.ones_like(
        events[:, 0]), func='sum', size=(w, h), fill_value=0)
    threventhotpixel = np.percentile(hotpixelarray.flatten(), thres_percentile)
    selindexarray = hotpixelarray >= threventhotpixel
    [hpx, hpy] = np.nonzero(selindexarray.astype(int))
    fs = np.array((hpx, hpy)).T

    events_filtered = events[~(events[:, 1:3] == fs[:, None]).all(-1).any(axis=0)]
    return events_filtered

def event_frame():
    raise NotImplementedError

def event_chunk(path, out_dir, frames_per_sequence=16, prefix='sequence'):
    """
    Chunk the original event aedat file into sequences of paired and packed events, frames and imu data.
    Args:
        path: path to the original event aedat file
        out_dir: path to the output directory
        frames_per_sequence: number of frames per sequence
        prefix: prefix of the output file name
    """
    with AedatFile(path) as f:
        assert np.all([k in f.names for k in ['events', 'frames', 'imu']])
        frame_tmsps = np.array([frame_pkg.timestamp for frame_pkg in f['frames']])
        # imu_tmsps = np.array([imu_pkg.timestamp for imu_pkg in f['imu']])

    # # for each frame_idx, find the closest imu_idx
    # imu_indexes = np.searchsorted(imu_tmsps, frame_tmsps, side='left') - 1
    
    # # set values in imu_indexes to 0 if they are negative using np.where
    # imu_indexes = np.where(imu_indexes < 0, 0, imu_indexes)
    
    # # Optimize the code block above
    # accelerometers = []
    # gyroscopes = []
    # with AedatFile(path) as f:
    #     for i, imu_pkg in enumerate(f['imu']):
    #         if i in imu_indexes:
    #             accelerometers.append(imu_pkg.accelerometer)
    #             gyroscopes.append(imu_pkg.gyroscope)
    # accelerometers = np.array(accelerometers)
    # gyroscopes = np.array(gyroscopes)

    # assert len(frame_tmsps) == len(accelerometers) == len(gyroscopes) == len(imu_indexes)
            

    with AedatFile(path) as f:
        leftover_events = None
        frame_images = []
        frame_events = []
        # frame_accelerometers = []
        # frame_gyroscopes = []
        frame_timestamp_used = []
        sequence_count = 0

        for idx, frame_pkg in enumerate(f['frames']):
            assert frame_pkg.timestamp == frame_tmsps[idx]
            frame_timestamp = frame_tmsps[idx]
            frame_next_timestamp = frame_tmsps[idx+1] if idx+1 < len(frame_tmsps) else frame_timestamp + 1e6
            frame_image = frame_pkg.image
            frame_paired_events = [] if leftover_events is None else [leftover_events]
            
            while True:
                try:
                    raw_event_packet = next(f['events'].numpy())
                except StopIteration:
                    break

                event_packet = raw_event_packet[np.bitwise_and(frame_timestamp <= raw_event_packet['timestamp'], raw_event_packet['timestamp'] < frame_next_timestamp)]
                if len(event_packet) != 0:
                    frame_paired_events.append(event_packet)
                    if raw_event_packet['timestamp'][-1] >= frame_next_timestamp:
                        leftover_events = raw_event_packet[raw_event_packet['timestamp'] > frame_next_timestamp]
                        break
                else:
                    if raw_event_packet['timestamp'][0] >= frame_next_timestamp:
                        leftover_events = raw_event_packet
                        break
                    else:
                        continue
            
            frame_paired_events = np.hstack(frame_paired_events)
            frame_paired_events = rfn.drop_fields(frame_paired_events, ['_p1', '_p2'], asrecarray=True).view(np.ndarray)
            
            frame_paired_events = hot_pixel_filter(frame_paired_events)
            frame_paired_events = background_filter(frame_paired_events)

            frame_images.append(frame_image)
            frame_events.append(frame_paired_events)
            # frame_accelerometers.append(accelerometers[idx])
            # frame_gyroscopes.append(gyroscopes[idx])
            frame_timestamp_used.append(frame_timestamp)

            if (idx != 0 and idx % frames_per_sequence == 0):
                if len(frame_images) <= 1:
                    continue

                # save a dictionary of the sequence
                sequence = {
                    'images': np.stack(frame_images),
                    'events': frame_events[:-1],
                    # 'accelerometers': np.vstack(frame_accelerometers),
                    # 'gyroscopes': np.vstack(frame_gyroscopes),
                    'timestamps': np.array(frame_timestamp_used)
                }
                # save the sequence
                filename = op.join(out_dir, f'{prefix}-{sequence_count}.pkl')
                with open(filename, 'wb') as fo:
                    pkl.dump(sequence, fo)
                # reset the sequence
                frame_images = [frame_image]
                frame_events = [frame_paired_events]
                # frame_accelerometers = [accelerometers[idx]]
                # frame_gyroscopes = [gyroscopes[idx]]
                frame_timestamp_used = [frame_timestamp]
                sequence_count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chunk events and frames into sequences')
    parser.add_argument('-i', '--path', type=str, help='path to the aedat file')
    parser.add_argument('-r', '--directory', type=str, help='path to the directory of aedat files')
    parser.add_argument('-o', '--out_dir', type=str, help='path to the output directory')
    parser.add_argument('-f', '--frames_per_sequence', type=int, default=2, help='number of frames per sequence')
    args = parser.parse_args()

    # If args.directory is specified, process all aedat files in the directory
    in_paths = []
    if args.directory is not None:
        for filename in os.listdir(args.directory):
            if filename.endswith('.aedat'):
                in_paths.append(op.join(args.directory, filename))
    else:
        in_paths.append(args.path)

    for path in in_paths:
        # If no output directory is specified, use the same directory as the input file, 
        # and use the filename as the folder name
        if args.out_dir is None:
            args.out_dir = op.join(op.dirname(path), op.splitext(op.basename(path))[0])

        # Create the output directory if it doesn't exist
        if not op.exists(args.out_dir):
            os.makedirs(args.out_dir, exist_ok=True)

        try:
            event_chunk(path, args.out_dir, args.frames_per_sequence, prefix=op.splitext(op.basename(path))[0])
        except AssertionError:
            print(f'File {path} does not contain frames or imu data, skipping...')