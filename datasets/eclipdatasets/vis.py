import copy

import numpy as np
import torch


def make_event_histogram(x, y, p, red, blue, shape, thresh=10., **kwargs):
    """Event polarity histogram."""
    # count the number of positive and negative events per pixel
    H, W = shape
    pos_x, pos_y = x[p > 0].astype(np.int32), y[p > 0].astype(np.int32)
    pos_count = np.bincount(pos_x + pos_y * W, minlength=H * W).reshape(H, W)
    neg_x, neg_y = x[p < 0].astype(np.int32), y[p < 0].astype(np.int32)
    neg_count = np.bincount(neg_x + neg_y * W, minlength=H * W).reshape(H, W)
    hist = np.stack([pos_count, neg_count], axis=-1)  # [H, W, 2]

    # remove hotpixels, i.e. pixels with event num > thresh * std + mean
    if thresh > 0:
        if kwargs.get('count_non_zero', False):
            mean = hist[hist > 0].mean()
            std = hist[hist > 0].std()
        else:
            mean = hist.mean()
            std = hist.std()
        hist[hist > thresh * std + mean] = 0

    # normalize
    hist = hist.astype(np.float32) / hist.max()  # [H, W, 2]

    # colorize
    cmap = np.stack([red, blue], axis=0).astype(np.float32)  # [2, 3]
    img = hist @ cmap  # [H, W, 3]

    # alpha-masking with pure white background
    if kwargs.get('background_mask', True):
        weights = np.clip(hist.sum(-1, keepdims=True), a_min=0, a_max=1)
        background = np.ones_like(img) * 255.
        img = img * weights + background * (1. - weights)

    img = np.round(img).astype(np.uint8)  # [H, W, 3], np.uint8 in (0, 255)

    return img


def parse_events(events):
    """Read (x,y,t,p) from input events (can be np.array or dict)."""
    if isinstance(events, dict):
        x, y, t, p = events['x'], events['y'], events['t'], events['p']
    else:
        x, y, t, p = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    x, y, p = x.astype(np.int32), y.astype(np.int32), p.astype(np.int32)
    t_us = t * 1e6  # convert to us unit
    return x, y, t_us, p


def split_event_count(t, N=30000):
    """Split the events according to event count."""
    tot_cnt = len(t)

    # if the event count is too small, just return the whole chunk
    if tot_cnt < N:
        return [0], [tot_cnt], [t[0]], [t[-1]]

    # find the start and end time of each event chunk w.r.t event index
    idx = np.arange(0, tot_cnt, N).tolist()
    idx1, idx0 = idx[1:], idx[:-1]
    # add the last index if the last chunk of events is not so small
    if tot_cnt - idx[-1] > N * 0.5:
        idx0.append(tot_cnt - N)
        idx1.append(tot_cnt)
    t0, t1 = t[idx0], t[np.array(idx1) - 1]

    return idx0, idx1, t0, t1

def event_frame(xs,ys,ps,sensor_size=(260,346)):
    x_positive=xs[ps==1]
    y_positive=ys[ps==1]
    p_positive=ps[ps==1]

    x_negative=xs[ps==-1]
    y_negative=ys[ps==-1]
    p_negative=ps[ps==-1]

    events_positive_frame=events_to_image_torch(x_positive,y_positive,p_positive,sensor_size=sensor_size)
    events_negative_frame=events_to_image_torch(x_negative,y_negative,p_negative,sensor_size=sensor_size)
    events_negative_frame=torch.abs(events_negative_frame)
    events_sum_frame=events_positive_frame+events_negative_frame
    event_frame=torch.stack([events_positive_frame,events_negative_frame,events_sum_frame],dim=0)
    return event_frame

def events_to_image_torch(xs, ys, ps,
        device=None, sensor_size=(260, 346), clip_out_of_range=False,
        interpolation=None, padding=True, default=0):
    """
    Method to turn event tensor to image. Allows for bilinear interpolation.
    @param xs Tensor of x coords of events
    @param ys Tensor of y coords of events
    @param ps Tensor of event polarities/weights
    @param device The device on which the image is. If none, set to events device
    @param sensor_size The size of the image sensor/output image
    @param clip_out_of_range If the events go beyond the desired image size,
       clip the events to fit into the image
    @param interpolation Which interpolation to use. Options=None,'bilinear'
    @param padding If bilinear interpolation, allow padding the image by 1 to allow events to fit:
    @returns Event image from the events
    """
    xs=torch.tensor(xs.copy(), dtype=torch.long)
    ys = torch.tensor(ys.copy(), dtype=torch.long)
    ps = torch.tensor(ps.copy(),dtype=torch.float32)
    
    if device is None:
        device = xs.device
    img_size = list(sensor_size)

    mask = torch.ones(xs.size(), device=device)
    if clip_out_of_range:
        zero_v = torch.tensor([0.], device=device)
        ones_v = torch.tensor([1.], device=device)
        clipx = img_size[1] if interpolation is None and padding==False else img_size[1]-1
        clipy = img_size[0] if interpolation is None and padding==False else img_size[0]-1
        mask = torch.where(xs>=clipx, zero_v, ones_v)*torch.where(ys>=clipy, zero_v, ones_v)

    img = (torch.ones(img_size)*default).to(device)
    if xs.dtype is not torch.long:
        xs = xs.long().to(device)
    if ys.dtype is not torch.long:
        ys = ys.long().to(device)
    try:
        mask = mask.long().to(device)
        xs, ys = xs*mask, ys*mask
        img.index_put_((ys, xs), ps, accumulate=True)
    except Exception as e:
        print("Unable to put tensor {} positions ({}, {}) into {}. Range = {},{}".format(
            ps.shape, ys.shape, xs.shape, img.shape,  torch.max(ys), torch.max(xs)))
        raise e
    return img


def events2frames(
        events,  # [N, 4 (x,y,t,p)]
        split_method ='event_count',  # 'event_count'
        convert_method = 'event_histogram',  # 'event_histogram'
        shape=(180, 240),
        **kwargs,
):
    """Convert events to 2D frames."""
    # some additional arguments
    grayscale = kwargs.pop('grayscale', True)  # True, False

    # parse different input formats
    x, y, t, p = parse_events(events)

    # split the events into different chunks
    assert split_method == 'event_count'
    N = int(kwargs['N'])
    idx0, idx1, t0, t1 = split_event_count(t, N)

    # color map for pos and neg events
    if grayscale:
        if isinstance(grayscale, bool):
            v = 127
        else:
            v = np.array(grayscale)  # values in addition to 127
        red = np.round(np.ones(3) * v).astype(np.uint8)
        blue = np.round(np.ones(3) * v).astype(np.uint8)
    else:
        red = np.array([255, 0, 0], dtype=np.uint8)
        blue = np.array([0, 0, 255], dtype=np.uint8)

    frames = []
    for t_idx, (i0, i1) in enumerate(zip(idx0, idx1)):
        xx, yy, pp, tt = x[i0:i1], y[i0:i1], p[i0:i1], t[i0:i1]
        if convert_method == 'event_histogram':
            frame = event_frame(xx, yy, pp, sensor_size=shape)
            frame=frame.permute(1,2,0)
        else:
            raise NotImplementedError(f'{convert_method} not implemented!')
        frames.append(copy.deepcopy(frame))
    frames = np.stack(frames)  # [N, H, W, 3]

    return frames  # [N, H, W, 3]