import os.path as op
import os
import pickle as pkl
import numpy as np

data_dir='/tsukimi/datasets/MVSEC/event_chunks_10t'
dest_dir='/tsukimi/datasets/MVSEC/event_chunks_imgbind'

def chunk(data_path):
    # load pickle
    with open(data_path, 'rb') as f:
        data_packet = pkl.load(f)
    
    filename=data_path.split('/')[-1].split('.')[0]
    
    count=0
    # dump 2 images with 1 events stream as a pickle
    for i in range(len(data_packet['images'])-1):
        images=data_packet['images'][i:i+2]
        events=data_packet['events'][i]
        sequence={
            'images':images,
            'events':events
        }
        with open(op.join(dest_dir, f'{filename}-{count}.pkl'), 'wb') as fo:
            pkl.dump(sequence, fo)
        count+=1

if __name__ == '__main__':
    # Iterate through the files in the directory
    length=len(os.listdir(data_dir))
    count=0
    for file in os.listdir(data_dir):
        if file.endswith('.pkl'):
            print(f'Processing {count}/{length}', end='\r')
            chunk(os.path.join(data_dir, file))
            count+=1
        