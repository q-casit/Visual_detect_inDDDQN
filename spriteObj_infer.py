'''
Created on 2019-5-10

this codes is used for SSD  to detect sprite object coordinates

ssd_libs refers from SSD-TensorFlow model which made by Lukasz Janyst

but in DDDQN object detection, does not use ssd by default though 


'''

import argparse
import pickle
import math
import sys
import cv2
import os

import tensorflow as tf
import numpy as np

from ssdutils import get_anchors_for_preset, decode_boxes, suppress_overlaps
from ssdvgg import SSDVGG
from utils import str2bool, load_data_source, draw_box
from tqdm import tqdm


def sample_generator(samples, image_size, batch_size):
    image_size = (image_size.w, image_size.h)
    for offset in range(0, len(samples), batch_size):
        files = samples[offset:offset+batch_size]
        images = []
        idxs   = []
        for i, image_file in enumerate(files):
            image = cv2.resize(cv2.imread(image_file), image_size)
            images.append(image.astype(np.float32))
            idxs.append(offset+i)
        yield np.array(images), idxs
        

def ssd_detect_bbox(train_data_fp,project_name,checkpoint_path, Img_files,batch_size):
    
    files = []
    batch_size = 1 # each time process 1 image
    threshold = 0.5
    
    files = Img_files # only 1 image
    bbox = []
    
    #---------------------------------------------------------------------------
    # Check if we can get the checkpoint
    #---------------------------------------------------------------------------
    state = tf.train.get_checkpoint_state(project_name)
    if state is None:
        print('[!] No network state found in ' + project_name)
        return 1

    try:
        checkpoint_file = state.all_model_checkpoint_paths[checkpoint_path]
    except IndexError:
        print('[!] Cannot find checkpoint ' + str(checkpoint_file))
        return 1

    metagraph_file = checkpoint_file + '.meta'

    if not os.path.exists(metagraph_file):
        print('[!] Cannot find metagraph ' + metagraph_file)
        return 1
    
    #---------------------------------------------------------------------------
    # Load the training data
    # like 'pascal-voc/training-data.pkl'
    #---------------------------------------------------------------------------
    try:
        with open(train_data_fp, 'rb') as f:
            data = pickle.load(f)
        preset = data['preset']
        colors = data['colors']
        lid2name = data['lid2name']
        num_classes = data['num-classes']
        image_size = preset.image_size
        anchors = get_anchors_for_preset(preset)
    except (FileNotFoundError, IOError, KeyError) as e:
        print('[!] Unable to load training data:', str(e))
        return 1
    
    #### detect obj using ssd model ----------
    with tf.Session() as sess:
        print('[i] Creating the model...')
        net = SSDVGG(sess, preset)
        net.build_from_metagraph(metagraph_file, checkpoint_file)

        #-----------------------------------------------------------------------
        # Process the images
        #-----------------------------------------------------------------------
        generator = sample_generator(files, image_size, batch_size)
        n_sample_batches = int(math.ceil(len(files)/batch_size))
        description = '[i] Processing samples'

        for x, idxs in tqdm(generator, total=n_sample_batches,
                      desc=description, unit='batches'):
            feed = {net.image_input:  x,
                    net.keep_prob:    1}
            enc_boxes = sess.run(net.result, feed_dict=feed)

            #-------------------------------------------------------------------
            # Process the predictions
            #-------------------------------------------------------------------
            for i in range(enc_boxes.shape[0]):
                boxes = decode_boxes(enc_boxes[i], anchors, threshold,
                                     lid2name, None)
                boxes = suppress_overlaps(boxes)[:200]
#                 filename = files[idxs[i]]
#                 basename = os.path.basename(filename)
    
    bbox =  boxes
    return  bbox        
    
    
        
    
