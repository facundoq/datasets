# Code for loading the RWTH-BOSTON-104 sign language dataset

import collections
import os
import numpy as np

VideoFrame= collections.namedtuple('VideoFrame', ['video','frame','positions','path'])
VideoPositions = collections.namedtuple('VideoPositions', ['id', 'positions'])
Point = collections.namedtuple('Point', ['x', 'y'])



def parse_videos_to_images(video_positions_filepath,images_path,ignore_images_with_outofbounds_positions=False):
    from xml.dom import minidom
    image_size=(336,312)
    xmldoc = minidom.parse(video_positions_filepath)
    video_elements = xmldoc.getElementsByTagName('video')
    video_frame_positions=[]
    for video_element in video_elements:
        video_id = int(video_element.attributes['name'].value)
        frames =video_element.getElementsByTagName('frame')
        keys=['left_hand','right_hand','head']
        positions={}
        for k in keys:
            positions[k]=[]
        for (frame_id,frame_element) in enumerate(frames):
            point_elements=frame_element.getElementsByTagName('point')
            frame_positions={}
            out_of_bounds=False
            for point in point_elements:
                key_index=int(point.attributes['n'].value)
                k = keys[key_index]
                x=int(point.attributes['x'].value)
                y=int(point.attributes['y'].value)
                p=np.array([y,x])
                positions[k].append(p)
                frame_positions[k]=p
                filename="frame%s_cam0.png" % str(frame_id).zfill(6)
                path=os.path.join(images_path,str(video_id).zfill(3),filename)
                if not (0<=p[0]<image_size[0] and 0<=p[1]<image_size[1]):
                    out_of_bounds=True
            if not (out_of_bounds and ignore_images_with_outofbounds_positions):
                video_frame_positions.append(VideoFrame(video_id,frame_id,frame_positions,path))

    return video_frame_positions

def parse_videos(video_positions_filepath):
    from xml.dom import minidom
    xmldoc = minidom.parse(video_positions_filepath)
    video_elements = xmldoc.getElementsByTagName('video')
    video_positions=[]
    keys=['head','left_hand','right_hand']
    for video_element in video_elements:
        video_id = int(video_element.attributes['name'].value)
        frames =video_element.getElementsByTagName('frame')
        for (frame_id,frame_element) in enumerate(frames):
            positions={}
            for k in keys:
                positions[k]=[]
            point_elements=frame_element.getElementsByTagName('point')
            for point in point_elements:
                key_index=int(point.attributes['n'].value)
                k = keys[key_index]
                x=int(point.attributes['x'].value)
                y=int(point.attributes['y'].value)
                p=np.array([y,x])
                positions[k].append(p)
        video_positions.append(VideoPositions(video_id,positions))
    return video_positions
