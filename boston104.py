import metadata
import os

import numpy as np
import skimage.io as io

import skimage
import tensorflow as tf
from keras import backend as K
from keras import metrics
import metadata



class Boston104LocalizationIterator(tf.keras.preprocessing.image.Iterator ):
    """Iterator yielding data from a Numpy array.
    # Arguments
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self, basepath,video_positions_filepath,localization_target,
                 batch_size=32,shuffle=False, seed=None):
        self.localization_target=localization_target
        
        self.basepath=basepath
        self.video_positions_filepath=video_positions_filepath
        self.images_path=os.path.join(basepath,'png-segments')
        self.frames=metadata.parse_videos_to_images(self.video_positions_filepath,self.images_path,ignore_images_with_outofbounds_positions=True)
        
        #self.image_data_generator = image_data_generator
        # Remove the useless text in the bottom and adjust positions accordingly
        self.sub_image=np.array([0,240,10,326]) # remove borders and stuff
        self.h,self.w=(self.sub_image[1]-self.sub_image[0],self.sub_image[3]-self.sub_image[2])
        for frame in self.frames:
            for (k,p) in frame.positions.items():
                adjusted_position=p-np.array([self.sub_image[0],self.sub_image[2]])
                frame.positions[k]=adjusted_position
                
        filter_function=lambda f: not self.has_outofbounds_position(f,(self.h,self.w))
        print("Loaded %d frames. " % len(self.frames))
        self.frames=list(filter(filter_function, self.frames))
        print("After filtering out of bounds positions, we have %d frames remaining. " % len(self.frames))
        
        #super(Boston104LocalizationIterator, self).__init__(len(self.frames), batch_size, shuffle, seed)
        super().__init__(len(self.frames), batch_size, shuffle, seed)
        
        
    def has_outofbounds_position(self,frame,image_size):
        result=False
        for body_part in self.localization_target.body_parts:
            p=frame.positions[body_part]
            if not(0<=p[0]<image_size[0] and 0<=p[1]<image_size[1]):
                    result=True
                    break;
        return result 

    def _get_batches_of_transformed_samples(self, index_array):
        #print(index_array)
        batch_x,batch_y=self.read_boston104_frames(index_array[0])
        return batch_x, batch_y

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)
    

    
    def read_boston104_frames(self,frame_indices):
        
        n=len(frame_indices)
        x=np.zeros((n,self.h,self.w,1))
        image_shape=np.array([self.h,self.w])
        y_dim=self.localization_target.dims()
        y=np.zeros((n,y_dim))
        
        w=self.sub_image
        for (i,j) in enumerate(frame_indices):
            frame=self.frames[j]
            image=io.imread(frame.path)
            image=image[w[0]:w[1],w[2]:w[3],:]
            image=skimage.color.rgb2grey(image)
#             image = self.image_data_generator.random_transform(image.astype(K.floatx()))
#             image = self.image_data_generator.standardize(image)
            x[i,:,:,0]=image
            y[i,:]=self.localization_target.frame_to_target(frame,image_shape)

            
        return x,y
