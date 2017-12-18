import metadata
import os

import numpy as np
import skimage.io as io

import skimage
import tensorflow as tf
from keras import backend as K
from keras import metrics
import metadata

from abc import ABC, abstractmethod

class LocalizationTarget(ABC):
    def __init__(self,body_parts):
        self.body_parts=sorted(body_parts)
    @abstractmethod
    def dims(self):
        pass
    @abstractmethod
    def frame_to_target(self,frame):
        pass
    @abstractmethod
    def loss(self,y,x):
        pass
    @abstractmethod
    def metric(self,y,x):
        pass
    @abstractmethod
    def predictions_to_body_parts_coordinates(self,predictions,image_shape):
        pass

class LocalizationTargetRegression(LocalizationTarget):
    
    
    def __init__(self,max_distance,body_parts):
        self.max_distance=max_distance
        super().__init__(body_parts)
        self.y_dim_per_body_part=2
        self.n_body_parts=len(self.body_parts)
        
    def dims(self):
        return self.y_dim_per_body_part*self.n_body_parts
        

    def frame_to_target(self,frame,image_shape):
        y=np.zeros((self.n_body_parts,self.y_dim_per_body_part))
        for i,body_part in enumerate(self.body_parts):
            image_position=frame.positions[body_part]
            y[i,:]=image_position
        return y.reshape(np.prod(y.shape))
    def loss(self,y_true,y_pred):
        delta=(y_true-y_pred)**2
        return K.sum(delta,axis=1)
    def metric(self,y_true,y_pred):
        
        distance=K.sqrt(self.loss(y_true,y_pred))
        in_radius=K.less_equal(distance, self.max_distance)
        return K.cast(in_radius,K.floatx())
    def predictions_to_body_parts_coordinates(self,predictions,image_shape):
        for i in range(len(image_shape)):
            predictions[predictions[:,i]<0]=0
            predictions[predictions[:,i]>=image_shape[i]]=image_shape[i]-1
        body_parts_coordinates={}
        for bp in range(self.n_body_parts):
            start=bp*self.y_dim_per_body_part
            end=start+self.y_dim_per_body_part
            coordinates=predictions[:,start:end]
            body_parts_coordinates[self.body_parts[bp]]=coordinates
        return body_parts_coordinates
        
class LocalizationTargetGrid(LocalizationTarget):
    
    
    def __init__(self,localization_grid_shape,body_parts):
        self.localization_grid_shape=localization_grid_shape
        super().__init__(body_parts)
        
        
    def dims(self):
        y_dim_per_body_part=np.prod(self.localization_grid_shape)
        n_body_parts=len(self.body_parts)
        y_dim=y_dim_per_body_part*n_body_parts
        return y_dim

    def frame_to_target(self,frame,image_shape):
        n_body_parts=len(self.body_parts)
        y_dim_per_body_part=np.prod(self.localization_grid_shape)
        sigma=1
        y=np.zeros((n_body_parts,self.localization_grid_shape[0],self.localization_grid_shape[1]))
        bp=0
        for body_part in sorted(self.body_parts):
            image_position=frame.positions[body_part]
            grid_position=self.image_position_to_grid_position(self.localization_grid_shape,image_shape,image_position)
            xs,ys = np.meshgrid(range(self.localization_grid_shape[1]),range(self.localization_grid_shape[0]))
            d=(xs-grid_position[1])**2+(ys-grid_position[0])**2
            y[bp,:,:]=np.exp(-( d / ( 2.0 * sigma**2 ) ) )
            y[bp,:,:]/=np.sum(y[bp,:,:])

            bp+=1
        y=np.reshape(y,np.prod(y.shape))
        return y
    def loss(self,y_true,y_pred):
        n_body_parts=len(self.body_parts)
        y_dim_per_body_part=np.prod(self.localization_grid_shape)
        cs=[]
        for i in range(n_body_parts):
            start=i*y_dim_per_body_part
            end=start+y_dim_per_body_part
            body_part_obj=K.categorical_crossentropy(y_true[:,start:end],y_pred[:,start:end])
            cs.append(body_part_obj)
            #print(start,end,body_part_obj,"\n")
        obj=cs[0]
        for i in range(1,n_body_parts):
            obj=obj+cs[i]
        return obj/n_body_parts    
    def metric(self,y_true,y_pred):
        n_body_parts=len(self.body_parts)
        y_dim_per_body_part=np.prod(self.localization_grid_shape)
        cs=[]
        for i in range(n_body_parts):
            start=i*y_dim_per_body_part
            end=start+y_dim_per_body_part
            body_part_accuracy=metrics.categorical_accuracy(y_true[:,start:end],y_pred[:,start:end])
            cs.append(body_part_accuracy)
            #print(start,end,body_part_obj,"\n")
        acc=cs[0]
        for i in range(1,n_body_parts):
            acc=acc+cs[i]
        return acc/n_body_parts
    def image_position_to_grid_position(self,grid_shape,image_shape,image_position):
        ratio= np.float64(grid_shape/image_shape)
        grid_position=np.floor(image_position*ratio)
#         print(image_shape," -> ",grid_shape," => ratio = ",ratio)
#         print(image_position," -> ",image_position*ratio," -> ",grid_position,"\n")
        return grid_position.astype(int)
    def grid_position_to_image_position(self,image_shape,grid_position):
        ratio=np.array(image_shape)/np.array(self.localization_grid_shape)
        image_position=np.round(np.array(grid_position)*ratio)
        
        return image_position.astype(int)
    
    def predictions_to_coordinates(self,predictions,image_shape):
        n,classes=predictions.shape
        grid_shape=self.localization_grid_shape
        assert classes==grid_shape[0]*grid_shape[1]
        coordinates=np.zeros((n,2))
        linear_positions=np.argmax(predictions,axis=1)
        for i in range(n):
            linear_position =linear_positions[i]
            r,c=np.unravel_index(linear_position, grid_shape)
            #grid_position = (linear_position // grid_shape[0],linear_position % grid_shape[1])
            grid_position=np.array([r,c])
            coordinates[i,:] = self.grid_position_to_image_position(image_shape,grid_position)
        return coordinates
    
    def predictions_to_body_parts_coordinates(self,predictions,image_shape):
        y_dim_per_body_part=np.prod(self.localization_grid_shape)
        
        body_parts_coordinates={}
        for i,bp in enumerate(sorted(self.body_parts)):
            start=i*y_dim_per_body_part
            end=start+y_dim_per_body_part
            coordinates=self.predictions_to_coordinates(predictions[:,start:end],image_shape)
            body_parts_coordinates[bp]=coordinates
        return body_parts_coordinates

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
