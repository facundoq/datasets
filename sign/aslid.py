import os
import numpy as np
    

#dataset: 'train' or 'test'
#basepath: of extracted zip 
def read_all(basepath,dataset):
    
    def read_image_filenames(annotations_filepath):
        import csv
        f = open(annotations_filepath, 'r')
        annotations_reader = csv.reader(f, delimiter=',')
        return [row[0] for row in annotations_reader]
    
    
    
    annotations_filepath=os.path.join(basepath,"%s.csv" %dataset)

    positions=np.loadtxt(annotations_filepath,delimiter=',',usecols=list(range(1,15)))
    n,m=positions.shape
    positions=positions.reshape(n,m//2,2)
    image_filenames=read_image_filenames(annotations_filepath)

    samples=[]
    images=[]
    for i in range(positions.shape[0]):
        sample_positions=positions[i,:,:]
        named_positions={}
        for j in range(sample_positions.shape[0]):
            named_positions[body_parts[j]]=sample_positions[j,:]
        sample=ASLIDImage(image_filenames[i],named_positions)
        samples.append(sample)

        image_filepath=os.path.join(basepath,dataset,sample.filepath)
        image=skimage.io.imread(image_filepath)
        images.append(image)
    return samples,images




