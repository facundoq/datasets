import collections 
import os
import skimage.io as io
ImageMetadata = collections.namedtuple('ImageMetadata', ['filename', 'objects'])
ObjectMetadata = collections.namedtuple('ObjectMetadata', ['id', 'type','confidence','bbox'])

def file_to_lines(filepath):
    with open(filepath) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content] 
    return content

def get_object_types():
    object_types={0:"fully-visible pedestrian",
                1:"bicyclist",
                2:"motorcyclist",
                10:"pedestrian group",
                255:"partially visible pedestrian, bicyclist, motorcyclist",}
    return object_types
def read_image_metadata(metadata_filepath):
    image_metadata=[]
    lines=file_to_lines(metadata_filepath)
    lines=lines[4:]
    i=0

    while i<len(lines):
        assert lines[i]==";"
        image_filename=lines[i+1]
        #image_resolution=lines[i+2]
        zero_str,object_count_str=lines[i+3].split(" ")
        assert zero_str=="0"
        object_count=int(object_count_str)
        objects=[]
        i=i+4
        for j in range(object_count):
            garbage,object_type_str=lines[i].split(" ")
            object_type=int(object_type_str)
            object_id_str,uniqueid=lines[i+1].split(" ")
            object_id=int(object_id_str)
            confidence=float(lines[i+2])
            bbox= [float(x) for x in list(lines[i+3].split(" "))]
            objects.append(ObjectMetadata(object_id,object_type,confidence,bbox))
            i+=5
        image_metadata.append(ImageMetadata(image_filename,objects))
        
    
    return image_metadata,get_object_types()

def read_image_directory(folderpath):
#     images=[]
    images = io.imread_collection(folderpath+'/*.pgm')
    return images
                                  
def get_dataset(basepath):
    metadata_filepath=os.path.join(basepath,"GroundTruth/GroundTruth2D.db")
    test_metadata,object_types=read_image_metadata(metadata_filepath)
    
    train_pedestrian_folderpath=os.path.join(basepath,"Data/TrainingData/Pedestrians/48x96/")
    train_non_pedestrian_folderpath=os.path.join(basepath,"Data/TrainingData/NonPedestrians/")
    test_folderpath=os.path.join(basepath,"Data/TestData")
    
    train_pedestrian=read_image_directory(train_pedestrian_folderpath)
    train_non_pedestrian=read_image_directory(train_non_pedestrian_folderpath)
    
    return train_pedestrian,train_non_pedestrian,test_metadata,object_types
    
    