

import os

#basepath='/media/facundoq/Seagate Expansion Drive/ffff/rwth-tracking'
basepath='/data/datasets/test_rwth'
files=os.listdir(basepath)
files.sort()
current_folderpath=''
for (i,f) in enumerate(files):
  fullpath=os.path.join(basepath,f)
  print(fullpath)
  if os.path.isfile(fullpath) and f.endswith('png'):
    basename=os.path.basename(f)
    name,extension=os.path.splitext(basename)
    separator_index=basename.rfind("_")
    folder,frame=basename[:separator_index],basename[separator_index:]
    folder,_=os.path.splitext(folder)

    print(folder)
    print(frame)
    folderpath=os.path.join(basepath,folder)
    if not (folderpath==current_folderpath):
      current_folderpath=folderpath
      os.mkdir(current_folderpath)
    destination_fullpath=os.path.join(current_folderpath,frame)
    os.rename(fullpath,destination_fullpath)
  
