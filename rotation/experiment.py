from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

def get_callbacks( no_improv_epochs = 10, min_delta = 1e-4 ):
    
    # Early stopping - End training early if we don't improve on the loss function
    # by a certain minimum threshold
    es = EarlyStopping( 'val_loss', patience = no_improv_epochs, 
                        mode = 'min', min_delta = min_delta )
    
    return [ es ]

def train_rotated(model,rotated_model,x_train,y_train,x_test,y_test,classes,input_shape,batch_size,epochs,rotated_epochs):
    
    data_generator=ImageDataGenerator()
    train_dataset=data_generator.flow(x_train,y_train,shuffle=True,batch_size=batch_size)
    test_dataset=data_generator.flow(x_test,y_test,batch_size=batch_size)
    print("Training model with unrotated dataset...")
    history=model.fit_generator(train_dataset,
          steps_per_epoch=len(x_train) / batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=test_dataset)
    callback_list = get_callbacks(no_improv_epochs=max(epochs//4,3))
    
    plot_history(history)
    
    rotated_data_generator=ImageDataGenerator(rotation_range=180)
    rotated_train_dataset=rotated_data_generator.flow(x_train,y_train,shuffle=True,batch_size=batch_size)
    rotated_test_dataset=rotated_data_generator.flow(x_test,y_test,batch_size=batch_size)
    print("Training rotated model with rotated dataset...")
    rotated_history=rotated_model.fit_generator(rotated_train_dataset,
      steps_per_epoch=len(x_train) / batch_size,
      epochs=rotated_epochs,
      verbose=1,callbacks=callback_list,
      validation_data=rotated_test_dataset)
    plot_history(rotated_history)
    print("Testing both models on both datasets...")
    scores={}
    scores["rotated_model_test_dataset"] = rotated_model.evaluate_generator(test_dataset)
    scores["model_test_dataset"] = model.evaluate_generator(test_dataset)
    scores["rotated_model_rotated_test_dataset"] = rotated_model.evaluate_generator(rotated_test_dataset)
    scores["model_rotated_test_dataset"] = model.evaluate_generator(rotated_test_dataset)
    return scores
def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
import matplotlib.pyplot as plt
import math

def visualize(images,labels,n_show,rotated=True):
    
    n,h,w,c=images.shape
    if rotated:
        data_generator=ImageDataGenerator(rotation_range=180)
    else:
        data_generator=ImageDataGenerator()
    batch=data_generator.flow(images,labels,batch_size=n_show)
    images_batch,labels_batch=batch[0]
    
    
    plots=math.ceil(math.sqrt(n_show))
    f, axarr = plt.subplots(plots,plots,figsize=(50,50), sharex=True)
    
    for i in range(n_show):
        j,k=i//plots,i%plots
        image=images_batch[i,:,:,:]
        if (c==1):
            image=image.squeeze(axis=-1)
        axarr[j,k].imshow(image)
        axarr[j,k].set_title("class %d" % labels_batch[i,:].argmax())
    plt.show()
    
    
    
    
    