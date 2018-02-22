from keras.preprocessing.image import ImageDataGenerator


def train_rotated(model,rotated_model,x_train,y_train,x_test,y_test,classes,input_shape,batch_size,epochs,rotated_epochs):
    
    data_generator=ImageDataGenerator()
    train_dataset=data_generator.flow(x_train,y_train,shuffle=True,batch_size=batch_size)
    test_dataset=data_generator.flow(x_test,y_test,batch_size=batch_size)
    print("Training model with unrotated dataset...")
    model.fit_generator(train_dataset,
          steps_per_epoch=len(x_train) / batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=test_dataset)
     
    rotated_data_generator=ImageDataGenerator(rotation_range=180)
    rotated_train_dataset=rotated_data_generator.flow(x_train,y_train,shuffle=True,batch_size=batch_size)
    rotated_test_dataset=rotated_data_generator.flow(x_test,y_test,batch_size=batch_size)
    print("Training rotated model with rotated dataset...")
    rotated_model.fit_generator(rotated_train_dataset,
      steps_per_epoch=len(x_train) / batch_size,
      epochs=rotated_epochs,
      verbose=1,
      validation_data=rotated_test_dataset)
    
    print("Testing both models on both datasets...")
    scores={}
    scores["rotated_model_test_dataset"] = rotated_model.evaluate_generator(test_dataset)
    scores["model_test_dataset"] = model.evaluate_generator(test_dataset)
    scores["rotated_model_rotated_test_dataset"] = rotated_model.evaluate_generator(rotated_test_dataset)
    scores["model_rotated_test_dataset"] = model.evaluate_generator(rotated_test_dataset)
    return scores