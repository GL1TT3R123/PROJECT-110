# import the opencv library
import cv2
import numpy as np 
import tensorflow  as tf 

model = tf.keras.models.load_model("keras_model.h5")

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()

    #resize the image
    image=cv2.resize(frame,(224,224))

    #convert the image into the numpy array and increase dimensions
    testimage=np.array(image,dtype=np.float32)
    testimage=np.expand_dims(testimage,axis=0)

    #normalize the image
    normalizeImage= testimage/255.0

    prediction= model.predict(normalizeImage)
    print("prediction",prediction)
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()