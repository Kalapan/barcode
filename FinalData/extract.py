from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

class extraction:
    #Takes image as input and returns deep feature
    def __init__(self):
        #Read model
        base_model = VGG16(weights="imagenet")
        self.model =Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)
        pass

    def getInfo(self,img):
        img = img.resize((224,224)).convert("RGB")
        x=image.img_to_array(img) #conver to numpy array
        x=np.expand_dims(x,axis=0) # (H,W,C) -> (1,H,W,C)
        x=preprocess_input(x)
        feature = self.model.predict(x)[0] # (1,4096) -> (4096) pass image to model
        return feature / np.linalg.norm(feature) #normalize and return



