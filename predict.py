import numpy as np
import pickle
import cv2
from keras.preprocessing.image import img_to_array
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True)
args = vars(ap.parse_args())

model = pickle.loads(open("cnn_model.pkl","rb").read())
label = pickle.loads(open("label_transform.pkl","rb").read())


image = cv2.imread(args["image"])
image = cv2.resize(image,tuple((256, 256)))
imageToArray=img_to_array(image)
imgdim=np.expand_dims(imageToArray,axis=0)

prediction = model.predict(imgdim)
data = label.inverse_transform(prediction)[0]

print("prediction : {}".format(prediction))
print("Class : {}".format(data))