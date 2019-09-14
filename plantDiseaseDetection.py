import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


################ image to array ##############
def image_to_array_def(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, tuple((256, 256)))
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print("Error : {}".format(e))
        return None


directory_root = 'input/'

imgList = []
labelist = []


try:
    rootDir = listdir( directory_root+"/")
    for dir in rootDir:
        # remove .DS_Store from list
        if dir == ".DS_Store":
            rootDir.remove(dir)

    for plantFolder in rootDir:
        plantDiseaseFolderList = listdir(directory_root+"/"+plantFolder+"/")

        for diseaseFolder in plantDiseaseFolderList:
            # remove .DS_Store from list
            if diseaseFolder == ".DS_Store":
                plantDiseaseFolderList.remove( diseaseFolder )

        for plantDiseaseFolder in plantDiseaseFolderList:
            plantDiseaseImageList = listdir(directory_root+plantFolder+"/"+plantDiseaseFolder+"/")

            for singlePlantDiseaseImage in plantDiseaseImageList:
                if singlePlantDiseaseImage == ".DS_Store":
                    plantDiseaseImageList.remove( singlePlantDiseaseImage )

            for image in plantDiseaseImageList[:500]:
                imageDirectory = directory_root+plantFolder+"/"+plantDiseaseFolder +"/"+image

                if imageDirectory.endswith( ".jpg" ) == True or imageDirectory.endswith( ".JPG" ) == True:
                    imgList.append( image_to_array_def(imageDirectory) )
                    labelist.append(plantDiseaseFolder)
    ###### Image loading completed #####
except Exception as e:
    print( "Error : {}".format(e) )

image_size = len(imgList)
print(image_size)

labelB = LabelBinarizer()
imgLabels = labelB.fit_transform(labelist)
pickle.dump(labelB,open('label_transform.pkl', 'wb'))
numClasses = len(labelB.classes_)
print("num of classes {}".format(numClasses))
print("classes {}".format(labelB.classes_))

np_imgList = np.array(imgList, dtype=np.float16) / 225.0 # FROM [0,255] TO [0,1]

####Split the data to train, test####
x_train, x_test, y_train, y_test = train_test_split(np_imgList, imgLabels, test_size=0.2, random_state = 42)

#This allows us to use a smaller dataset and still achieve high results.
aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2,
    zoom_range=0.2,horizontal_flip=True,
    fill_mode="nearest")

model = Sequential()
inputShape = (256, 256, 3)
chanDim = -1
if K.image_data_format() == "channels_first":
    inputShape = (3, 256, 256)
    chanDim = 1
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(numClasses))
model.add(Activation("softmax"))
model.summary()

opt = Adam(lr=1e-3, decay=1e-3 / 25)
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the network
history = model.fit_generator(
    aug.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // 32,
    epochs=25, verbose=1
    )

trainAcc = history.history['trainAcc']
valAcc = history.history['valAcc']
trainLoss = history.history['trainLoss']
valLoss = history.history['valLoss']
epochs = range(1, len(trainAcc) + 1)


#accuracy
plt.plot(epochs, trainAcc, 'b', label='Training accurarcy')
plt.plot(epochs, valAcc, 'r', label='Validation accurarcy')
plt.title('accuracy')
plt.legend()
plt.figure()
#loss
plt.plot(epochs, trainLoss, 'b', label='Training loss')
plt.plot(epochs, valLoss, 'r', label='Validation loss')
plt.title('loss')
plt.legend()
plt.show()


scores = model.evaluate(x_test, y_test) #Calculating model accuracy
print("Test Accuracy: {}".format(scores[1]*100))

pickle.dump(model,open('cnn_model.pkl', 'wb')) # save the model in pkl file