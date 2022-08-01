"""" CNN """
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
import gc
import zMyDl_utils
import gdal
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import patheffects


""" setting """
np.set_printoptions(precision=2)
gc.collect()

plt.style.use("dark_background")
plt.xkcd()
plt.rcParams['path.effects'] = [patheffects.withStroke(linewidth=1)]
plt.rcParams['figure.facecolor'] = 'black'

""" paths """
path = "/media/user/DATA/Article/MyDlArticle/Dl_SatClassification13990403/"

ClpTrnAreaPth = "1- Clp train area"
ClpCsStdyPth = "1- Clp case study"

NDVIReClsTrnAreaPth = "3-1_NDVI_Re-Class_TrainArea/NDVI_Re-Class_TrainArea.tif"
NDVIReClsCsStdyPth = "3-1_NDVI_Re-Class_CaseStudy/NDVI_Re-class_CaseStudy.tif"

""" load and read Train area data """
ticTrnArea = time.time()

StckBandTrnArea = zMyDl_utils.load_sentinel2_img(path + ClpTrnAreaPth)

NDVITrnArea = np.ravel(gdal.Open(path + NDVIReClsTrnAreaPth).ReadAsArray())

xTrnTrnArea, xTstTrnArea, yTrnTrnArea, yTstTrnArea = \
    train_test_split(StckBandTrnArea, NDVITrnArea,
                     random_state=0,
                     stratify=NDVITrnArea)

""" pre-process """
xTrnTrnArea = xTrnTrnArea.reshape((xTrnTrnArea.shape[0], 12, 1, 1))
xTrnTrnArea = xTrnTrnArea.astype("float32")
xTstTrnArea = xTstTrnArea.reshape((xTstTrnArea.shape[0], 12, 1, 1))
xTstTrnArea = xTstTrnArea.astype("float32")

StckBandTrnArea = StckBandTrnArea.reshape((StckBandTrnArea.shape[0],
                                           12, 1, 1))

mu = np.mean(xTstTrnArea, axis=0)
xTrnTrnArea -= mu
xTstTrnArea -= mu

yTrnTrnArea = to_categorical(yTrnTrnArea)   # has 3 classes
# yTstTrnArea = to_categorical(yTstTrnArea)   # has 3 classes


""" NN multi-layers classifier """


def create_cnn():
    model = Sequential()


    """64 = number of filter 
    (3, 3) = filter size"""

    """ padding = (valid or same)
    same = output feature map with the same spatial dimension
    valid = (default)"""


    """ Conv Block 1 (input lyer) """
    model.add(Conv2D(8, (3, 3), padding="same",
                     input_shape=(12, 1, 1), activation="relu"))
    # model.add(Conv2D(64, (3, 3), activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 1)))
    model.add(Dropout(.25))

    """ Conv Block 2 (hidden lyer1)"""
    model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(128, (3, 3), activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 1)))
    model.add(Dropout(.25))

    """ Conv Block3 (hidden lyer2)"""
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(256, (3, 3), activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 1)))
    model.add(Dropout(0.25))

    """ Conv Block4 (hidden lyer3)"""
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(256, (3, 3), activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 1)))
    model.add(Dropout(0.25))

    """ Conv Block5 (hidden lyer4)"""
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(256, (3, 3), activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 1)))
    model.add(Dropout(0.25))

    """ classifier """
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(.25))
    model.add(Dense(4, activation="softmax"))

    print(model.summary())

    return model


Model = create_cnn()

optimizer = RMSprop(learning_rate=0.001)
# optimizer = Adam(learning_rate=0.001)
Model.compile(optimizer=optimizer, loss="categorical_crossentropy",
              metrics=["accuracy"])

# history = model.fit(xTrnTrnArea, yTrnTrnArea,
#                     batch_size=512,
#                     epochs=15,
#                     verbose=2,
#                     shuffle=True)

""" separate train and validation index """
num_train = int(xTrnTrnArea.shape[0] * 90 / 100)
history = Model.fit(xTrnTrnArea[: num_train], yTrnTrnArea[: num_train],
                    batch_size=64,
                    epochs=6,
                    verbose=2,
                    validation_data=(xTrnTrnArea[num_train:],
                                     yTrnTrnArea[num_train:]),
                    shuffle=True)

""" plot train area train and validation loss """
plt.figure(figsize=(12, 8))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="validation Loss")
plt.legend()
plt.title("CNN training Vs Validation loss - 4hiddenLyr"
          ", (8, 16, 32, 32, 32, 64), (3, 3), epoch=6, without MaxPool2D ")
plt.ylim(-0.01)
# plt.grid()
plt.show()
plt.grid()
plt.savefig(path + "zMapJPGExport/" +
            "10_6- CNN training Vs Validation loss -"
            "4hiddenLyr, (8, 16, 32, 32, 32, 64), (3, 3), "
            "epoch=6, without MaxPool2D .png",
            dpi=700)

""" plot train area train and validation accuracy """
# plt.figure(figsize=(12, 8))
# plt.plot(history.history["acc"], label="Training acc")
# plt.plot(history.history["val_acc"], label="validation acc")
# plt.legend()
# plt.title("training Vs Validation Accuracy")
# plt.grid()
# plt.ylim(-0.1)
# plt.show()


""" prediction """
yHtTrnArea = Model.predict_classes(xTstTrnArea)
Acc = 100 * np.mean(yHtTrnArea == yTstTrnArea)
print("\n CNN accuracy: %.2f \n" % Acc)

# yHtTrnAreaEval = Model.evaluate(xTstTrnArea, yTstTrnArea)
# print("\n CNN accuracy: %.2f \n" % Acc)


""" apply to train area """
predictTrnArea = Model.predict_classes(StckBandTrnArea)
# predictTrnArea = Model.predict(StckBandTrnArea)

unique, counts = np.unique(predictTrnArea, return_counts=True)
print("\n CNN for train area: \n",
      dict(zip(unique, counts)))


""" incorrect change value """
predictTrnAreaInc = predictTrnArea
predictTrnAreaInc[np.where(predictTrnArea != NDVITrnArea)] = 10

zMyDl_utils.export_output_data_set(path + ClpTrnAreaPth,
                                   "10_6- CNN_TrainAreaInc",
                                   "10_6- CNN_TrainAreaInc- 3hiddenLyr"
                                   ", (8, 16, 32, 32, 32, 64), (3, 3),  "
                                   "epoch=6,without MaxPool2D",
                                   predictTrnAreaInc)

"""" confusion matrix for train area """
cm = confusion_matrix(predictTrnArea, NDVITrnArea)
plt.figure(figsize=(12, 8))
plt.imshow(cm, interpolation="nearest")
plt.title("NN-singLayer confusion matrix for train area \n"
          "(rmsprop, 4hiddenLyr, (8, 16, 32, 32, 32, 64)"
          ", (3, 3), epoch=6, without MaxPool2D")
plt.savefig(path + "zMapJPGExport/" +
            "10_6- CNN confusion matrix for train area - "
            "4hiddenLyr, (8, 16, 32, 32, 32, 64), (3, 3),"
            " epoch=6, without MaxPool2D .png")


""" train area duration """
tocTrnArea = time.time()
du = tocTrnArea - ticTrnArea
unit = "Sec"
if du > 60:
    du = du / 60
    unit = "Min"
print("NN-multiLayer duration for train area: %.2f %s \n" % (du, unit))

""" del train area value"""
del du, unit
# del StckBandTrnArea, NDVITrnArea
# del xTrnTrnArea, xTstTrnArea, yTrnTrnArea, yTstTrnArea
