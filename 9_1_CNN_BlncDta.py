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
from tensorflow.keras.models import load_model

""" setting """
np.set_printoptions(precision=2)
gc.collect()

plt.style.use("dark_background")
plt.xkcd()
plt.rcParams['path.effects'] = [patheffects.withStroke(linewidth=1)]
plt.rcParams['figure.facecolor'] = 'black'


""" paths """
path = "/media/kiarash/DATA/Article/MyDlArticle/DL_SatClassification_balanceData_13990501/"
BlnceDtaFldr = "1_1_balance_data"


"""" load and read balance data and NDVI Re-classed for CsStdySmll """
ticBlncDta = time.time()

BlncDta = np.load(path + BlnceDtaFldr + "/" + "balance data.npy")
BlncDta = BlncDta.astype("float32")

xBlncDta = BlncDta[:, :-1]

NDVITrn = BlncDta[:, -1:]
yBlncDta = NDVITrn.reshape(-1, 1)
del NDVITrn

""" split balance data to train and test """

xTrn, xTst, yTrn, yTst = train_test_split(xBlncDta,
                                          yBlncDta,
                                          random_state=0,
                                          stratify=yBlncDta)

""" pre-process """
xTrn = xTrn.reshape((xTrn.shape[0], 12, 1, 1))
xTrn = xTrn.astype("float32")
xTst = xTst.reshape((xTst.shape[0], 12, 1, 1))
xTst = xTst.astype("float32")

mu = np.mean(xTst, axis=0)
xTrn -= mu
xTst -= mu

yTrn = to_categorical(yTrn)   # has 3 classes
# yTstTrn = to_categorical(yTstTrn)   # has 3 classes


def create_cnn():
    model = Sequential()

    """32 = number of filter 
    (3, 3) = filter size"""

    """ padding = (valid or same)
    same = output feature map with the same spatial dimension
    valid = (default)"""

    """ Conv Block 1 (input layer) """
    model.add(Conv2D(8, (3, 3), padding="same",
                     input_shape=(12, 1, 1), activation="relu"))
    # model.add(Conv2D(32, (3, 3), activation="relu"))
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
    model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
    # model.add(Conv2D(256, (3, 3), activation="relu"))
    # model.add(MaxPool2D(pool_size=(2, 1)))
    model.add(Dropout(0.25))

    """ classifier """
    model.add(Flatten())
    model.add(Dense(8, activation="relu"))
    model.add(Dropout(.25))
    model.add(Dense(4, activation="softmax"))

    print(model.summary())

    return model


Model = create_cnn()

optimizer = RMSprop(learning_rate=0.001)
# optimizer = Adam(learning_rate=0.001)
Model.compile(optimizer=optimizer, loss="categorical_crossentropy",
              metrics=["accuracy"])


""" separate train and validation index """
num_train = int(xTrn.shape[0] * 90 / 100)
history = Model.fit(xTrn[: num_train], yTrn[: num_train],
                    batch_size=32,
                    epochs=6,
                    verbose=2,
                    validation_data=(xTrn[num_train:],
                                     yTrn[num_train:]),
                    shuffle=True)

""" plot train  train and validation loss """
plt.figure(figsize=(12, 8))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="validation Loss")
plt.legend()
plt.title("CNN training Vs Validation loss - 4hiddenLyr \n"
          ", (8, 16, 32, 32, 64, 64), (3, 3),\n epoch=50, without MaxPool2D ")
plt.ylim(-0.01)
# plt.grid()
plt.show()
plt.grid()
plt.savefig(path + "zMapJPGExport-(BlnceDta)/" +
            "10_1- CNN training Vs Validation loss -"
            "4hiddenLyr, (8, 16, 32, 32, 64, 64), (3, 3), "
            "epoch=50, without MaxPool2D .png",
            dpi=700)

""" plot train  train and validation accuracy """
# plt.figure(figsize=(12, 8))
# plt.plot(history.history["acc"], label="Training acc")
# plt.plot(history.history["val_acc"], label="validation acc")
# plt.legend()
# plt.title("training Vs Validation Accuracy")
# plt.grid()
# plt.ylim(-0.1)
# plt.show()


""" prediction """
yHtBlncDta = Model.predict_classes(xTst)

Acc = 100 * np.mean(yHtBlncDta.reshape(-1, 1) == yTst)
print("\n CNN accuracy: %.2f \n" % Acc)

unique, counts = np.unique(yHtBlncDta, return_counts=True)
print("\n CNN count for (BlnceDta): \n",
      dict(zip(unique, counts)))

""" replace incorrect classified value to 10 """
predictBlncDtaInc = yHtBlncDta.reshape(-1, 1).copy()
predictBlncDtaInc[np.where(yHtBlncDta.reshape(-1, 1) != yTst)] = 10

unique, counts = np.unique(predictBlncDtaInc, return_counts=True)
print("\n CNN count Incorrect CsStdySmll for balance data: \n",
      dict(zip(unique, counts)))


tocBlncDta = time.time()
du = tocBlncDta - ticBlncDta
unit = "sec"
if du > 60:
    unit = "Min"
    du = du / 60
print("CNN layer for balance data duration: \n%.2f " % du, unit)
print("***************************** \n")

del ticBlncDta, tocBlncDta, du, unit
del BlnceDtaFldr, BlncDta,  optimizer, \
    Acc, counts, history, Model, mu, num_train, \
    predictBlncDtaInc, unique, xTrn, xTst, \
    yHtBlncDta, yTrn, yTst

""" build Model for whole balance data """
""" prepare data"""

Model = create_cnn()

optimizer = RMSprop(learning_rate=0.001)
# optimizer = Adam(learning_rate=0.001)
Model.compile(optimizer=optimizer, loss="categorical_crossentropy",
              metrics=["accuracy"])

xBlncDta = xBlncDta.reshape((xBlncDta.shape[0], 12, 1, 1))
yBlncDta = to_categorical(yBlncDta)

Model.fit(xBlncDta, yBlncDta,
          batch_size=32,
          epochs=6,
          verbose=2,
          shuffle=True)
Model.save("CNN_Model-DrpOut.h5")

del xBlncDta, yBlncDta, Adam, RMSprop, optimizer, Model

""" **************************************************************** """
""" case study small """

""" case study path """
#
# ClpCsStdySmllPath = "1- Clp CsStdySmll"
# NDVIReClssCsStdySmllPath = "3-1_NDVI_Re-Class_CsStdySmll/NDVI_Re-Class_CsStdySmll.tif"
#
#
# ticCsStdSmll = time.time()
# """ load Case study small data """
# M = load_model("CNN_Model-DrpOut.h5")
# StckBandCsStdySmll = zMyDl_utils.load_sentinel2_img(path + ClpCsStdySmllPath)
# StckBandCsStdySmll = StckBandCsStdySmll.reshape((StckBandCsStdySmll.shape[0], 12, 1, 1))
#
# yCsStdySmll = np.ravel(gdal.Open(path + NDVIReClssCsStdySmllPath).ReadAsArray())
# yCsStdySmll1 = yCsStdySmll.copy()
# yCsStdySmll = to_categorical(yCsStdySmll)
#
#
# yHatCsStdySmll = M.predict_classes(StckBandCsStdySmll)
# unique, counts = np.unique(yHatCsStdySmll, return_counts=True)
# print("\n CNN CsStdySmll for balance data: \n",
#       dict(zip(unique, counts)))
#
# """ replace incorrect classified value to 10 """
# predictBlncDtaInc = yHatCsStdySmll.reshape(-1, 1).copy()
# predictBlncDtaInc[np.where(yHatCsStdySmll.reshape(-1, 1) != yCsStdySmll1.reshape(-1, 1))] = 10
#
# unique, counts = np.unique(predictBlncDtaInc, return_counts=True)
# print("\n CNN Incorrect CsStdySmll for balance data: \n",
#       dict(zip(unique, counts)))
#
# Acc = 100 * np.mean(yHatCsStdySmll.reshape(1, -1) == yCsStdySmll1.reshape((1, -1)))
# print("accuracy CNN for CsStdySmll:\n %.2f " % Acc)
#
# # loss, acc = M.evaluate(StckBandCsStdySmll, yCsStdySmll)
#
# """" write CNN for CsStdySmll balance data"""
# zMyDl_utils.export_output_data_set(path + ClpCsStdySmllPath,
#                                    "8-1_CNN_CsStdySmll(BlncDta)Drop-out=.2",
#                                    "CNN_CsStdySmll(BlncDta)Drop-out=.2",
#                                    predictBlncDtaInc)
#
#
# """" duration """
# tocCsStdySmll = time.time()
# du = tocCsStdySmll - ticCsStdSmll
# unit = "sec"
# if du > 60:
#     unit = "Min"
#     du = du / 60
# print("CNN layer for Case Study small balance data duration: \n %.2f " % du, unit)
# print("***************************** \n")
#
# del Acc, ClpCsStdySmllPath, M, NDVIReClssCsStdySmllPath, \
#     StckBandCsStdySmll, counts, du, predictBlncDtaInc, \
#     ticCsStdSmll, tocCsStdySmll, unique, unit, yCsStdySmll1, \
#     yHatCsStdySmll
#
#
#
#
# """ **************************************************************** """
# """ case study Big """
#
# """ case study path """
# ClpCsStdyBigPath = "1- Clp CsStdyBig"
# NDVIReClssCsStdyBigPath = "3-1_NDVI_Re-Class_CsStdyBig/NDVI_Re-Class_CsStdyBig.tif"
#
#
# ticCsStdBig = time.time()
# """ load Case study Big data """
# M = load_model("CNN_Model-DrpOut.h5")
# StckBandCsStdyBig = zMyDl_utils.load_sentinel2_img(path + ClpCsStdyBigPath)
# StckBandCsStdyBig = StckBandCsStdyBig.reshape((StckBandCsStdyBig.shape[0], 12, 1, 1))
#
# yCsStdyBig = np.ravel(gdal.Open(path + NDVIReClssCsStdyBigPath).ReadAsArray())
# yCsStdyBig1 = yCsStdyBig.copy()
# yCsStdyBig = to_categorical(yCsStdyBig)
#
#
# yHatCsStdyBig = M.predict_classes(StckBandCsStdyBig)
# unique, counts = np.unique(yHatCsStdyBig, return_counts=True)
# print("\n CNN-layers_DrpOut CsStdyBig for balance data: \n",
#       dict(zip(unique, counts)))
#
# """ replace incorrect classified value to 10 """
# predictBlncDtaInc = yHatCsStdyBig.reshape(-1, 1).copy()
# predictBlncDtaInc[np.where(yHatCsStdyBig.reshape(-1, 1) != yCsStdyBig1.reshape(-1, 1))] = 10
#
# unique, counts = np.unique(predictBlncDtaInc, return_counts=True)
# print("\n CNN-layers_DrpOut Incorrect CsStdyBig for balance data: \n",
#       dict(zip(unique, counts)))
#
# Acc = 100 * np.mean(yHatCsStdyBig.reshape(1, -1) == yCsStdyBig1.reshape((1, -1)))
# print("accuracy CNN-layers_DrpOut for CsStdyBig\n: %.2f " % Acc)
#
# # loss, acc = M.evaluate(StckBandCsStdyBig, yCsStdyBig)
#
# """" write CNN for CsStdyBig balance data"""
# zMyDl_utils.export_output_data_set(path + ClpCsStdyBigPath,
#                                    "8-2_CNN-layers_DrpOut_CsStdyBig(BlncDta)",
#                                    "CNN-layers_DrpOut_CsStdyBig(BlncDta)",
#                                    predictBlncDtaInc)
#
#
# """" duration """
# tocCsStdyBig = time.time()
# du = tocCsStdyBig - ticCsStdBig
# unit = "sec"
# if du > 60:
#     unit = "Min"
#     du = du / 60
# print("CNN-layers Drop out for Case Study Big balance data duration:\n %.2f " % du, unit)
# print("***************************** \n")
