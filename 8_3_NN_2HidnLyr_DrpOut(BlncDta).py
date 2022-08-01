""" NN - 2hidden-layers (Drop-out=.2)"""
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
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
path = "/media/user/DATA/Article/MyDlArticle/DL_SatClassification_balanceData_13990501/"
BlnceDtaFldr = "1_1_balance_data"


"""" load and read balance data and NDVI Re-classed for CsStdySmll """
ticBlncDta = time.time()

BlncDta = np.load(path + BlnceDtaFldr + "/" + "balance data.npy")
BlncDta = BlncDta.astype("float32")

xBlncDta = BlncDta[:, :-1]

NDVITrnArea = BlncDta[:, -1:]
yBlncDta = NDVITrnArea.reshape(-1, 1)
del NDVITrnArea

""" split balance data to train and test """

xTrn, xTst, \
yTrn, yTst = train_test_split(xBlncDta,
                              yBlncDta,
                              random_state=0,
                              stratify=yBlncDta)

mu = np.mean(xTst, axis=0)
xTrn -= mu
xTst -= mu

yTrn = to_categorical(yTrn)

""" NN multi-layers classifier """
model = Sequential()
""""First hidden Layer """
model.add(Dense(units=100, activation="relu", input_shape=(12,)))
model.add(Dropout(.2))
""" second hidden layer"""
model.add(Dense(units=100, activation="relu"))
model.add(Dropout(.2))

""" output layer """
model.add(Dense(units=4, activation="softmax"))
print(model.summary())

# optimizer = RMSprop(learning_rate=0.001)
optimizer = Adam(learning_rate=0.02, decay=1e-6)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",
              metrics=["accuracy"])


num_train = int(xTrn.shape[0] * 90 / 100)    # separate train and validation index
history = model.fit(xTrn[: num_train], yTrn[: num_train],
                    batch_size=32,
                    epochs=50,
                    verbose=2,
                    validation_data=(xTrn[num_train:],
                                     yTrn[num_train:]),
                    shuffle=True)

""" plot train area train and validation loss """
plt.figure(figsize=(12, 8))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="validation Loss")
plt.legend()
plt.title("NN-2hidden layer layer training Vs Validation loss \n"
          "(Adam, unit=100, 4, epoch=50, Drop-out=.2) \n"
          "activation=relu, softmax ")
plt.ylim(-0.01)
plt.grid()
plt.show()
plt.savefig(path + "zMapJPGExport-(BlnceDta)/" +
            "9-3- NN-2hidden layer training Vs Validation loss "
            "(Adam, unit=100, 4 epoch=50)"
            "activation=relu, softmax, Drop-out=.2 .png", dpi=700)

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
yHtBlncDta = model.predict_classes(xTst)

Acc = 100 * np.mean(yHtBlncDta.reshape(-1, 1) == yTst)
print("\n NN-2hidden-layers accuracy: %.2f \n" % Acc)

unique, counts = np.unique(yHtBlncDta, return_counts=True)
print("\n NN-2hidden-layersr count for (BlnceDta): \n",
      dict(zip(unique, counts)))

""" replace incorrect classified value to 10 """
predictBlncDtaInc = yHtBlncDta.reshape(-1, 1).copy()
predictBlncDtaInc[np.where(yHtBlncDta.reshape(-1, 1) != yTst)] = 10

unique, counts = np.unique(predictBlncDtaInc, return_counts=True)
print("\n NN-2hidden-layersr count Incorrect CsStdySmll for balance data: \n",
      dict(zip(unique, counts)))


tocBlncDta = time.time()
du = tocBlncDta - ticBlncDta
unit = "sec"
if du > 60:
    unit = "Min"
    du = du / 60
print("NN-2hidden-layersr layer for balance data duration: \n%.2f " % du, unit)
print("***************************** \n")

del ticBlncDta, tocBlncDta, du, unit
del BlnceDtaFldr, BlncDta,  \
    Acc, counts, history, model, mu, num_train, \
    predictBlncDtaInc, unique, xTrn, xTst, \
    yHtBlncDta, yTrn, yTst

""" build Model for whole balance data """
""" prepare data"""


""" NN multi-layers classifier """
model = Sequential()
""""First hidden Layer """
model.add(Dense(units=100, activation="relu", input_shape=(12,)))
model.add(Dropout(.2))
""" second hidden layer"""
model.add(Dense(units=100, activation="relu"))
model.add(Dropout(.2))
""" output layer """
model.add(Dense(units=4, activation="softmax"))
print(model.summary())

# optimizer = RMSprop(learning_rate=0.001)
optimizer = Adam(learning_rate=0.02, decay=1e-6)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",
              metrics=["accuracy"])

yBlncDta = to_categorical(yBlncDta)
model.fit(xBlncDta, yBlncDta,
          batch_size=32,
          epochs=50,
          verbose=2,
          shuffle=True)

model.save("MyNN_2HiddenLyr_Model-DrpOut.h5")

del xBlncDta, yBlncDta, Adam, RMSprop


""" **************************************************************** """
""" case study small """

""" case study path """
ClpCsStdySmllPath = "1- Clp CsStdySmll"
NDVIReClssCsStdySmllPath = "3-1_NDVI_Re-Class_CsStdySmll/NDVI_Re-Class_CsStdySmll.tif"


ticCsStdSmll = time.time()
""" load Case study small data """
M = load_model("MyNN_2HiddenLyr_Model-DrpOut.h5")
StckBandCsStdySmll = zMyDl_utils.load_sentinel2_img(path + ClpCsStdySmllPath)
yCsStdySmll = np.ravel(gdal.Open(path + NDVIReClssCsStdySmllPath).ReadAsArray())
yCsStdySmll1 = yCsStdySmll.copy()
yCsStdySmll = to_categorical(yCsStdySmll)


yHatCsStdySmll = M.predict_classes(StckBandCsStdySmll)
unique, counts = np.unique(yHatCsStdySmll, return_counts=True)
print("\n NN-2hidden CsStdySmll for balance data: \n",
      dict(zip(unique, counts)))

""" replace incorrect classified value to 10 """
predictBlncDtaInc = yHatCsStdySmll.reshape(-1, 1).copy()
predictBlncDtaInc[np.where(yHatCsStdySmll.reshape(-1, 1) != yCsStdySmll1.reshape(-1, 1))] = 10

unique, counts = np.unique(predictBlncDtaInc, return_counts=True)
print("\n NN-2hidden Incorrect CsStdySmll for balance data: \n",
      dict(zip(unique, counts)))

Acc = 100 * np.mean(yHatCsStdySmll.reshape(1, -1) == yCsStdySmll1.reshape((1, -1)))
print("accuracy NN-2hidden for CsStdySmll:\n %.2f " % Acc)

# loss, acc = M.evaluate(StckBandCsStdySmll, yCsStdySmll)

"""" write logistc regression for CsStdySmll balance data"""
zMyDl_utils.export_output_data_set(path + ClpCsStdySmllPath,
                                   "7-3_NN-2hidden_CsStdySmll(BlncDta)Drop-out=.2",
                                   "NN-2hidden_CsStdySmll(BlncDta)Drop-out=.2",
                                   predictBlncDtaInc)


"""" duration """
tocCsStdySmll = time.time()
du = tocCsStdySmll - ticCsStdSmll
unit = "sec"
if du > 60:
    unit = "Min"
    du = du / 60
print("NN-single layer for Case Study small balance data duration: \n %.2f " % du, unit)
print("***************************** \n")

del Acc, ClpCsStdySmllPath, M, NDVIReClssCsStdySmllPath, \
    StckBandCsStdySmll, counts, du, predictBlncDtaInc, \
    ticCsStdSmll, tocCsStdySmll, unique, unit, yCsStdySmll1, \
    yHatCsStdySmll




""" **************************************************************** """
""" case study Big """

""" case study path """
ClpCsStdyBigPath = "1- Clp CsStdyBig"
NDVIReClssCsStdyBigPath = "3-1_NDVI_Re-Class_CsStdyBig/NDVI_Re-Class_CsStdyBig.tif"


ticCsStdBig = time.time()
""" load Case study Big data """
M = load_model("MyNN_2HiddenLyr_Model-DrpOut.h5")
StckBandCsStdyBig = zMyDl_utils.load_sentinel2_img(path + ClpCsStdyBigPath)
yCsStdyBig = np.ravel(gdal.Open(path + NDVIReClssCsStdyBigPath).ReadAsArray())
yCsStdyBig1 = yCsStdyBig.copy()
yCsStdyBig = to_categorical(yCsStdyBig)


yHatCsStdyBig = M.predict_classes(StckBandCsStdyBig)
unique, counts = np.unique(yHatCsStdyBig, return_counts=True)
print("\n NN-2hidden-layers_DrpOut CsStdyBig for balance data: \n",
      dict(zip(unique, counts)))

""" replace incorrect classified value to 10 """
predictBlncDtaInc = yHatCsStdyBig.reshape(-1, 1).copy()
predictBlncDtaInc[np.where(yHatCsStdyBig.reshape(-1, 1) != yCsStdyBig1.reshape(-1, 1))] = 10

unique, counts = np.unique(predictBlncDtaInc, return_counts=True)
print("\n NN-2hidden-layers_DrpOut Incorrect CsStdyBig for balance data: \n",
      dict(zip(unique, counts)))

Acc = 100 * np.mean(yHatCsStdyBig.reshape(1, -1) == yCsStdyBig1.reshape((1, -1)))
print("accuracy NN-2hidden-layers_DrpOut for CsStdyBig\n: %.2f " % Acc)

# loss, acc = M.evaluate(StckBandCsStdyBig, yCsStdyBig)

"""" write logistc regression for CsStdyBig balance data"""
zMyDl_utils.export_output_data_set(path + ClpCsStdyBigPath,
                                   "7-3_NN-2hidden-layers_DrpOut_CsStdyBig(BlncDta)",
                                   "NN-2hidden-layers_DrpOut_CsStdyBig(BlncDta)",
                                   predictBlncDtaInc)


"""" duration """
tocCsStdyBig = time.time()
du = tocCsStdyBig - ticCsStdBig
unit = "sec"
if du > 60:
    unit = "Min"
    du = du / 60
print("NN-2hidden-layersr Drop out for Case Study Big balance data duration:\n %.2f " % du, unit)
print("***************************** \n")
