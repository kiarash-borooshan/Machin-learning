""" NN - single layer"""
import time
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import gc
import zMyDl_utils
import gdal
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import patheffects
from sklearn.metrics import confusion_matrix
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
# BlnceDtaFldr = "1_1_balance_data"
#
#
# """" load and read balance data and NDVI Re-classed for CsStdySmll """
# ticBlncDta = time.time()
#
# BlncDta = np.load(path + BlnceDtaFldr + "/" + "balance data.npy")
# BlncDta = BlncDta.astype("float32")
#
# xBlncDta = BlncDta[:, :-1]
#
# NDVITrnArea = BlncDta[:, -1:]
# yBlncDta = NDVITrnArea.reshape(-1, 1)
# del NDVITrnArea
#
# """ split balance data to train and test"""
#
# xTrn, xTst, \
# yTrn, yTst = train_test_split(xBlncDta,
#                               yBlncDta,
#                               random_state=0,
#                               stratify=yBlncDta)
#
# mu = np.mean(xTst, axis=0)
# xTrn -= mu
# xTst -= mu
#
# yTrn = to_categorical(yTrn)
#
# """ NN Linear classifier """
# model = Sequential()
# model.add(Dense(4, activation="softmax", input_shape=(12,)))  # has 3 classes
# print(model.summary())
#
# model.compile(optimizer="sgd", loss="categorical_crossentropy",
#               metrics=["accuracy"])
#
# num_train = int(xTrn.shape[0] * 90 / 100)    # separate train and validation index
# history = model.fit(xTrn[: num_train], yTrn[: num_train],
#                     batch_size=32,
#                     epochs=50,
#                     verbose=2,
#                     validation_data=(xTrn[num_train:],
#                                      yTrn[num_train:]),
#                     shuffle=True)
#
# """ plot train area train and validation loss """
# plt.figure(figsize=(12, 8))
# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="validation Loss")
# plt.legend()
# plt.title("NN-single layer training Vs Validation loss \n"
#           "batch_size=32, epochs=50,")
# plt.ylim(-0.01)
# plt.grid()
# plt.show()
# plt.savefig(path + "zMapJPGExport-(BlnceDta)" + "/" +
#             "7-1 NN-single layer training Vs Validation loss(BlnceDta).png", dpi=700)
#
# """ prediction """
# yHtBlncDta = model.predict_classes(xTst)
# unique, counts = np.unique(yHtBlncDta, return_counts=True)
# print("\nNN-single layer count for (BlnceDta): \n",
#       dict(zip(unique, counts)))
#
#
# Acc = 100 * np.mean(yHtBlncDta.reshape(-1, 1) == yTst)
# print("\n NN-single layer accuracy: %.2f \n" % Acc)
#
# """ replace incorrect classified value to 10 """
# predictBlncDtaInc = yHtBlncDta.reshape(-1, 1).copy()
# predictBlncDtaInc[np.where(yHtBlncDta.reshape(-1, 1) != yTst)] = 10
#
# unique, counts = np.unique(predictBlncDtaInc, return_counts=True)
# print("\n Normal Equation round count Incorrect CsStdySmll for balance data: \n",
#       dict(zip(unique, counts)))
#
#
# """" duration """
# tocBlncDta = time.time()
# du = tocBlncDta - ticBlncDta
# unit = "sec"
# if du > 60:
#     unit = "Min"
#     du = du / 60


# print("NN-single layer for balance data duration: %.2f " % du, unit)
# print("***************************** \n")
#
# del ticBlncDta, tocBlncDta, du, unit
# del BlnceDtaFldr, BlncDta,  \
#     Acc, counts, history, model, mu, num_train, \
#     predictBlncDtaInc, unique, xTrn, xTst, \
#     yHtBlncDta, yTrn, yTst
#
#
# """ build Model for whole balance data """
# """ prepare data"""
# model = Sequential()
# model.add(Dense(4, activation="softmax", input_shape=(12,)))  # has 3 classes
# print(model.summary())
#
# model.compile(optimizer="sgd", loss="categorical_crossentropy",
#               metrics=["accuracy"])
# yBlncDta = to_categorical(yBlncDta)
# model.fit(xBlncDta, yBlncDta,
#           batch_size=32,
#           epochs=50,
#           verbose=2,
#           shuffle=True)
#
# model.save("MyNNSingleModel.h5")
#
# del xBlncDta, yBlncDta,


""" **************************************************************** """
""" case study small """

""" case study path """
ClpCsStdySmllPath = "1- Clp CsStdySmll"
NDVIReClssCsStdySmllPath = "3-1_NDVI_Re-Class_CsStdySmll/NDVI_Re-Class_CsStdySmll.tif"


ticCsStdSmll = time.time()
""" load train data """
M = load_model("MyNNSingleModel.h5")
StckBandCsStdySmll = zMyDl_utils.load_sentinel2_img(path + ClpCsStdySmllPath)
yCsStdySmll = np.ravel(gdal.Open(path + NDVIReClssCsStdySmllPath).ReadAsArray())
yCsStdySmll1 = yCsStdySmll.copy()
yCsStdySmll = to_categorical(yCsStdySmll)


yHatCsStdySmll = M.predict_classes(StckBandCsStdySmll)
unique, counts = np.unique(yHatCsStdySmll, return_counts=True)
print("\n NN-singleLyr CsStdySmll for balance data: \n",
      dict(zip(unique, counts)))

""" replace incorrect classified value to 10 """
predictBlncDtaInc = yHatCsStdySmll.reshape(-1, 1).copy()
predictBlncDtaInc[np.where(yHatCsStdySmll.reshape(-1, 1) != yCsStdySmll1.reshape(-1, 1))] = 10

unique, counts = np.unique(predictBlncDtaInc, return_counts=True)
print("\n NN-singleLyr Incorrect CsStdySmll for balance data: \n",
      dict(zip(unique, counts)))

Acc = 100 * np.mean(yHatCsStdySmll.reshape(1, -1) == yCsStdySmll1.reshape((1, -1)))
print("accuracy NN-singleLyr for CsStdySmll: %.2f " % Acc)

# loss, acc = M.evaluate(StckBandCsStdySmll, yCsStdySmll)

"""" write logistc regression for CsStdySmll balance data"""
zMyDl_utils.export_output_data_set(path + ClpCsStdySmllPath,
                                   "7-1_NN-singleLyr_CsStdySmll(BlncDta)",
                                   "NN-singleLyr_CsStdySmll(BlncDta)",
                                   predictBlncDtaInc)


"""" duration """
tocCsStdySmll = time.time()
du = tocCsStdySmll - ticCsStdSmll
unit = "sec"
if du > 60:
    unit = "Min"
    du = du / 60
print("NN-single layer for Case Study small balance data duration: %.2f " % du, unit)
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
""" load train data """
M = load_model("MyNNSingleModel.h5")
StckBandCsStdyBig = zMyDl_utils.load_sentinel2_img(path + ClpCsStdyBigPath)
yCsStdyBig = np.ravel(gdal.Open(path + NDVIReClssCsStdyBigPath).ReadAsArray())
yCsStdyBig1 = yCsStdyBig.copy()
yCsStdyBig = to_categorical(yCsStdyBig)


yHatCsStdyBig = M.predict_classes(StckBandCsStdyBig)
unique, counts = np.unique(yHatCsStdyBig, return_counts=True)
print("\n NN-singleLyr CsStdyBig for balance data: \n",
      dict(zip(unique, counts)))

""" replace incorrect classified value to 10 """
predictBlncDtaInc = yHatCsStdyBig.reshape(-1, 1).copy()
predictBlncDtaInc[np.where(yHatCsStdyBig.reshape(-1, 1) != yCsStdyBig1.reshape(-1, 1))] = 10

unique, counts = np.unique(predictBlncDtaInc, return_counts=True)
print("\n NN-singleLyr Incorrect CsStdyBig for balance data: \n",
      dict(zip(unique, counts)))

Acc = 100 * np.mean(yHatCsStdyBig.reshape(1, -1) == yCsStdyBig1.reshape((1, -1)))
print("accuracy NN-singleLyr for CsStdyBig: %.2f " % Acc)

# loss, acc = M.evaluate(StckBandCsStdyBig, yCsStdyBig)

"""" write logistc regression for CsStdyBig balance data"""
zMyDl_utils.export_output_data_set(path + ClpCsStdyBigPath,
                                   "7-1_NN-singleLyr_CsStdyBig(BlncDta)",
                                   "NN-singleLyr_CsStdyBig(BlncDta)",
                                   predictBlncDtaInc)


"""" duration """
tocCsStdyBig = time.time()
du = tocCsStdyBig - ticCsStdBig
unit = "sec"
if du > 60:
    unit = "Min"
    du = du / 60
print("NN-single layer for Case Study Big balance data duration: %.2f " % du, unit)
print("***************************** \n")
