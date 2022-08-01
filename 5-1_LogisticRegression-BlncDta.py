""" logistic regression Balance data """
import time
import numpy as np
import os
import shutil
import gdal
import gc
import zMyDl_utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


gc.collect()

""" paths """
path = "/media/user/DATA/Article/MyDlArticle/DL_SatClassification_balanceData_13990501/"

BlnceDtaFldr = "1_1_balance_data"



"""" load and read balance data and NDVI Re-classed  """
ticBlncDta = time.time()

BlncDta = np.load(path + BlnceDtaFldr + "/" + "balance data.npy")

xBlncDta = BlncDta[:, :-1]


NDVITrnArea = BlncDta[:, -1:]
yBlncDta = NDVITrnArea.reshape(-1, 1)
del NDVITrnArea


""" logistic regression for (balance data -
 train & test splited data) """

xTrn, xTst, \
yTrn, yTst = train_test_split(xBlncDta,
                              yBlncDta,
                              random_state=0,
                              stratify=yBlncDta)

""" prepare logistic regression """
LgstcRgrBlncDta = LogisticRegression()

LgstcRgrBlncDta.fit(xTrn, yTrn)

predictBlncDta = LgstcRgrBlncDta.predict(xTst)

unique, counts = np.unique(predictBlncDta, return_counts=True)
print("\nlogistic regression count for CsStdySmll: \n",
      dict(zip(unique, counts)))


Scr = LgstcRgrBlncDta.score(xTst, yTst)
print("\nlogistic regression accuracy for Balance data: \n %.2f " % Scr)

""" replace incorrect classified value to 10 """
predictBlncDtaInc = predictBlncDta.reshape(-1, 1).copy()
predictBlncDtaInc[np.where(predictBlncDta.reshape(-1, 1) != yTst)] = 10

unique, counts = np.unique(predictBlncDtaInc, return_counts=True)
print("\n Normal Equation round count Incorrect CsStdySmll for balance data: \n",
      dict(zip(unique, counts)))

"""" duration """
tocBlncDta = time.time()
du = tocBlncDta - ticBlncDta
unit = "sec"
if du > 60:
    unit = "Min"
    du = du / 60
print("Normal equation for balance data duration: %.2f " % du, unit)
print("***************************** \n")

del LgstcRgrBlncDta, Scr, counts, predictBlncDtaInc, predictBlncDta, unique
del  xTrn, xTst, yTrn, yTst, du, unit, tocBlncDta, ticBlncDta, BlncDta, BlnceDtaFldr


""" **************************************************************"""
""" logistic regression for Case Study Small (whole balance data) """

""" case study path """
ClpCsStdySmllPath = "1- Clp CsStdySmll"
NDVIReClssCsStdySmllPath = "3-1_NDVI_Re-Class_CsStdySmll/NDVI_Re-Class_TrainArea.tif"


ticCsStdSmll = time.time()

LgstcRgrBlncDta = LogisticRegression()
LgstcRgrBlncDta.fit(xBlncDta, yBlncDta)

""" load train data """

StckBandCsStdySmll = zMyDl_utils.load_sentinel2_img(path + ClpCsStdySmllPath)
yCsStdySmll = np.ravel(gdal.Open(path + NDVIReClssCsStdySmllPath).ReadAsArray())


""" apply logistic regression to CsStdySmll """
yHtCsStdySmll = LgstcRgrBlncDta.predict(StckBandCsStdySmll)

unique, counts = np.unique(yHtCsStdySmll, return_counts=True)
print("\nlogistic regression count for CsStdySmll: \n",
      dict(zip(unique, counts)))

""" replace incorrect classified value to 10 """
predictBlncDtaInc = yHtCsStdySmll.reshape(-1, 1).copy()
predictBlncDtaInc[np.where(yHtCsStdySmll.reshape(-1, 1) != yCsStdySmll.reshape(-1, 1))] = 10

unique, counts = np.unique(predictBlncDtaInc, return_counts=True)
print("\n Normal Equation round count Incorrect CsStdySmll for balance data: \n",
      dict(zip(unique, counts)))

Acc = 100 * np.mean(yHtCsStdySmll.reshape(1, -1) == yCsStdySmll.reshape((1, -1)))
print("accuracy logistic regression for CsStdySmll: %.2f " % Acc)


# Scr = LogisticRegression.score(StckBandCsStdySmll, yCsStdySmll)
# print("accuracy logistic regression for CsStdySmll: %.2f " % Scr)


"""" write logistc regression for CsStdySmll balance data"""
zMyDl_utils.export_output_data_set(path + ClpCsStdySmllPath,
                                   "5-1_logistic regression_CsStdySmll(BlncDta)",
                                   "logistic regression_CsStdySmll(BlncDta)",
                                   predictBlncDtaInc)

""" duration for CsStdySmll"""
tocTrnArea = time.time()
du = tocTrnArea - ticCsStdSmll
unit = "sec"
if du > 60:
    unit = "Min"
    du = du / 60
print("Normal equation for balance data duration CsStdySmll: %.2f " % du, unit)
print("***************************** \n")
del StckBandCsStdySmll, counts, predictBlncDtaInc, unique, \
    xBlncDta, yBlncDta, yHtCsStdySmll, yCsStdySmll, Acc, \
    ClpCsStdySmllPath, NDVIReClssCsStdySmllPath
del ticCsStdSmll, tocTrnArea, du, unit


""" **************************************************************"""
""" logistic regression for Case Study Big (whole balance data) """

""" case study big path """
ClpCsStdyBigPth = "1- Clp CsStdyBig"
NDVIReClssCsStdyBigPth = "3-1_NDVI_Re-Class_CsStdyBig/NDVI_Re-Class_CsStdyBig.tif"

ticCsStdyBig = time.time()

""" load case study big data"""
StckBandCsStdyBig = zMyDl_utils.load_sentinel2_img(path + ClpCsStdyBigPth)
yCsStdyBig = np.ravel(gdal.Open(path + NDVIReClssCsStdyBigPth).ReadAsArray())


""" apply logistic regression to CsStdySmll """
yHtCsStdyBig = LgstcRgrBlncDta.predict(StckBandCsStdyBig)

unique, counts = np.unique(yHtCsStdyBig, return_counts=True)
print("\nlogistic regression count for CsStdyBig: \n",
      dict(zip(unique, counts)))


""" replace incorrect classified value to 10 """
predictBlncDtaInc = yHtCsStdyBig.reshape(-1, 1).copy()
predictBlncDtaInc[np.where(yHtCsStdyBig.reshape(-1, 1) != yCsStdyBig.reshape(-1, 1))] = 10

unique, counts = np.unique(predictBlncDtaInc, return_counts=True)
print("\n Normal Equation round count Incorrect CsStdyBig for balance data: \n",
      dict(zip(unique, counts)))

Acc = 100 * np.mean(yHtCsStdyBig.reshape(1, -1) == yCsStdyBig.reshape((1, -1)))
print("accuracy logistic regression for CsStdyBig: %.2f " % Acc)


"""" write logistc regression for CsStdySmll balance data"""
zMyDl_utils.export_output_data_set(path + ClpCsStdyBigPth,
                                   "5-1_logistic regression_CsStdyBig(BlncDta)",
                                   "logistic regression_CsStdyBig(BlncDta)",
                                   predictBlncDtaInc)

""" duration for CsStdySmll"""
tocCsStdyBig = time.time()
du = tocCsStdyBig - ticCsStdyBig
unit = "sec"
if du > 60:
    unit = "Min"
    du = du / 60
print("Normal equation for balance data duration CsStdyBig: %.2f " % du, unit)
print("***************************** \n")
del StckBandCsStdyBig, counts, predictBlncDtaInc, unique, yHtCsStdyBig, \
    yCsStdyBig, Acc,ClpCsStdyBigPth, NDVIReClssCsStdyBigPth
del ticCsStdyBig, tocCsStdyBig, du, unit