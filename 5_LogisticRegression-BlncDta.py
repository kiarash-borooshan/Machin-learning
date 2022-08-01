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


ticTrnArea = time.time()
gc.collect()

""" paths """
path = "/media/user/DATA/Article/MyDlArticle/DL_SatClassification_balanceData_13990501/"

BlnceDtaFldr = "1_1_balance_data"

ClpCsStdyPth = "1- Clp case study"
ClpTrnAreaPath = "1- Clp train area"

NDVIReClssCsStdyPth = "3-1_NDVI_Re-Class_CaseStudy/NDVI_Re-class_CaseStudy.tif"
NDVIReClssTrnAreaPth = "3-1_NDVI_Re-Class_TrainArea/NDVI_Re-Class_TrainArea.tif"


"""" load and read balance data and NDVI Re-classed for Train area """
ticBlncDta = time.time()

BlncDta = np.load(path + BlnceDtaFldr + "/" + "balance data.npy")

StckBand_BlncDta = BlncDta[:, :-1]

one = np.ones((StckBand_BlncDta.shape[0]))
oneT = one.reshape(-1, 1)

xBlncDta = np.hstack((oneT, StckBand_BlncDta))
del StckBand_BlncDta, oneT, one

NDVITrnArea = BlncDta[:, -1:]
yBlncDta = NDVITrnArea.reshape(-1, 1)
del NDVITrnArea

""" solving logistic regression """
TrnTrnArea, xTstTrnArea, yTrnTrnArea, yTstTrnArea = train_test_split(StckBandTrnArea,
                                                                      NDVITrnArea,
                                                                      random_state=0)

""" prepare logistic regression """
LgstcRgr = LogisticRegression()
# LgstcRgr.fit(StckBandTrnArea, NDVITrnArea)

LgstcRgr.fit(xTrnTrnArea, yTrnTrnArea)

# predictTrnAreaSplt = LgstcRgr.predict(xTstTrnArea)

# ScorTrnAreaPredctSplt = 100 * np.mean(predictTrnAreaSplt == yTstTrnArea)
# print(ScorTrnAreaPredctSplt)
# print(" ******************* ")

predictTrnArea = LgstcRgr.predict(StckBandTrnArea)

unique, counts = np.unique(predictTrnArea, return_counts=True)
print("\nlogistic regression count for train area: \n",
      dict(zip(unique, counts)))
ScorTrnAreaPredct = 100 * np.mean(predictTrnArea == NDVITrnArea)
print("\n logistic regressio accuracy for train area: \n %.2f"
      % ScorTrnAreaPredct)
tocBlncDta = time.time()
du = tocBlncDta - ticBlncDta
unit = "sec"
if du > 60:
    unit = "Min"
    du = du / 60
print("Normal equation for balance data duration: %.2f " % du, unit)
del ticBlncDta, tocBlncDta, unit, du
print("***************************** \n")


""" *************************************************************** """
"""" logistic regression (rounded value export)"""
ticTrnArea = time.time()

StckBandTrnArea = zMyDl_utils.load_sentinel2_img(path + ClpTrnAreaPath)


""" load and read Train area data """
StckBandTrnArea = zMyDl_utils.load_sentinel2_img(path + ClpTrnAreaPth)
NDVITrnArea = np.ravel(gdal.Open(path + NDVIReClsTrnAreaPth).ReadAsArray())

xTrnTrnArea, xTstTrnArea, yTrnTrnArea, yTstTrnArea = train_test_split(StckBandTrnArea,
                                                                      NDVITrnArea,
                                                                      random_state=0)

""" prepare logistic regression """
LgstcRgr = LogisticRegression()
# LgstcRgr.fit(StckBandTrnArea, NDVITrnArea)

LgstcRgr.fit(xTrnTrnArea, yTrnTrnArea)

# predictTrnAreaSplt = LgstcRgr.predict(xTstTrnArea)

# ScorTrnAreaPredctSplt = 100 * np.mean(predictTrnAreaSplt == yTstTrnArea)
# print(ScorTrnAreaPredctSplt)
# print(" ******************* ")

predictTrnArea = LgstcRgr.predict(StckBandTrnArea)

unique, counts = np.unique(predictTrnArea, return_counts=True)
print("\nlogistic regression count for train area: \n",
      dict(zip(unique, counts)))
ScorTrnAreaPredct = 100 * np.mean(predictTrnArea == NDVITrnArea)
print("\n logistic regressio accuracy for train area: \n %.2f"
      % ScorTrnAreaPredct)

tocTrnAre = time.time()
print("logistic regression for train area duration: \n %.2f"
      %(tocTrnAre-ticTrnArea))

zMyDl_utils.export_output_data_set(path + ClpTrnAreaPth,
                                   "5-0_Logistic Regression_TrainArea",
                                   "Logistic Regression_TrainArea",
                                   predictTrnArea)

"""" ****************** """
""" load and read case study data """
ticCsStdy = time.time()

StckBandCsTdy = zMyDl_utils.load_sentinel2_img(path + ClpCsStdyPth)
NDVIReClsCsStdy = np.ravel(gdal.Open(path + NDVIReClsCsStdyPth).ReadAsArray())

predictCsStudy = LgstcRgr.predict(StckBandCsTdy)

unique, counts = np.unique(predictCsStudy, return_counts=True)
print(" \n ****** \n ")
print("logistic regression case study count: \n" ,
      dict(zip(unique, counts)), "\n")

ScorCsStdyPredct = 100 * np.mean(predictCsStudy == NDVIReClsCsStdy)
print("logistic regressio case study accuracy: %.2f \n" % ScorCsStdyPredct)

zMyDl_utils.export_output_data_set(path + ClpCsStdyPth,
                                   "5-1_logistic regression_CaseStudy",
                                   "logistic regression_CaseStudy",
                                   predictCsStudy)

tocCsStdy = time.time()
print("logistic regression for case study duration: %.2f sec \n" % (tocCsStdy - ticCsStdy))

gc.collect()