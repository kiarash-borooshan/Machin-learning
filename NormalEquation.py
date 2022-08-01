""" balance data """
import time
import gc
import gdal
import numpy as np
import zMyDl_utils

""" Regression Normal Equation """

""" setting """
gc.collect()
np.set_printoptions(precision=2, formatter={"all": lambda x: "%.2f" % x})

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

""" solving Normal Equation """
theta = np.linalg.pinv(xBlncDta) @ yBlncDta
# theta2 = np.linalg.pinv(xBlncDta.T @ xBlncDta) @ xBlncDta.T @ yBlncDta

""" checking """
yHatBlncDta = xBlncDta @ theta
yHatBlncDtaR = np.round(yHatBlncDta)

unique, counts = np.unique(yHatBlncDtaR, return_counts=True)
print("\n Normal Equation round count for balance data: \n",
      dict(zip(unique, counts)))

Acc = 100 * np.mean(yHatBlncDtaR.reshape(1, -1) == yBlncDta.reshape(1, -1))
print(" Normal Equation rounded accuracy for balance data: %.2f " % Acc)
del Acc, unique, counts, yHatBlncDta, yHatBlncDtaR, yBlncDta, xBlncDta, BlncDta

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
"""" Normal Equation (rounded value export)"""
ticTrnArea = time.time()

StckBandTrnArea = zMyDl_utils.load_sentinel2_img(path + ClpTrnAreaPath)

one = np.ones((StckBandTrnArea.shape[0]))
oneT = one.reshape(-1, 1)

xTrnArea = np.hstack((oneT, StckBandTrnArea))
del StckBandTrnArea, oneT, one

""" load train area NDVI Re-classed """
yTrnArea = np.ravel(gdal.Open(path + NDVIReClssTrnAreaPth).ReadAsArray())   # (m, )
yTrnArea = yTrnArea.reshape(-1, 1)  # (m, 1)

"""" apply theeta to train area """
yHtTrnArea = xTrnArea @ theta

print("minimum export value for Normal equation balance data: %.2f " % yHtTrnArea.min())
print("maximum export value for Normal equation balance data: %.2f " % yHtTrnArea.max())

yHtTrnAreaR = np.round(yHtTrnArea)
unique, counts = np.unique(yHtTrnAreaR, return_counts=True)
print("\n Normal Equation round count train area for balance data: \n",
      dict(zip(unique, counts)))

""" replace incorrect classified value to 10 """
predictTrnAreaRoundInc = yHtTrnAreaR.copy()
predictTrnAreaRoundInc[np.where(yHtTrnAreaR != yTrnArea)] = 10

unique, counts = np.unique(predictTrnAreaRoundInc, return_counts=True)
print("\n Normal Equation round count Incorrect train area for balance data: \n",
      dict(zip(unique, counts)))


""" calculate accuracy  """
Acc = 100 * np.mean(yHtTrnAreaR == yTrnArea)
print("accuracy for Normal Equation rounded train area accuracy for balance data: %.2f " % Acc)
print("***************** \n ")
del Acc

""" Write Normal Equation rounded for train area """
zMyDl_utils.export_output_data_set(path + ClpTrnAreaPath,
                                   "4-0_Normal Equation rounded TrainArea(BlncDta)",
                                   "Normal Equation rounded TrainArea(BlncDta)",
                                   predictTrnAreaRoundInc)

tocTrnArea = time.time()
du = (tocTrnArea - ticTrnArea)
unit = "sec"
if du > 60:
    unit = "Min"
    du = du / 60
print("round Normal equation for train area  balance data duration: %.2f " % du, unit)
del counts, predictTrnAreaRoundInc, unique, xTrnArea, yHtTrnAreaR
print("*************************************************")

"""" *****************************************************************"""
""" Normal Equation (Re-class value export)"""
ticTrnAreaReClss = time.time()

yHtTrsh = (yHtTrnArea.max() - yHtTrnArea.min()) / 3
yHtRclss = yHtTrnArea
yHtRclss[np.where(yHtTrnArea >= 2*yHtTrsh)] = 3
yHtRclss[np.where((yHtRclss > yHtTrsh) & (yHtRclss < 2*yHtTrsh))] = 2
yHtRclss[np.where(yHtRclss <= yHtTrsh)] = 1

unique, counts = np.unique(yHtRclss, return_counts=True)
print("\n Normal Equation Re-clas count balance data for train area: \n",
      dict(zip(unique, counts)))

predictTrnAreaReClssInc = yHtRclss.copy()
predictTrnAreaReClssInc[np.where(yHtRclss != yTrnArea)] = 10

unique, counts = np.unique(predictTrnAreaReClssInc, return_counts=True)
print("\n Normal Equation Re-clas count balance data for train area: \n",
      dict(zip(unique, counts)))

Acc = 100 * np.mean(yHtRclss == yTrnArea)
print(" Normal Equation Re-class accuracy: %.2f " % Acc)
del Acc


""" write and export Normal Equation Re-class for train area """
zMyDl_utils.export_output_data_set(path+ClpTrnAreaPath,
                                   "4-1_Normal Equation Re-class TrainArea(BlncDta)",
                                   "Normal Equation Re-class TrainArea(BlncDta)",
                                   predictTrnAreaReClssInc)
tocTrnAreaReClss = time.time()
du = (tocTrnAreaReClss - ticTrnAreaReClss)
unit = "sec"
if du > 60:
    unit = "Min"
    du = du / 60
print("Normal equation for balance data duration: %.2f " % du, unit)
del ticTrnAreaReClss, tocTrnAreaReClss, unit, du
del counts, predictTrnAreaReClssInc,

print(" ************************************************ \n")

del yHtTrnArea, yTrnArea, unique, yHtRclss, yHtTrsh, ClpTrnAreaPath, NDVIReClssTrnAreaPth

""""" *********************************** """
""" apply theta on case study """

""" load case study data"""
ticCsStudy = time.time()

StckBandCsStdy = zMyDl_utils.load_sentinel2_img(path + ClpCsStdyPth)

one = np.ones((StckBandCsStdy.shape[0]))
oneT = one.reshape(-1, 1)

xCsStdy = np.hstack((oneT, StckBandCsStdy))
del one, oneT, StckBandCsStdy

""" load case study NDVI Re-classed """
yCsStdy = np.ravel(gdal.Open(path + NDVIReClssCsStdyPth).ReadAsArray())  # (m, )
yCsStdy = yCsStdy.reshape(-1, 1)  # (m, 1)

""" apply theta to case study """
yHatCsStdy = xCsStdy @ theta

print("minimum export value for Normal equation balance data: %.2f " % yHatCsStdy.min())
print("maximum export value for Normal equation balance data: %.2f " % yHatCsStdy.max())

yHatCsStdyR = np.round(yHatCsStdy)
unique, counts = np.unique(yHatCsStdyR, return_counts=True)
print("\n Normal Equation round count case study for case study: \n",
      dict(zip(unique, counts)))

""" replace incorrect classified value to 10 """
predictCsStdyRoundInc = yHatCsStdyR.copy()
predictCsStdyRoundInc[np.where(yHatCsStdyR != yCsStdy)] = 10

unique, counts = np.unique(predictCsStdyRoundInc, return_counts=True)
print("\n Normal Equation round count Incorrect case study for balance data: \n",
      dict(zip(unique, counts)))


""" calculate accuracy  """
Acc = 100 * np.mean(yHatCsStdyR == yCsStdy)
print("accuracy for Normal Equation rounded case study for balance data: %.2f " % Acc)
print("***************** \n ")


""" write Normal Equation rounded for case study """
zMyDl_utils.export_output_data_set(path + ClpCsStdyPth,
                                   "4-2_NormalEquation_rounded_CaseStudy(BlncDta)",
                                   "NormalEquation_rounded_CaseStudy(BlncDta)",
                                   predictCsStdyRoundInc)

tocCsStudy = time.time()
du = (tocCsStudy - ticCsStudy)
unit = "sec"
if du > 60:
    unit = "Min"
    du = du / 60
print("round Normal equation for case study  balance data duration: %.2f " % du, unit)
del counts, predictCsStdyRoundInc, unique, xCsStdy, yHatCsStdyR
print("*************************************************")

""" Normal Equation Re-class case study chechking2 """
ticCsStdyReClss = time.time()

yHtTrshCsStdy = (yHatCsStdy.max() - yHatCsStdy.min()) / 3
yHtRclssCsStdy = yHatCsStdy
yHtRclssCsStdy[np.where(yHtRclssCsStdy >= 2*yHtTrshCsStdy)] = 3
yHtRclssCsStdy[np.where((yHtRclssCsStdy > yHtTrshCsStdy) & (yHtRclssCsStdy < 2*yHtTrshCsStdy))] = 2
yHtRclssCsStdy[np.where(yHtRclssCsStdy <= yHtTrshCsStdy)] = 1

unique, counts = np.unique(yHtRclssCsStdy, return_counts=True)
print("\n Normal Equation Re-class count for case study: \n",
      dict(zip(unique, counts)))

Acc = 100 * np.mean(yHtRclssCsStdy == yCsStdy)
print(" Normal Equation Re-class accuracy: %.2f " % Acc)
del Acc
print(" **************************** \n")

""" replace incorrect classified value to 10 """
predictCsStdyRoundInc = yHtRclssCsStdy.copy()
predictCsStdyRoundInc[np.where(yHtRclssCsStdy != yCsStdy)] = 10

unique, counts = np.unique(predictCsStdyRoundInc, return_counts=True)
print("\n Normal Equation Re-class count Incorrect case study for balance data: \n",
      dict(zip(unique, counts)))

""" write Normal Equation Re-class CaseStudy """
zMyDl_utils.export_output_data_set(path + ClpCsStdyPth,
                                   "4-3_Normal Equation_Re-class_CaseStudy",
                                   "Normal Equation_Re-class_CaseStudy",
                                   predictCsStdyRoundInc)

tocCsStdyReClss = time.time()
print("Case study duration for Normal Equation Re-Class: %.2f sec"
      % (tocCsStdyReClss - ticCsStdyReClss))
