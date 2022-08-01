""" neural network MLP """

import time
import os
import numpy as np
import gc
import pandas as pd
import zMyDl_utils
import gdal
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

""" scatter plot """
# idx = np.random.randint(StckBandTrnArea.shape[0], size=10000)
# StckBandTrnAreaClpRndm = StckBandTrnArea[idx, :]
# clmn = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", ]
#
# satellite_df = pd.DataFrame(StckBandTrnAreaClpRndm, columns=clmn)
# # plt.title("sentinel2 satellite bands scatter plot ")
# pd.plotting.scatter_matrix(satellite_df,
#                            diagonal="kde",
#                            marker=".",
#                            s=10)

# plt.savefig("sentinel2 satellite bands scatter plot ", dpi=700)

""" MLP """
clf = MLPClassifier()
clf.fit(xTrnTrnArea, yTrnTrnArea)

yTrnHt = clf.predict(xTstTrnArea)
acc = 100 * np.mean(yTrnHt == yTstTrnArea)
print("MLP Accuracy for train area: %.2f \n" % acc)

score = clf.score(xTstTrnArea, yTstTrnArea)
print("MLP score (R^2) for train area: %.2f \n" % acc)

""" apply to train area """
predictTrnArea = clf.predict(StckBandTrnArea)

unique, counts = np.unique(predictTrnArea, return_counts=True)
print("\n MLP count for train area: \n",
      dict(zip(unique, counts)))

zMyDl_utils.export_output_data_set(path + ClpTrnAreaPth,
                                   "6-1_MLP_TrainArea",
                                   "MLP_TrainArea",
                                   predictTrnArea)

""" incorrent change value """
predictTrnAreaInc = predictTrnArea
predictTrnAreaInc[np.where(predictTrnArea != NDVITrnArea)] = 10

zMyDl_utils.export_output_data_set(path + ClpTrnAreaPth,
                                   "6-1-1_MLP_TrainAreaInc",
                                   "MLP_TrainAreaInc",
                                   predictTrnAreaInc)

"""" confusion matrix for train area """
cm = confusion_matrix(predictTrnArea, NDVITrnArea)
plt.figure(figsize=(12, 8))
plt.imshow(cm, interpolation="nearest")
plt.title("MLP confusion matrix for train area")
plt.savefig("MLP confusion matrix for train area.png")

print("MLP weight coefficients for train area: ", clf.coefs_)
print("MLP intrecept for train area: ", clf.intercepts_)
""" y = weight * x + intercept """

""" train area duration """
tocTrnArea = time.time()
du = tocTrnArea - ticTrnArea
unit = "Sec"
if du > 60:
    du = du / 60
    unit = "Min"
print("MLP duration for train area: %.2f %s \n" % (du, unit))

del predictTrnAreaInc, predictTrnArea, acc, StckBandTrnArea
del yTrnTrnArea, yTstTrnArea, yTrnHt
del xTstTrnArea, xTrnTrnArea
del NDVITrnArea, NDVIReClsTrnAreaPth
gc.collect()


""" ***************************************** """
""" load case study data and apply"""
ticCsStdy = time.time()

StckBandCsTdy = zMyDl_utils.load_sentinel2_img(path + ClpCsStdyPth)
NDVIReClsCsStdy = np.ravel(gdal.Open(path + NDVIReClsCsStdyPth).ReadAsArray())

predictCsStdy = clf.predict(StckBandCsTdy)

unique, counts = np.unique(predictCsStdy, return_counts=True)
print("\nlogistic regression count for Case study: \n",
      dict(zip(unique, counts)))

acc = 100 * np.mean(NDVIReClsCsStdy == predictCsStdy)
print("MLP Accuracy for case study: %.2f \n" % acc)

score = clf.score(predictCsStdy, NDVIReClsCsStdy)
print("MLP score (R^2) for train area: %.2f \n" % acc)

zMyDl_utils.export_output_data_set(path + ClpTrnAreaPth,
                                   "6-2_MLP_CaseStudy",
                                   "MLP_CaseStudy",
                                   predictCsStdy)


cm = confusion_matrix(predictCsStdy, NDVIReClsCsStdy)
plt.figure(figsize=(12, 8))
plt.imshow(cm, interpolation="nearest")
plt.title("MLP confusion matrix for tcase study")
plt.savefig("MLP confusion matrix for case study.png")

print("MLP weight coefficients for case study: ", clf.coefs_)
print("MLP intrecept for case study: ", clf.intercepts_)

""" case study duration """
tocCsStdy = time.time()
du = tocCsStdy - ticCsStdy
unit = "Sec"
if du > 60:
    du = du / 60
    unit = "Min"
print("MLP duration for train area: %.2f %s \n" % (du, unit))
