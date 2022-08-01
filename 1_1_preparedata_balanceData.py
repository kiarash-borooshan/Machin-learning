import zMyDl_utils
import numpy as np
import gdal
import random
import os

""" paths """
path = "/media/user/DATA/Article/MyDlArticle/Dl_SatClassification13990403/"

ClpTrnAreaPth = "1- Clp train area"
ClpCsStdyPth = "1- Clp case study"

NDVIReClsTrnAreaPth = "3-1_NDVI_Re-Class_TrainArea/NDVI_Re-Class_TrainArea.tif"
NDVIReClsCsStdyPth = "3-1_NDVI_Re-Class_CaseStudy/NDVI_Re-class_CaseStudy.tif"

""" load and read Train area data """
StckBandTrnArea = zMyDl_utils.load_sentinel2_img(path + ClpTrnAreaPth)

NDVITrnArea = np.ravel(gdal.Open(path + NDVIReClsTrnAreaPth).ReadAsArray())

unique, counts = np.unique(NDVITrnArea, return_counts=True)
print("\n NN-multi layer for train area: \n",
      dict(zip(unique, counts)))

a = np.random.choice([np.where(NDVITrnArea == 1)[0]][0], min(counts),
                     replace=False)
a1 = StckBandTrnArea[:][a]
aa = np.ones((a1.shape[0], 1))
aa1 = np.hstack((a1, aa))

b = np.random.choice([np.where(NDVITrnArea == 2)[0]][0], min(counts),
                     replace=False)
b2 = StckBandTrnArea[:][b]
bb = np.ones((a1.shape[0], 1)) * 2
bb1 = np.hstack((b2, bb))

c = np.where(NDVITrnArea == 3)[0]
c3 = StckBandTrnArea[:][c]
cc = np.ones((a1.shape[0], 1)) * 3
cc1 = np.hstack((c3, cc))

Tt = np.vstack((aa1, bb1, cc1))

Fldr = os.mkdir(path + "1_1_balance_data")
np.save( path + "1_1_balance_data" + "/" + "balance data.npy" , Tt)

# unique, counts = np.unique(a, return_counts=True)
# print("\n NN-multi layer for train area: \n",
#       dict(zip(unique, counts)))
# print(max(counts))
# d = b[:4]
# c = StckBandTrnArea[:][d]