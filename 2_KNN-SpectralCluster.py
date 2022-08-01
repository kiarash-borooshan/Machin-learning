import time
import zMyDl_utils
# import os, shutil
# import gdal
import numpy as np
import gc
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import patheffects


""" calculate NDVI for case study and train area """
ticKNN = time.time()
gc.collect()

plt.style.use("dark_background")
plt.xkcd()
plt.rcParams['path.effects'] = [patheffects.withStroke(linewidth=1)]
plt.rcParams['figure.facecolor'] = 'black'


""" paths
clip case study path: ClpCsStudyPth
clip train area path: ClpTrnAreaPth
"""
path = "/media/user/DATA/Article/MyDlArticle/Dl_SatClassification13990403/"
ClpCsStudyPth = "1- Clp case study"
ClpTrnAreaPth = "1- Clp train area"

""" load and read Train area data """
StckBandTrnArea = zMyDl_utils.load_sentinel2_img(path + ClpTrnAreaPth)

""" the Elbow Method for KNN """
distortions = []
# nCls = 21
# for i in range(1, nCls):
#     kn = KMeans(n_clusters=i, random_state=0)
#     kn.fit(StckBandTrnArea)
#     distortions.append(kn.inertia_)

i = 1
nCls = []
while True:
    kn = KMeans(n_clusters=i, random_state=0)
    kn.fit(StckBandTrnArea)
    distortions.append(kn.inertia_)
    if kn.inertia_ < 5000:
        nCls = i

        """ apply KNN and export """
        Lbl = kn.fit_predict(StckBandTrnArea)
        zMyDl_utils.export_output_data_set(path + ClpTrnAreaPth,
                                           "2-0_KNN_TrainArea",
                                           "KNN_TrainArea",
                                           Lbl)

        unique, counts = np.unique(Lbl, return_counts=True)
        print("\n MLP count for train area: \n",
              dict(zip(unique, counts)))

        break
    i += 1

""" plot elbow result """
plt.plot(range(1, nCls+1), distortions, marker="o")
plt.xlabel("number of cluster")
plt.ylabel("distortion")
plt.grid()
plt.title(" the Elbow Method for KNN ")


""" time """
tocKNN = time.time()
unit = "sec"
du = tocKNN - ticKNN
if du > 60:
    du = du / 60
    unit = "MIN"

print("KNN duration: %.2f " % du, unit)
