import time
import zMyDl_utils
import os, shutil
import gdal
import numpy as np
import gc

""" calculate NDVI for case study and train area """
tic =  time.time()
gc.collect()

""" paths
clip case study path: ClpCsStudyPth
clip train area path: ClpTrnAreaPth
"""
path = "/media/user/DATA/Article/MyDlArticle/Dl_SatClassification13990403/"
ClpCsStudyPth = "1- Clp case study"
ClpTrnAreaPth = "1- Clp train area"


""" load, read and calculate NDVI for case study """
BandName = [band for band in os.listdir(path + ClpCsStudyPth)
            if band[-4:] == ".tif"]

DsRed = gdal.Open(path + ClpCsStudyPth + "/" + BandName[3])
RedBnd = DsRed.GetRasterBand(1).ReadAsArray()
del DsRed

DsNIR = gdal.Open(path + ClpCsStudyPth + "/" + BandName[7])
NIRBnd = DsNIR.GetRasterBand(1).ReadAsArray()
del DsNIR

NDVICsStdy = (NIRBnd - RedBnd) / (NIRBnd + RedBnd)
del BandName

""" write NDVI export for case study data-set """
zMyDl_utils.export_output_data_set(path+ClpCsStudyPth,
                                   "3-0_NDVI_CaseStudy",
                                   "NDVI_CaseStudy",
                                   NDVICsStdy)

""" Re-class case study NDVI """
NDVIReClsCsStdy = NDVICsStdy
del NDVICsStdy
NDVIReClsCsStdy[np.where(NDVIReClsCsStdy >= 0.2)] = 3
NDVIReClsCsStdy[np.where((NDVIReClsCsStdy > 0) & (NDVIReClsCsStdy < 0.2))] = 2
NDVIReClsCsStdy[np.where(NDVIReClsCsStdy <= 0)] = 1

unique, counts = np.unique(NDVIReClsCsStdy, return_counts=True)
print("\n NDVI Re-class count for case study: \n",
      dict(zip(unique, counts)))

""" write Re-class case study NDVI """
zMyDl_utils.export_output_data_set(path+ClpCsStudyPth,
                                   "3-1_NDVI_Re-Class_CaseStudy",
                                   "NDVI_Re-class_CaseStudy",
                                   NDVIReClsCsStdy)

toc = time.time()
print("NDVI Case Study duration: %.2f " % (toc - tic))

""""  ************************ Train area ***********************  """
tic1 = time.time()
""" laoad, read and calculat NDVI for train area """
BandName = [band for band in os.listdir(path+ClpTrnAreaPth)
            if band[-4:] == ".tif"]

DsRed = gdal.Open(path + ClpTrnAreaPth + "/" + BandName[3])
RedBnd = DsRed.GetRasterBand(1).ReadAsArray()
del DsRed

DsNIR = gdal.Open(path + ClpTrnAreaPth + "/" + BandName[7])
NIRBnd = DsNIR.GetRasterBand(1).ReadAsArray()
del DsNIR

NDVITrnArea = (NIRBnd - RedBnd) / (NIRBnd + RedBnd)

""" write NDVI train area """
zMyDl_utils.export_output_data_set(path+ClpTrnAreaPth,
                                   "3-0_NDVI_TrainArea",
                                   "NDVI_TrainArea",
                                   NDVITrnArea)

""" Re-class NDVI for train area """

NDVIReClsTrnArea = NDVITrnArea
del NDVITrnArea

NDVIReClsTrnArea[np.where(NDVIReClsTrnArea >= 0.2)] = 3
NDVIReClsTrnArea[np.where((NDVIReClsTrnArea > 0) & (NDVIReClsTrnArea < 0.2))] = 2
NDVIReClsTrnArea[np.where(NDVIReClsTrnArea <= 0)] = 1

unique, counts = np.unique(NDVIReClsTrnArea, return_counts=True)
print("\n NDVI Re-class count for train area: \n",
      dict(zip(unique, counts)))

""" write NDVI Re-class for Train area """
zMyDl_utils.export_output_data_set(path+ClpTrnAreaPth,
                                   "3-1_NDVI_Re-Class_TrainArea",
                                   "NDVI_Re-Class_TrainArea",
                                   NDVIReClsTrnArea)


toc1 = time.time()
print("NDVI Train area duration is: %.2f sec" % (toc1 - tic1))