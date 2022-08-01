""" assistant function """
import gdal
import numpy as np
import os
import shutil
import gc


"""" ******************************************* """
""" load, read, stack and reshape sentinel2 images as ( , n_bands) """


def load_sentinel2_img(folder_img):

    gc.collect()
    """ 1. make a list name """
    band_name = [band for band in os.listdir(folder_img)
                 if band[-4:] == ".tif"]
    # del band

    """ 2. open data-set """
    """ Ds = Data-set"""
    Ds = gdal.Open(folder_img + "/" + band_name[0])

    """ 3. pre-allocate """
    Bands = np.zeros((band_name.__len__(),
                      Ds.RasterXSize * Ds.RasterYSize))

    del Ds

    """ 4. load, read and stack sentinel2 images """
    for i in range(band_name.__len__()):
        """ band 8A should be placed after band 8 """
        if i == 8:
            Ds = gdal.Open(folder_img + "/" + band_name[11])
            B = Ds.GetRasterBand(1).ReadAsArray()
            Bands[8][:] = np.ravel(B)
            del Ds, B
            continue

        """ band 9 to 12 should placed after band 8A """
        if i > 8:
            Ds = gdal.Open(folder_img + "/" + band_name[i-1])
            B = Ds.GetRasterBand(1).ReadAsArray()
            Bands[i][:] = np.ravel(B)
            del Ds, B
            continue

        """ band 1 to 8 should placed in row 1 to 8 """
        Ds = gdal.Open(folder_img + "/" + band_name[i])
        B = Ds.GetRasterBand(1).ReadAsArray()
        Bands[i] = np.ravel(B)

    BandT = Bands.transpose()
    return BandT


""" ********************************************* """

def export_output_data_set(folder_img, output_folder_name, output_name, output_value, bands=1):
    band_name = [band for band in os.listdir(folder_img)
                 if band[-4:] == ".tif"]

    Ds = gdal.Open(folder_img + "/" + band_name[0])

    x_size = Ds.RasterXSize
    y_size = Ds.RasterYSize

    """ make output folder """
    if os.path.exists(folder_img + "/.." + "/" + output_folder_name):
        shutil.rmtree(folder_img + "/.." + "/" + output_folder_name)
        os.mkdir(folder_img + "/.." + "/" + output_folder_name)
    else:
        os.mkdir(folder_img + "/.." + "/" + output_folder_name)

    """ build output data-set """
    Driver = gdal.GetDriverByName("GTiff")
    outputDs = Driver.Create(folder_img + "/.." + "/" + output_folder_name + "/" + output_name + ".tif",
                             bands=bands,
                             xsize=x_size,
                             ysize=y_size,
                             eType=Ds.GetRasterBand(1).DataType)

    outputDs.SetProjection(Ds.GetProjection())
    outputDs.SetGeoTransform(Ds.GetGeoTransform())

    outputDs.GetRasterBand(1).WriteArray(output_value.reshape(y_size, x_size))

    outputDs.FlushCache()
    del outputDs, Ds
