import gdal
import numpy as np
import glob
import os
import osr
from PIL import Image
import matplotlib.pyplot as plt

def reproject_image(source_img, target_img, clip_shapefile = None, no_data_val = 0):

    s = gdal.Open(source_img)
    geo_s = s.GetGeoTransform()
    s_x_size, s_y_size = s.RasterXSize, s.RasterYSize
    s_xmin = min(geo_s[0], geo_s[0] + s_x_size * geo_s[1])
    s_xmax = max(geo_s[0], geo_s[0] + s_x_size * geo_s[1])
    s_ymin = min(geo_s[3], geo_s[3] + s_y_size * geo_s[5])
    s_ymax = max(geo_s[3], geo_s[3] + s_y_size * geo_s[5])
    s_xRes, s_yRes = abs(geo_s[1]), abs(geo_s[5])
    if type(target_img) == str:
        t = gdal.Open(target_img)    
    else:
        t = target_img
    geo_t = t.GetGeoTransform()
    x_size, y_size = t.RasterXSize, t.RasterYSize
    xmin = min(geo_t[0], geo_t[0] + x_size * geo_t[1])
    xmax = max(geo_t[0], geo_t[0] + x_size * geo_t[1])
    ymin = min(geo_t[3], geo_t[3] + y_size * geo_t[5])
    ymax = max(geo_t[3], geo_t[3] + y_size * geo_t[5])
    xRes, yRes = abs(geo_t[1]), abs(geo_t[5])
    if (s_x_size == x_size) & (s_y_size == y_size) & \
       (s_xmin == xmin) & (s_ymin == ymin) & \
       (s_xmax == xmax) & (s_ymax == ymax) & \
       (s_xRes == xRes) & (s_yRes == yRes):

        if clip_shapefile is not None:

            g = gdal.Warp('', source_img, format='MEM',
                        cutlineDSName=clip_shapefile,
                        cropToCutline=False,
                        dstNodata = no_data_val)
        else:
            g = gdal.Open(source_img)
    else:
        dstSRS = osr.SpatialReference()
        raster_wkt = t.GetProjection()
        dstSRS.ImportFromWkt(raster_wkt)

        if clip_shapefile is not None:

            g = gdal.Warp('', source_img, format='MEM',
                      outputBounds=[xmin, ymin, xmax, ymax], xRes=xRes, yRes=yRes,
                      dstSRS=dstSRS, cutlineDSName=clip_shapefile,
                      cropToCutline=False,
                      dstNodata = no_data_val)
        else:
            g = gdal.Warp('', source_img, format='MEM',
                      outputBounds=[xmin, ymin, xmax, ymax], xRes=xRes, yRes=yRes,
                       dstSRS=dstSRS)
    return g    


def compile_sentinel_1_stacks(data_root, s1_bands, year, to_reproject_to, shapefile = None):
    
    # as we are opening the S1 data, we havge to specify where the 
    # S1 data is called 'S1_GRD'
    file_root = os.path.join(data_root, 'S1_GRD')
    
    s1_lyr_stacks = {}
    
    for band in s1_bands:
        
        # build the file path to the data repository
        file_repository = os.path.join(file_root,band,'aoi')

        # use 'glob' to list all the files in the repository
        list_of_files = glob.glob(file_repository+'/*')

        # glob randomizes the result, so sort them which will 
        # put them into chronological order
        sorted_file_list = sorted(list_of_files)

        arrays_to_concatenate = []

        # loop throguh each of the files that fit the description
        for file in sorted_file_list:

            # find if the filename has the year of interest inside
            if year not in file:
                # if it does not, skip it and dont open it
                continue
            
            opn = reproject_image(file, to_reproject_to, 
                                  clip_shapefile = shapefile)
                
            arr = opn.ReadAsArray()
            arrays_to_concatenate.append(arr)

        # concatenate all the monthly datasets
        loop_lyr_stack = np.concatenate(arrays_to_concatenate, axis = 0)
        loop_lyr_stack[loop_lyr_stack == 0] = np.nan
        s1_lyr_stacks[band] = loop_lyr_stack
        
    return s1_lyr_stacks


def compile_sentinel_2_stacks(data_root, s2_bands, year, to_reproject_to, shapefile = None):
    
    # as we are opening the S2 data, we havge to specify where the 
    # S2 data is 'S2_SR'
    file_root = os.path.join(data_root, 'S2_SR')
    
    s2_lyr_stacks = {}
    
    for band in s2_bands:
        
        # build the file path to the data repository
        file_repository = os.path.join(file_root,band,'aoi')

        # use 'glob' to list all the files in the repository
        list_of_files = glob.glob(file_repository+'/*')

        # glob randomizes the result, so sort them which will 
        # put them into chronological order
        sorted_file_list = sorted(list_of_files)

        arrays_to_concatenate = []

        # loop throguh each of the files that fit the description
        for file in sorted_file_list:

            # find if the filename has the year of interest inside
            if year not in file:
                # if it does not, skip it and dont open it
                continue
            
            opn = reproject_image(file, to_reproject_to, 
                                  clip_shapefile = shapefile)
                
            # recast the data 
            arr = opn.ReadAsArray()
            arrays_to_concatenate.append(arr)

        # concatenate all the monthly datasets
        loop_lyr_stack = np.concatenate(arrays_to_concatenate, axis = 0).astype(np.float32,copy=False)
        loop_lyr_stack[loop_lyr_stack == 0] = np.nan
        s2_lyr_stacks[band] = loop_lyr_stack
        
    return s2_lyr_stacks

def find_sentinel_2_cloudmask(in_stack, critical_cloud_threshold = 1000):
    
    # find the minimum for each pixel, to act as the 'reference'
    #min_of_pixels = np.nanmin(in_stack['B2'], axis = 0)
    min_of_pixels = np.nanquantile(in_stack['B2'], 0.1, axis=0)
    
    # find the difference of each datapoint to the reference
    dif_from_ref = in_stack['B2'] - min_of_pixels

    # create a variable to designate which pixels are clouded
    are_clouded = dif_from_ref > critical_cloud_threshold
    
    return are_clouded

def display_esri_2022_basemap(axs = None):
    
    basemap_path = '/media/DataShare/Alex/un_surinam/esri_basemap_v4.tif'
    clip_shapefile = '/media/DataShare/Alex/un_surinam/aoi.shp'
    
    dataset = gdal.Warp('', basemap_path,  format='MEM',
                        cutlineDSName=clip_shapefile,
                        cropToCutline=True,dstNodata = 0)
    arr = dataset.ReadAsArray()
    rgb_array = np.dstack(arr[:3])
    im = Image.fromarray(rgb_array)
    
    if axs is None:
        plt.figure(figsize=(10,8))
        plt.imshow(im)
    else:
        axs.imshow(im)
        
def overlay_output_on_esri_2022_basemap(output):
    
    basemap_path = '/media/DataShare/Alex/un_surinam/esri_basemap_v4.tif'
    clip_shapefile = '/media/DataShare/Alex/un_surinam/aoi.shp'
    to_project_to = '/data/un_suriname_demo/datacube/S2_SR/B2/aoi/S2_SR_B2_aoi_2019-09.tif'
    
    dataset = reproject_image(basemap_path, to_project_to,
                             clip_shapefile=clip_shapefile)
    arr = dataset.ReadAsArray()
    rgb_array = np.dstack(arr[:3])
    im = Image.fromarray(rgb_array)
    
    cpy = np.copy(output).astype(np.float32,copy=False)
    cpy[cpy == 0] = np.nan
    
    plt.figure(figsize=(10,8))
    plt.imshow(im)
    plt.imshow(cpy, cmap='Purples_r',alpha=0.2,interpolation='none',vmin=0,vmax=1)
    plt.grid()
    

