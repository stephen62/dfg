from glob import glob
import pickle
import os
import cv2
import itk
import random
import numpy as np
import SimpleITK as sitk
import matplotlib
from matplotlib import pyplot as plt
from scipy import linalg, ndimage


class DRR(object):
    def __init__(self, length, distance, threshold):
        self.focal_length = length
        self.distance = distance  # mm, the distance from projection plane to body
        self.threshold = threshold
        self.dimension = 3

    def __call__(self, volume, spacing, para, img_shape, img_spacing):
        itk_vol = itk.GetImageFromArray(volume)
        itk_vol.SetSpacing(spacing)
        vol_origin = itk_vol.GetOrigin()  # Origin is [0, 0, 0]
        vol_res = itk_vol.GetSpacing()
        vol_region = itk_vol.GetBufferedRegion()
        vol_size = vol_region.GetSize()
        InputImageType = itk.Image.SS3
        pixeltype = itk.D

        vol_center = itk.Point[pixeltype, self.dimension](np.array(vol_res) * np.array(vol_size) / 2)
        focalpoint = itk.Point[pixeltype, self.dimension](
            (vol_center[0], vol_center[1], self.focal_length - self.distance))

        rx, ry, rz, tx, ty, tz = para
        translation = itk.Vector[pixeltype, self.dimension]((tx, ty, tz))
        transform = itk.CenteredEuler3DTransform.D.New()
        transform.SetCenter(vol_center)
        transform.SetComputeZYX(True)
        transform.SetTranslation(translation)
        transform.SetRotation(np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(rz))

        interpolator = itk.RayCastInterpolateImageFunction[InputImageType, itk.ctype('double')].New()
        interpolator.SetTransform(transform)
        interpolator.SetThreshold(self.threshold)
        interpolator.SetFocalPoint(focalpoint)

        itk_vol.SetOrigin((0, 0, 0))
        resample_filter = itk.ResampleImageFilter[InputImageType, InputImageType].New()
        resample_filter.SetInterpolator(interpolator)
        resample_filter.SetInput(itk_vol)
        resample_filter.SetDefaultPixelValue(0)
        resample_filter.SetTransform(transform)

        img_size = itk.Size[self.dimension]((img_shape[0],img_shape[1], 1))
        resample_filter.SetSize(img_size)
        resample_filter.SetOutputSpacing((img_spacing[0],img_spacing[1], 1.0))

        # outputorigin indicate the ralationship between projection image (0, 0) and CT volume(0,0,0)
        img_origin = np.array(vol_center)[:2] - np.array(img_shape) * img_spacing / 2
        outputorigin = itk.Point[pixeltype, self.dimension](
            (img_origin[0], img_origin[1], -self.distance))  # physical corrdinate
        resample_filter.SetOutputOrigin(outputorigin)
        resample_filter.Update()

        ray = resample_filter.GetOutput()
        drr = itk.GetArrayFromImage(ray)
        drr = drr.squeeze()
        #         grayimg = (drr - drr.min())/(drr.max() - drr.min()) * 255
        grayimg = (np.zeros_like(drr) - (drr - drr.min()) / (drr.max() - drr.min())) * 255

        return grayimg

def main(para):
    ct_file = 'volume.pkl'
    with open(ct_file, 'rb') as f:
        ct_volume = pickle.load(f)

    ct_spacing = [ 0.3125,  0.3125,  1.    ]
    length = 1000
    distance = 0
    threshold = 1150
    img_shape = (1024, 1024)
    img_spacing = (0.2, 0.2)
    function = DRR(length, distance, threshold)
    image = function(ct_volume, ct_spacing, para, img_shape, img_spacing)
    plt.figure(figsize=(12, 12))
    plt.imshow(image, 'gray')
    plt.show()

if __name__ == '__main__':
    para = [1, 2, 3, 0, 0, 0]
    main(para)