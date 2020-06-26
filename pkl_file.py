#-*- coding: utf-8 -*-
import os
# import numpy
# from PIL import Image
# from pylab import *
# import _pickle as cPickle
#
#
# def get_imlist(path):   #此函数读取特定文件夹下的png格式图像
#     return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dcm')]
#
# x=[]    #存放图片路径
# len_data=0  #图片个数
# for i in range(8):
#     img_path=get_imlist('D:\\CK\\'+str(i))  #字符串连接
#     #print neutral  #这里以list形式输出bmp格式的所有图像（带路径）
#     for j in range(len(img_path)):
#         x.append(img_path[j])
# #   print len(img_path)
#     len_data=len_data+len(img_path) #这可以以输出图像个数
# # print len_data
# #print len(x)
# d=len_data
# data=numpy.empty((d,64*64)) #建立d*（64*64）的矩阵
# while d>0:
#     img=Image.open(x[d-1])  #打开图像
#     #img_ndarray=numpy.asarray(img)
#     img_ndarray=numpy.asarray(img,dtype='float64')/256  #将图像转化为数组并将像素转化到0-1之间
#     data[d-1]=numpy.ndarray.flatten(img_ndarray)    #将图像的矩阵形式转化为一维数组保存到data中
#     d=d-1
# # print len_data
# # print shape(data)[1]  #输出矩阵大小
#
#
# data_label=numpy.empty(len_data)
# for label in range(len_data):
#     if label<924:
#         data_label[label]=0
#     elif (label>=905 and label<1180):
#         data_label[label]=1
#     elif (label>=1180 and label<1291):
#         data_label[label]=2
#     elif (label>=1291 and label<1619):
#         data_label[label]=3
#     elif (label>=1619 and label<1780):
#         data_label[label]=4
#     elif (label>=1780 and label<2289):
#         data_label[label]=5
#     elif (label>=2289 and label<2477):
#         data_label[label]=6
#     else:
#         data_label[label]=7
#
# data_label=data_label.astype(numpy.int)  #将标签转化为int类型
# # print data_label[924]
# # print data_label[1638]
#
#
# #保存data以及data_label到data.pkl文件
# write_file=open('D:\\CK\\data.pkl','wb')
# cPickle.dump(data,write_file,-1)
# cPickle.dump(data_label,write_file,-1)
# write_file.close()
#
import os
import pydicom
import numpy
from matplotlib import pyplot
import _pickle as cPickle
# 用lstFilesDCM作为存放DICOM files的列表
lstFilesDCM = []

def pkl_generate(PathDicom):
    lstFilesDCM = []
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # 判断文件是否为dicom文件
                print(filename)
                lstFilesDCM.append(os.path.join(dirName, filename))  # 加入到列表中

    ## 将第一张图片作为参考图
    RefDs = pydicom.read_file(lstFilesDCM[0])  # 读取第一张dicom图片

    # 建立三维数组
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

    # 得到spacing值 (mm为单位)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    # 三维数据
    x = numpy.arange(0.0, (ConstPixelDims[0] + 1) * ConstPixelSpacing[0],
                     ConstPixelSpacing[0])  # 0到（第一个维数加一*像素间的间隔），步长为constpixelSpacing
    y = numpy.arange(0.0, (ConstPixelDims[1] + 1) * ConstPixelSpacing[1], ConstPixelSpacing[1])  #
    z = numpy.arange(0.0, (ConstPixelDims[2] + 1) * ConstPixelSpacing[2], ConstPixelSpacing[2])  #

    ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    # 遍历所有的dicom文件，读取图像数据，存放在numpy数组中
    for filenameDCM in lstFilesDCM:
        ds = pydicom.read_file(filenameDCM)
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

    return ArrayDicom

def main(PathDicom):
    f = open('store.pkl', 'wb')
     # 与python文件同一个目录下的文件夹
    dicom = pkl_generate(PathDicom)
    cPickle.dump(dicom, f)
    f.close()


if __name__ == '__main__':
   # PathDicom = "./s1/"
    PathDicom = "E:/CT/CT164142/"
    main(PathDicom)

