import cv2
import os
import sys
import random
import shutil
import math
from math import cos, sin,pi
import numpy as np
import json

def mkDir(dir):
    if os.path.exists(dir):
        return
    else:
        os.mkdir(dir)

def get_Image(dir_list_file):
        #获取人脸图片
        Image_list=[]
        for fpathe,dirs,fs in os.walk(dir_list_file):
            for f in fs:
                if f.split('.')[1]=='jpg':
                    path=os.path.join(fpathe,f)
                    path = path.replace('\\','/')
                    Image_list.append(path)
        return Image_list

# def loadListFromFile(file_name):
#     fin = open(file_name, 'r')
#     file_list = []
#     for line in fin:        
#         file_list.append(line.strip("\n"))
#     fin.close()
#     return file_list

def loadBoxesFile(file_name):
    boxes = []
    skip=0
    usability=0
    f = open(file_name, encoding='utf-8')
    detect_result = json.load(f)
    if detect_result['error_msg']!='SUCCESS':
        skip=1
        boxex=None
        return boxex,skip
    face_num = detect_result['result']['face_num']
    for i in range(face_num):            
        label = i+1             
        x = detect_result['result']['face_list'][i]['location']['left']
        y = detect_result['result']['face_list'][i]['location']['top']
        w = detect_result['result']['face_list'][i]['location']['width']
        h = detect_result['result']['face_list'][i]['location']['height']
        rotation = detect_result['result']['face_list'][i]['location']['rotation']
        if float(detect_result['result']['face_list'][i]['face_probability']) > 0.6: 
            # print(detect_result['result']['face_list'][i]['face_probability'])
            usability = 1  
        box = [label, float(x), float(y), float(w), float(h), float(rotation), usability]
        boxes.append(box)
    f.close()
    return boxes,skip
    
def getTwoPointsBox(box, width, height):
    label = box[0]
    x = box[1]
    y = box[2]
    w = box[3]
    h = box[4]
    angle=box[5]
    usability=box[6]
    if usability==0:
        label=0
    angle1=-int(angle)    
    x_1 = int( x )
    y_1 = int( y )
    x_2 = int( x+w )
    y_2 = int( y+h )
    #求出斜率
    k1 = 0
    k2 = float(h/w)
        
        #方向向量
    x = np.array([1,k1])
    y = np.array([1,k2])
        #模长
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
        #根据向量之间求其夹角并四舍五入
    Cobb = int((np.arccos(x.dot(y)/(float(Lx*Ly)))*180/np.pi)+0.5)
    angle2 = -Cobb+angle1

    r_1=w
    r_2=h
    r_3=(w**2+h**2)**0.5
    x2 = x_1 + r_1 * cos(angle1 * pi / 180)
    y2 = y_1 - r_1 * sin(angle1 * pi /180)
    x4 = x_1 + r_2 * sin(angle1 * pi / 180)
    y4 = y_1 + r_2 * cos(angle1 * pi /180)
    x3 = x_1 + r_3 * cos(angle2 * pi / 180)
    y3 = y_1 - r_3 * sin(angle2 * pi /180)
    if angle1>0:
        x3=x3*(1-0.04*w/width-0.04*angle1/180)
        y4=y4*(1-0.001*h/height-0.001*angle1/180)
    else:
        x_1=x_1*(1+0.04*w/width-0.04*angle1/180)
        y4=y4*(1 - 0.001*h/height + 0.001*angle1/180)
    
    x1=int(max(min(x_1,x4),0))
    y1=int(max(min(y_1,y2),0))
    x2=int(min(max(x2,x3),width))
    y2=int(min(max(y4,y3),height))
    return label, (x1, y1), (x2, y2)


if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print "<fin_file_list> <fout_dir>"
    #     sys.exit()
    # file_list_name = sys.argv[1]
    # fout_dir = sys.argv[2]
    file_list_name = "D:/罗睿智/Download/img_and_label"
    fout_dir = "D:/罗睿智/Download/output"
    # file_list = loadListFromFile(file_list_name)
    file_list = get_Image(file_list_name)
    mkDir(fout_dir)    
    for file_name in file_list:
        img_name=file_name.split('/')[-1]
        img_out=fout_dir+'/'+img_name
        json_name=file_name.replace('jpg','json')
        boxes, skip = loadBoxesFile(json_name)
        if skip==1:
            continue
        img = cv2.imread("img_and_label/"+img_name)
        height = img.shape[0]
        width = img.shape[1]
        nchannels = img.shape[2]        
        for i in range(len(boxes)):
            box = boxes[i]
            label, point1, point2 = getTwoPointsBox(box, width, height)
            if label==0:
                continue
            cv2.rectangle(img, point1, point2, (0, 255, int(255.0 / len(boxes) * i)), 2)
            cv2.putText(img, str(label), (int(point1[0])-5, int(point1[1])), cv2.FONT_HERSHEY_COMPLEX,0.25,(0,255,255), 1)
        cv2.imwrite("output/"+img_name, img)
        #shutil.copyfile(img_in, img_out)
        print (img_name)
        print (img_out)
