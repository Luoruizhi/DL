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
        os.makedirs(dir)

def getImage(dir_list_file):
        # 获取人脸图片
        Image_list=[]
        for fpathe,dirs,fs in os.walk(dir_list_file):
            for f in fs:
                portion = os.path.splitext(f)
                if portion[1] ==".jpg" or portion[1] ==".jpeg" or portion[1] ==".png":
                    Image_list.append(os.path.join(fpathe,f))
                else:
                    continue
        return Image_list

def loadListFromFile(file_name):
    fin = open(file_name, 'r')
    file_list = []
    for line in fin:        
        file_list.append(line.strip("\n"))
    fin.close()
    return file_list

def loadBaiduBoxesFile(file_name):
    boxes = []
    skip=0
    f = open(file_name, encoding='utf-8')
    detect_result = json.load(f)    
    f.close()
    if detect_result['error_msg']!='SUCCESS' or int(detect_result['result']['face_num'])>6:
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
        box = [label, float(x), float(y), float(w), float(h), float(rotation)]
        boxes.append(box)
    return boxes,skip
    
def loadLocalBoxesFile(file_name):
    boxes = []
    f = open(file_name, encoding='utf-8')
    detect_result = json.load(f)    
    f.close()
    face_num = len(detect_result['people'])
    for i in range(face_num):            
        label = i+1             
        x = detect_result['people'][i]['face_rectangles'][0]
        y = detect_result['people'][i]['face_rectangles'][1]
        w = detect_result['people'][i]['face_rectangles'][2]
        h = detect_result['people'][i]['face_rectangles'][3]
        box = [label, float(x), float(y), float(w), float(h)]
        boxes.append(box)
    return boxes

def getBaiduTwoPoints(box, width, height):
    """
    computing two points
    :param box: [label, float(x), float(y), float(w), float(h), float(rotation)]
    :param width
    :param height
    :return: label (left,top,right,bottom)
    """    
    label = box[0]
    x = box[1]
    y = box[2]
    w = box[3]
    h = box[4]
    angle=box[5]
    angle1=-int(angle)    
    x_1 = int( x )
    y_1 = int( y )
    x_2 = int( x+w )
    y_2 = int( y+h )
    # Calculate the slope
    k1 = 0
    k2 = float(h/w)        
    # Calculate direction vector
    x = np.array([1,k1])
    y = np.array([1,k2])
    # Calculate the length
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    # You compute the Angle based on the vector and then you round it
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
    
    x1=int(max(min(x_1,x4)*(1-0.06*w/width),0))
    y1=int(max(min(y_1,y2)*(1-0.06*h/height),0))
    x2=int(min(max(x2,x3)*(1+0.06*w/width),width-1))
    y2=int(min(max(y4,y3)*(1+0.06*h/height),height-1))
    w=x2-x1
    h=y2-y1
    return label, (x1, y1, x2, y2)

def getLocalTwoPoints(box, width, height):
    """
    computing two points
    :param box: [label, float(x), float(y), float(w), float(h), float(rotation)]
    :param width
    :param height
    :return: label (left,top,right,bottom)
    """    
    label = box[0]
    x = box[1]
    y = box[2]
    w = box[3]
    h = box[4]
    
    x_1 = int( x )
    y_1 = int( y )
    x_2 = int( x+w )
    y_2 = int( y+h )
    x1=int(max(x_1,0))
    y1=int(max(y_1,0))
    x2=int(min(x_2,width-1))
    y2=int(min(y_2,height-1))
    return label, (x1, y1, x2, y2)

def computeIou(rec1, rec2):
    """
    computing IoU
    :param rec1: (x0, y0, x1, y1), which reflects
            (left, top, right,bottom)
    :param rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1]) 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2 
    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3])
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return intersect / (sum_area - intersect)
 
def computeIntersection(rec1, rec2):
    """
    computing intersection
    :param rec1: (x0, y0, x1, y1), which reflects
            (left, top, right,bottom)
    :param rec2: (x0, y0, x1, y1)
    :return: Intersection of two reces
    """
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3], rec2[3]) 
    bottom_line = max(bottom_line,rec1[3])
    return left_line,right_line,top_line,bottom_line


def selectBoxes(baidu_boxes,local_boxes,width,height):
    """
    select boxes
    :param baidu_boxes: [[label, float(top), float(left), float(w), float(h)]...]
    :param local_boxes: [[label, float(top), float(left), float(w), float(h)]...]
    :param width: width of picture
    :param height: height of picture
    :return: list of calculated box
    """
    final_boxes=[]
    for i in range(len(baidu_boxes)):
        baidu_box = baidu_boxes[i]
        baidu_label, baidu_parameters = getBaiduTwoPoints(baidu_box, width, height)
        box=[]
        max_iou=0
        for j in range(len(local_boxes)):
            local_box=local_boxes[j]            
            localLabel, local_parameters = getLocalTwoPoints(local_box, width, height)
            iou=computeIou(baidu_parameters,local_parameters)
            if iou<0.1:
                continue
            if(iou>max_iou):       
                left_line,right_line,top_line,bottom_line=computeIntersection(baidu_parameters,local_parameters)
                left=left_line
                box_width=right_line-left_line
                top=top_line
                box_height=bottom_line-top_line
                box=[baidu_label,left,top,box_width,box_height]
                max_iou=iou
        if max_iou>=0.1:         
            final_boxes.append(box)
    return final_boxes
    


def writeNewJson(baidu_boxes,local_boxes,width,height,new_json_name):
    result={}
    result['version']=1.2
    result['people']=[]
    final_boxes=selectBoxes(baidu_boxes,local_boxes,width,height)
    for i in range(len(final_boxes)):
        final_box = final_boxes[i]
        result['people'].append({})
        result['people'][i]['pose_keypoints_2d']=[]
        result['people'][i]['face_keypoints_2d']=[]
        result['people'][i]['face_rectangles']=[final_box[1],final_box[2],final_box[3],final_box[4]]        
        result['people'][i]['hand_left_keypoints_2d']=[]
        result['people'][i]['hand_right_keypoints_2d']=[]
        result['people'][i]['pose_keypoints_3d']=[]
        result['people'][i]['hand_left_keypoints_3d']=[]
        result['people'][i]['hand_right_keypoints_3d']=[]
    with open(new_json_name,"w") as f:
        json_result=json.dumps(result,indent=4)
        f.write(json_result)
        print("加载入文件完成...")



if __name__ == "__main__":
    if len(sys.argv) < 6:
        print ("<fin_file_list> <img_dir> <baidu_json_dir> <local_json_dir> <fout_dir>")
        sys.exit()
    file_list_name = sys.argv[1]
    img_dir=sys.argv[2]
    baidu_json_dir = sys.argv[3]
    local_json_dir = sys.argv[4]    
    fout_dir = sys.argv[5]
    # file_list_name = "D:/罗睿智/face_label_generate/valid_data/"
    # json_dir="D:/罗睿智/face_label_generate/baidu_valid_label/"
    # fout_dir = "D:/罗睿智/face_label_generate/new_json/"
    file_list = loadListFromFile(file_list_name)
    # file_list = getImage(file_list_name)
    mkDir(fout_dir)    
    for file_name in file_list:
        #print(file_name)
        portion = os.path.splitext(file_name)
        json_name = portion[0]+".json" 
        local_json=portion[0]+"_keypoints.json" 
        img_name=img_dir+file_name     
        new_json_name=fout_dir + json_name
        new_json_dir=os.path.split(new_json_name)[0]
        mkDir(new_json_dir)
        baidu_json_name=baidu_json_dir + json_name 
        local_json_name=local_json_dir+local_json
        print(img_dir,baidu_json_name,local_json_name,new_json_name)
        if (not os.path.exists(baidu_json_name)) or (not os.path.exists(local_json_name)):
            continue         
        baidu_boxes, skip = loadBaiduBoxesFile(baidu_json_name)
        if skip==1:
            continue
        local_boxes=loadLocalBoxesFile(local_json_name)
        img = cv2.imread(img_name)
        height = img.shape[0]
        width = img.shape[1]
        nchannels = img.shape[2]    
        writeNewJson(baidu_boxes,local_boxes,width,height,new_json_name)
       

