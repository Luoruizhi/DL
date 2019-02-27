# -*- coding: utf-8 -*-

import base64
import json
import urllib, sys, requests
import os, shutil
import time


class BaiduPicIndentify:
	def __init__(self,list_file,root_fin,root_fout):
		self.AK = "MGQnLeZVhcIf7E6MGHfY7QSZ"
		self.SK = "GjF8go4HcdGBhbUnPDPgvk5pGL2BA0Sg "
		self.list_file = list_file
		self.root_fin = root_fin
		self.root_fout = root_fout
		self.headers = {
            "Content-Type": "application/json; charset=UTF-8"
		}

	def getAccessToken(self):
		host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + self.AK + '&client_secret=' + self.SK        
		request = urllib.request.Request(host)
		request.add_header('Content-Type', 'application/json; charset=UTF-8')
		response = urllib.request.urlopen(request)
		content = response.read()
		content=eval(content)
		return content['access_token'] 

	def imgToBASE64(self,Image_path):
		with open(Image_path,'rb') as f:
			base64_data = base64.b64encode(f.read())
			return base64_data
	
	def mkDir(self,dir):
		if os.path.exists(dir):
			return
		else:
			os.makedirs(dir)

	def loadListFromFile(self):
		fin = open(self.list_file, 'r')
		file_list = []
		for line in fin:        
			file_list.append(line.strip("\n"))
		fin.close()
		return file_list

	def getImage(self):
		# 获取人脸图片
		Image_list=[]
		for fpathe,dirs,fs in os.walk(self.dir_list_file):
			for f in fs:
				portion = os.path.splitext(f)
				if portion[1] ==".jpg" or portion[1] ==".jpeg" or portion[1] ==".png":
					Image_list.append(os.path.join(fpathe,f))
				else:
					continue
		return Image_list

	def detectFace(self):
		# 人脸检测与属性分析
		# Image_list = self.getImage()
		Image_list = self.loadListFromFile()
		self.mkDir(self.root_fout)		
		for img_src in Image_list:		
			img_BASE64 = self.imgToBASE64(self.root_fin+img_src)
			request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"
			post_data = {
				"image": img_BASE64,
				"image_type": "BASE64",
				"face_field": "gender,age,race,landmark,expression",
				"max_face_num": "10"
			}
			access_token = self.getAccessToken()
			request_url = request_url + "?access_token=" + access_token
			response = requests.post(url=request_url, data=post_data, headers=self.headers)
			json_result = json.loads(response.text)
			label_name=self.root_fout+img_src
			label_path=os.path.split(label_name)[0]
			self.mkDir(label_path)
			portion = os.path.splitext(label_name)
			new_label_name = portion[0]+".json"
       		with open(new_label_name,"w") as f:
				response=json.dumps(json_result,indent=4)
				f.write(response)
				print("加载入文件完成...")
			time.sleep(0.5)
			# dir_out = os.path.join(self.root_fin, img_name)
			# shutil.copyfile(img_src,dir_out) 
	
		
if __name__=='__main__':
	print(len(sys.argv))
	print(sys.argv[0],sys.argv[1],sys.argv[2])
	if len(sys.argv) < 4:
		print ("<fin_dir_list> <fin_root_dir> <fout_root_dir>")	
		sys.exit()
	dir_list_file = sys.argv[1]
	root_fin = sys.argv[2]
	root_fout = sys.argv[3]
	baiduDetect = BaiduPicIndentify(dir_list_file,root_fin,root_fout)
	baiduDetect.detectFace()


