# -*- coding: utf-8 -*-
# /usr/bin/python3
"""
Options:
    -Eskimez 

"""
try:
    from params import Speech2LandmarkModel
except ModuleNotFoundError:
    Speech2LandmarkModel_select = ["Eskimez","speechdriven"]
    Speech2LandmarkModel = Speech2LandmarkModel_select[1]


import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import os, shutil, subprocess
from keras import backend as K
from keras.models import Model, Sequential, load_model
from tqdm import tqdm
import argparse
import cv2
import torch
import shutil, subprocess
import sys

root_file_path = os.path.abspath(__file__)

if  Speech2LandmarkModel == "Eskimez":
    sys.path.append(os.path.join(os.path.dirname(root_file_path),"Eskimez"))    
    import audio_utils
    from generate import *
    from stabilization import stabilize
    config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
    )
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.1

    # Create a session with the above options specified.
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)
elif Speech2LandmarkModel== "speechdriven":
    sys.path.append(os.path.join(os.path.dirname(root_file_path),"speech-driven-animation"))    
    import sda
    import scipy.io.wavfile as wav
    from PIL import Image
    import numpy as np
    from skimage import io
    import cv2
    from PIL import Image, ImageDraw
    import face_recognition
    from stabilization import stabilize


class Speech2LandMark:
    def __init__(self):
        if(Speech2LandmarkModel=="Eskimez"):
            self.audio_landmark_model = initiate_model()            
        elif(Speech2LandmarkModel=="speechdriven"):
            self.va = sda.VideoAnimator(gpu=0,model_path="timit")
            
            self.still_frame = cv2.imread(os.path.join(os.path.dirname(root_file_path),"example/d.png"))
            self.still_frame = cv2.cvtColor(self.still_frame,cv2.COLOR_BGR2RGB)
    def synthesize_speech_to_landmark(self):
        if(Speech2LandmarkModel=="Eskimez"):
            if __name__=="__main__":
                audio_path = "../bText2Speech/OUT/temp.wav"
            else:
                audio_path = "./bText2Speech/OUT/temp.wav"
            abs_path = os.path.abspath(os.path.dirname(__file__))
            if(os.path.isfile(audio_path)):
                with sess.as_default():
                    with sess.graph.as_default():
                        landmarks = generate_landmarks(audio_path,self.audio_landmark_model,abs_path)
            else:
                raise FileNotFoundError("Audio path not found, please use the pipeline correctly")
            return landmarks
        elif(Speech2LandmarkModel=="speechdriven"):                    
            if __name__=="__main__":
                audio_path = "../bText2Speech/OUT/temp.wav"
            else:
                audio_path = "./bText2Speech/OUT/temp.wav"
            self.fs, self.audio_clip = wav.read(os.path.join("bText2Speech","OUT","temp.wav"))
            self.audio_clip = self.audio_clip*1000
            self.audio_clip = self.audio_clip.astype(np.uint8)
            # Define the codec and create VideoWriter object
            Mouth = [[48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], \
                 [57, 58], [58, 59], [59, 48] ]
            Mouth2 = [[60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], \
                 [66, 67], [67, 60]]
            #   out = cv2.VideoWriter('output.mp4', -1, 20.0, (640,480))
            faces = []
            move = 0.5
            move_idx = 0
            size = 256
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            save_temp_dir = os.path.join(os.path.dirname(root_file_path),'OUT/temp.mp4')
            save_dir = os.path.join(os.path.dirname(root_file_path),'OUT/out.mp4')
            out = cv2.VideoWriter(save_temp_dir,fourcc, 25.0, (size,size))
            """
            Randomly Choose one of the actors 
            """
            actor = np.random.choice(["2.avi"])
            cap = cv2.VideoCapture(os.path.join(os.path.dirname(root_file_path),"Eskimez","actor/{}".format(actor)))
            
            vid, aud = self.va(self.still_frame, self.audio_clip, fs=self.fs)
            count = 0
            for idx in range(vid.shape[0]):        
                image = np.array(vid[idx])
                image = np.swapaxes(image,0,2)
                image = np.swapaxes(image,0,1).astype(np.uint8)
                #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                if(cap.isOpened()):
                    ret, face = cap.read()
                    face = cv2.cvtColor(face,cv2.COLOR_RGB2BGR)
                    if(count%2==0 and idx<vid.shape[0]-25):
                        face_rec = face_recognition.face_landmarks(image)[0]
                        item = face_rec
                        item2 = []
                        for rec in face_rec.values():
                            item2 += rec    
                        item2 = np.array(item2)
                    scale = 2.5
                    shiftx = 0
                    shifty = -40
                    #scale = 2.8
                    #shiftx = -10
                    #shifty = -60
                    #scale = 3
                    #shiftx = -20
                    #shifty = -80
                    count +=1

                    mouth_polygon = []
                    for mouth_item in Mouth:
                        keypoint = item2[mouth_item][0]
                        location = [int(keypoint[0]*scale+shiftx),int(keypoint[1]*scale+shifty)]
                        mouth_polygon.append(location)
                    mouth_polygon = np.array([mouth_polygon],dtype=np.int32)
                    if(idx<vid.shape[0]-25):
                        cv2.fillPoly(face,mouth_polygon,(0,0,0),lineType=cv2.LINE_AA)
                    else:
                        cv2.fillPoly(face,mouth_polygon,(40,40,255),lineType=cv2.LINE_AA)
                    mouth_polygon = []
                    for mouth_item in Mouth2:
                        keypoint = item2[mouth_item][0]
                        location = [int(keypoint[0]*scale+shiftx),int(keypoint[1]*scale+shifty)]
                        mouth_polygon.append(location)
                    mouth_polygon = np.array([mouth_polygon],dtype=np.int32)
                    if(idx<vid.shape[0]-25):
                        cv2.fillPoly(face,mouth_polygon,(0,0,0),lineType=cv2.LINE_AA)
                    else:
                        cv2.fillPoly(face,mouth_polygon,(40,40,255),lineType=cv2.LINE_AA)

                    mouth_polygon = []  
                    for keypoint in item['bottom_lip']:
                        #keypoint = item[mouth_item][0]
                        location = [int(keypoint[0]*scale+shiftx),int(keypoint[1]*scale+shifty)]
                        mouth_polygon.append(location)
                    mouth_polygon = np.array([mouth_polygon],dtype=np.int32)
                    cv2.fillPoly(face,mouth_polygon,(40,40,255),lineType=cv2.LINE_AA)
                    
                    mouth_polygon = []
                    for keypoint in item['top_lip']:
                        #keypoint = item[mouth_item][0]
                        location = [int(keypoint[0]*scale+shiftx),int(keypoint[1]*scale+shifty)]
                        mouth_polygon.append(location)
                    mouth_polygon = np.array([mouth_polygon],dtype=np.int32)
                    cv2.fillPoly(face,mouth_polygon,(40,40,255),lineType=cv2.LINE_AA)
                    

                    #cv2.imshow("hi",face)
                    #k = cv2.waitKey(0)
                    #if(k==ord('q')):
                    #    break
                    out.write(face)
                else:
                    print("actor loading failed")
                    break                    
                #face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            
            cap.release()
            out.release()    
            #cv2.destroyAllWindows()
            stabilize(save_temp_dir,save_dir)
            #print("done")
            #print("saving")
            #va.save_video(vid,aud,"generated.mp4")
            #print("done")



if __name__=="__main__":    
    stl = Speech2LandMark()
    landmarks = stl.synthesize_speech_to_landmark()
