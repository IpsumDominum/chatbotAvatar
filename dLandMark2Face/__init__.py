# -*- coding: utf-8 -*-
# /usr/bin/python3
"""
Options:
    -Eskimez 

"""
try:
    from params import LandMark2FaceModel
except ModuleNotFoundError:
    LandMark2FaceModel_select = ["firstorder"]
    LandMark2FaceModel = LandMark2FaceModel_select[0]
import os
import sys
import shutil
root_file_path = os.path.dirname(os.path.abspath(__file__))

if  LandMark2FaceModel == "firstorder":
    sys.path.append(os.path.join(root_file_path,"firstorder"))
    import os
    import yaml
    from argparse import ArgumentParser
    from tqdm import tqdm
    import imageio
    import numpy as np
    from skimage.transform import resize
    import torch
    from sync_batchnorm import DataParallelWithCallback
    from firstorder_modules.generator import OcclusionAwareGenerator
    from firstorder_modules.keypoint_detector import KPDetector
    from animate import normalize_kp
    from scipy.spatial import ConvexHull
    import cv2
    from demo import load_checkpoints,make_animation
    

class LandMark2Face:
    def __init__(self):
        if(LandMark2FaceModel=="firstorder"):
            source_image = imageio.imread(os.path.join("/",*root_file_path.split("/")[:-1],"REF/current/ref.png"))            
            self.source_image = resize(source_image, (256, 256))[..., :3]
            self.generator, self.kp_detector = load_checkpoints(config_path=os.path.join(root_file_path,"firstorder/config/vox-adv-256.yaml"), 
                                                      checkpoint_path=os.path.join(root_file_path,"firstorder/vox-cpk.pth.tar"))
    def synthesize_landmark_to_face(self):
        if(LandMark2FaceModel=="firstorder"):
            reader = imageio.get_reader(os.path.join("/",*root_file_path.split("/")[:-1],"cSpeech2Landmark/OUT/out.mp4"))
            self.fps = reader.get_meta_data()['fps']
            self.fps = self.fps -5
            reader.close()
            source_image = imageio.imread(os.path.join("/",*root_file_path.split("/")[:-1],"REF/current/ref.png"))
            self.source_image = resize(source_image, (256, 256))[..., :3]
            driving_video = imageio.mimread(os.path.join("/",*root_file_path.split("/")[:-1],"cSpeech2Landmark/OUT/out.mp4"), memtest=False)
            self.driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
            
            cv2.destroyAllWindows()
            predictions = make_animation(self.source_image, self.driving_video, self.generator, self.kp_detector)
            imageio.mimsave(os.path.join(root_file_path,"OUT/face.mp4"), predictions, fps=self.fps)
            #os.chdir("../finalface")
            #os.system("bash test.sh")        
            #os.chdir("/home/ipsum/fatssd/Anya/TTS")
            #stabilize("../finalface/result.mp4","stabilized.mp4")
            cmd = 'ffmpeg -y -i '+'dLandMark2Face/OUT/face.mp4 -i '+'bText2Speech/OUT/temp.wav -c:v copy -c:a aac -strict experimental fOUTPUT/queued.mp4'
            os.system(cmd)
            shutil.copy("fOUTPUT/queued.mp4","Queue/queued.mp4")
            
if __name__=="__main__":    
    LTF = LandMark2Face()
    LTF.synthesize_landmark_to_face()
    
    
