#from synthesize import synthesize,initialize_models
import os
import sys
try:
    from bText2Speech import Text2Speech
except ImportError:
    from bText2Speech.__init__ import bText2Speech
try:
    from cSpeech2Landmark import Speech2Landmark
except ImportError:
    from cSpeech2Landmark.__init__ import Speech2LandMark
try:
    from dLandMark2Face import LandMark2Face
except ImportError:
    from dLandMark2Face.__init__ import LandMark2Face
try:
    from bText2Speech import Text2Speech
except ImportError:
    from bText2Speech.__init__ import bText2Speech
try:
    from cSpeech2Landmark import Speech2Landmark
except ImportError:
    from cSpeech2Landmark.__init__ import Speech2LandMark
try:
    from dLandMark2Face import LandMark2Face
except ImportError:
    from dLandMark2Face.__init__ import LandMark2Face

def quit_anya(bot_path):
    with open(os.path.join(bot_path,"AnyaBot/Queue/quit_signal"),'w') as file:
        file.write("0")
if __name__=="__main__":
    queue_path = "/home/ipsum/fatssd/Anya/AnyaBot/Queue"
    TTS = Text2Speech()
    STL = Speech2LandMark()
    LTF = LandMark2Face()

    print("=============ANYA v1.1==============")
    while True:
        if(len(os.listdir(queue_path))>0):
            print("ANYA SPEAKING...")
            while(len(os.listdir(queue_path))>0):
                pass
            print("You may now speak. :)")
        prompt = input(">>> ")
        while not prompt:
            prompt = input(">>> ")
        if(prompt=="quit"):
            break          
        TTS.synthesize_text_to_speech(prompt)
        STL.synthesize_speech_to_landmark()
        LTF.synthesize_landmark_to_face()
    #quit_anya(bot_path)

    

    
