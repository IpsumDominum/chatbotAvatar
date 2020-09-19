Text2SpeechModel_select = ["dc_tts","RTVC","AudioSynth"]
Speech2LandmarkModel_select = ["Eskimez","speechdriven"]
LandMark2FaceModel_select = ["firstorder"]
ChatBotModel_select = ["gpt2"]

ChatBotModel = ChatBotModel_select[0]
Text2SpeechModel = Text2SpeechModel_select[0]
Speech2LandmarkModel = Speech2LandmarkModel_select[1]
LandMark2FaceModel = LandMark2FaceModel_select[0]

import time

print("CHATBOT USING: ",ChatBotModel)
print("Text2Speech USING: ",Text2SpeechModel)
print("Speech2Landmark USING: ",Speech2LandmarkModel)
print("LandMark2Face USING: ",LandMark2FaceModel)

time.sleep(1)
