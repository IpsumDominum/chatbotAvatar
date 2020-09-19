# -*- coding: utf-8 -*-
# /usr/bin/python3
"""
Options: 
    -dc_tts robust convolution/attention based tts
        '''
        By kyubyong park. kbpark.linguist@gmail.com.
        https://www.github.com/kyubyong/dc_tts
        '''
"""
try:
    from params import Text2SpeechModel
except ModuleNotFoundError:
    Text2SpeechModel_select = ["dc_tts","RTVC","AudioSynth"]
    Text2SpeechModel = Text2SpeechModel_select[2]
import os
import sys
import numpy as np
import tensorflow as tf
import simpleaudio as sa
from scipy.io.wavfile import write
from tqdm import tqdm
root_file_path = os.path.abspath(os.path.dirname(__file__))

if Text2SpeechModel == "dc_tts":    
    sys.path.append(os.path.join(root_file_path,"dc_tts"))
    from hyperparams import Hyperparams as hp
    from train import Graph
    from utils import *
    from data_load import load_data,text_normalize,load_vocab    
    from playsound import *
elif Text2SpeechModel == "RTVC":    
    sys.path.append(os.path.join(root_file_path,"RTVC"))
    from encoder.params_model import model_embedding_size as speaker_embedding_size
    from utils.argutils import print_args
    from synthesizer.inference import Synthesizer
    from keras.backend import clear_session
    from encoder import inference as encoder
    from vocoder import inference as vocoder
    from pathlib import Path
    import numpy as np
    import librosa
    import argparse
    import torch
    import sys
elif Text2SpeechModel =="AudioSynth":
    import os
    import sys
    sys.path.append(os.path.join(root_file_path,"AudioSynth"))
    sys.path.append(os.path.join(root_file_path,"AudioSynth","TensorFlowTTS/"))
    import tensorflow as tf
    import yaml
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.io.wavfile import write

    def do_synthesis(input_text, text2mel_model, vocoder_model, text2mel_name, vocoder_name,processor):
        input_ids = processor.text_to_sequence(input_text)
        # text2mel part
        if text2mel_name == "TACOTRON":
            _, mel_outputs, stop_token_prediction, alignment_history = text2mel_model.inference(
                tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                tf.convert_to_tensor([len(input_ids)], tf.int32),
                tf.convert_to_tensor([0], dtype=tf.int32)
            )
        elif text2mel_name == "FASTSPEECH":
            mel_before, mel_outputs, duration_outputs = text2mel_model.inference(
                input_ids=tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
                speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            )
        elif text2mel_name == "FASTSPEECH2":
            mel_before, mel_outputs, duration_outputs, _, _ = text2mel_model.inference(
                tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
                speaker_ids=tf.convert_to_tensor([0], dtype=tf.int32),
                speed_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
                f0_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
                energy_ratios=tf.convert_to_tensor([1.0], dtype=tf.float32),
            )
        else:
            raise ValueError("Only TACOTRON, FASTSPEECH, FASTSPEECH2 are supported on text2mel_name")

        # vocoder part
        if vocoder_name == "MELGAN" or vocoder_name == "MELGAN-STFT":
            audio = vocoder_model(mel_outputs)[0, :, 0]
        elif vocoder_name == "MB-MELGAN":
            audio = vocoder_model(mel_outputs)[0, :, 0]
        else:
            raise ValueError("Only MELGAN, MELGAN-STFT and MB_MELGAN are supported on vocoder_name")

        if text2mel_name == "TACOTRON":
            return mel_outputs.numpy(), alignment_history.numpy(), audio.numpy()
        else:
            return mel_outputs.numpy(), audio.numpy()

    from tensorflow_tts.inference import TFAutoModel
    from tensorflow_tts.inference import AutoConfig
    from tensorflow_tts.inference import AutoProcessor
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
        try:
            tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
        except RuntimeError as e:
            print(e)

class Text2Speech:
    def __init__(self):        
        if(Text2SpeechModel == "dc_tts"):
            self.g = Graph(mode="synthesize"); print("Text2Speech Tensorflow Graph loaded")
        elif(Text2SpeechModel=="RTVC"):
            enc_model_fpath = os.path.join(root_file_path,"RTVC","encoder/saved_models/pretrained.pt")
            syn_model_dir = os.path.join(root_file_path,"RTVC","synthesizer/saved_models/logs-pretrained")
            voc_model_fpath = os.path.join(root_file_path,"RTVC","vocoder/saved_models/pretrained/pretrained.pt")
            encoder.load_model(enc_model_fpath)
            self.synthesizer = Synthesizer(os.path.join(syn_model_dir,"taco_pretrained"), low_mem=False)
            vocoder.load_model(voc_model_fpath)
            in_fpath = os.path.join("/",*root_file_path.split("/")[:-1],"REF/refaudioRTVC/ref.wav")
            preprocessed_wav = encoder.preprocess_wav(in_fpath)
            original_wav, sampling_rate = librosa.load(in_fpath)
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            embed = encoder.embed_utterance(preprocessed_wav)
            self.embeds = [embed]
        elif(Text2SpeechModel=="AudioSynth"):
            taco_pretrained_config_path = os.path.join(root_file_path,'AudioSynth/TensorFlowTTS/examples/tacotron2/conf/tacotron2.v1.yaml')            
            tacotron2_config = AutoConfig.from_pretrained(taco_pretrained_config_path)
            taco_path = os.path.join(root_file_path,"AudioSynth/tacotron2-120k.h5")
            self.tacotron2 = TFAutoModel.from_pretrained(config=tacotron2_config,pretrained_path=taco_path,training=False,name="tacotron2")

            melgan_stft_pretrained_config_path = os.path.join(root_file_path,'AudioSynth/TensorFlowTTS/examples/melgan.stft/conf/melgan.stft.v1.yaml')
            melgan_stft_config = AutoConfig.from_pretrained(melgan_stft_pretrained_config_path)
            melgan_stft_path = os.path.join(root_file_path,"AudioSynth/melgan.stft-2M.h5")
            self.melgan_stft = TFAutoModel.from_pretrained(
                config=melgan_stft_config,
                pretrained_path=melgan_stft_path,
                name="melgan_stft"
                )
            self.processor = AutoProcessor.from_pretrained(pretrained_path=os.path.join(root_file_path,"AudioSynth/ljspeech_mapper.json"))
            mels, alignment_history, audios = do_synthesis("Hello, how can I help you today?", self.tacotron2, self.melgan_stft, "TACOTRON", "MELGAN-STFT",self.processor)
    def synthesize_text_to_speech(self,lines):
        if(Text2SpeechModel == "dc_tts"):
            char2idx, idx2char = load_vocab()

            sents = [text_normalize(lines) + "E"]
            texts = np.zeros((len(sents), hp.max_N), np.int32)
            for i, sent in enumerate(sents):
                texts[i, :len(sent)] = [char2idx[char] for char in sent]
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1,allow_growth = True)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                sess.run(tf.global_variables_initializer())

                # Restore parameters
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
                saver1 = tf.train.Saver(var_list=var_list)
                saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
                print("Text2Mel Restored!")

                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
                saver2 = tf.train.Saver(var_list=var_list)
                saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
                print("SSRN Restored!")

                # Feed Forward
                ## mel
                L = texts
                Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
                prev_max_attentions = np.zeros((len(L),), np.int32)

                for j in tqdm(range(hp.max_T)):
                    _gs, _Y, _max_attentions, _alignments = \
                        sess.run([self.g.global_step, self.g.Y, self.g.max_attentions, self.g.alignments],
                                {self.g.L: L,
                                self.g.mels: Y,
                                self.g.prev_max_attentions: prev_max_attentions})
                    Y[:, j, :] = _Y[:, j, :]
                    prev_max_attentions = _max_attentions[:, j]

                # Get magnitude
                Z = sess.run(self.g.Z, {self.g.Y: Y})

                # Generate wav files
                #if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
                for i, mag in enumerate(Z):
                    wav = spectrogram2wav(mag)
                    #write(hp.sampledir + "/{}.wav".format(i+1), hp.sr, wav)
                    save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"OUT/temp.wav")
                    end = np.zeros((22050))
                    wav = np.concatenate((wav,end),axis=0)
                    write(save_path,hp.sr,wav)        
        elif Text2SpeechModel=="RTVC":            
            text = lines 
            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [text]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = self.synthesizer.synthesize_spectrograms(texts, self.embeds)
            spec = specs[0]
            generated_wav = vocoder.infer_waveform(spec)
            generated_wav = np.pad(generated_wav, (0, self.synthesizer.sample_rate), mode="constant")
            save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"OUT/temp.wav")
            end = np.zeros((22050))
            generated_wav = np.concatenate((generated_wav,end),axis=0)
            librosa.output.write_wav(save_path, generated_wav.astype(np.float32), 
                                    self.synthesizer.sample_rate)
            clear_session()
        elif(Text2SpeechModel=="AudioSynth"):
            mels, alignment_history, audios = do_synthesis(lines, self.tacotron2, self.melgan_stft, "TACOTRON", "MELGAN-STFT",self.processor)
            end = np.zeros((22050))
            audios = np.concatenate((audios,end),axis=0)
            write(os.path.join(root_file_path,"OUT/temp.wav"),22050,audios.astype(np.float32))
if __name__=="__main__":    
    tts = Text2Speech()
    tts.synthesize_text_to_speech("Hi there.")
