
# -*- coding: utf-8 -*-
# /usr/bin/python3
"""
Options: 
    -gpt2
"""
try:
    from params import ChatBotModel
except ModuleNotFoundError:
    ChatBotModel_select = ["gpt2"]
    ChatBotModel = ChatBotModel_select[0]

import json
import os,sys
import numpy as np
import tensorflow as tf
import re
root_file_path = os.path.abspath(os.path.dirname(__file__))

if ChatBotModel == "gpt2":    
    sys.path.append(os.path.join(root_file_path,"gpt2"))
    import src.model as model
    import src.sample as sample
    import src.encoder as encoder


class ChatBot:
    def __init__(self):
        if ChatBotModel =="gpt2":
            model_name='355M'
            seed=None
            length=None
            temperature=0.7
            top_k=40
            top_p=0.2
            models_dir= os.path.join(root_file_path,'gpt2/models')
            models_dir = os.path.expanduser(os.path.expandvars(models_dir))

            self.enc = encoder.get_encoder(model_name, models_dir)
            hparams = model.default_hparams()
            with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
                hparams.override_from_dict(json.load(f))

            if length is None:
                length = hparams.n_ctx // 2
            elif length > hparams.n_ctx:
                raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3,allow_growth = True)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.context = tf.placeholder(tf.int32, [1, None])
            np.random.seed(seed)
            tf.set_random_seed(seed)
            self.output = sample.sample_sequence(
                hparams=hparams, length=length,
                context=self.context,
                batch_size=1,
                temperature=temperature, top_k=top_k, top_p=top_p
            )
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
            saver.restore(self.sess, ckpt)
    def chat(self,prompt):
        if(ChatBotModel=="gpt2"):    
            input_prompt = "Me: \"{}\"".format(prompt)
            context_tokens = self.enc.encode(input_prompt)
            out = self.sess.run(self.output, feed_dict={
                self.context: [context_tokens]
            })[:, len(context_tokens):]
            text = self.enc.decode(out[0])
            return text.split("\n")[2][5:-1]   

if __name__ =="__main__":
    model = ChatBot()
    response_save_path = os.path.join("/",*root_file_path.split("/")[:-1],"TEMP")
    while(True):
        print("WAITING ON FILE")
        while(not os.path.isfile("gpt2_prompt/ishere_signal")):
            pass
        os.remove("gpt2_prompt/ishere_signal")
        if(os.path.isfile("gpt2_prompt/prompt.txt")):
            with open("gpt2_prompt/prompt.txt",'r') as file:
                prompt = file.read()
            os.remove("gpt2_prompt/prompt.txt")
            if(prompt.startswith("raw:")):
                response = prompt[5:]
            else:
                response = model.chat(prompt)
            with open(os.path.join(response_save_path,"response.txt"),"w") as file:
                file.write(response)
            with open(os.path.join(response_save_path,"response_is_here_signal"),"w") as file:
                file.write("0")
        else:
            print("'ishere_signal' received but no prompt file found!!!")
        
