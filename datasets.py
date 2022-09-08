from collections import defaultdict
import json
import random
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

class InputExample(object):
    """A single training/test example for the language model."""
    def __init__(self,sent,input_ids,attention_mask,token_type_ids,
                visual_feat,v_len,
                audio_feat,a_len,
                masked_input_ids,masked_label,
                masked_visual,masked_visual_label,
                masked_audio,masked_audio_label,
                negative_visual,negativae_visual_len,
                negative_audio,negativae_audio_len):

        self.sent = sent
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.visual_feat = visual_feat
        self.audio_feat = audio_feat
        self.v_len = v_len
        self.a_len = a_len
        
        # For masked language modeling task
        self.masked_input_ids = masked_input_ids
        self.masked_label = masked_label

        self.masked_visual = masked_visual
        self.masked_visual_label = masked_visual_label

        self.masked_audio = masked_audio
        self.masked_audio_label = masked_audio_label

        # For matching task
        self.negative_visual = negative_visual
        self.negativae_visual_len = negativae_visual_len

        self.negative_audio = negative_audio
        self.negativae_audio_len = negativae_audio_len

    def __str__ (self):
        return "Text: " + self.sent + ",Visual: " + str(self.visual_feat.shape) + ",Audio: " + str(self.audio_feat.shape)
        

class MSA_data(Dataset):
    def __init__(self, dataset, is_list_of_data=False, tokenizer=None, collator=None, text_type="word", negative_sample=1, max_len=50):
        self.text = []
        self.visual = []
        self.audio = []
        self.label = []

        self.text_len = []
        self.visual_len = []
        self.audio_len = []
        
        self.text_type = text_type
        if text_type == "word":
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.collator = collator

        self.negative_sample = negative_sample
        
        if is_list_of_data:
            dataset = [sample for subdata in dataset for sample in subdata]
        non_sample = 0
        for sample in dataset:
            if len(sample) == 4:
                (_words, _glove, _visual, _acoustic), label, segment, (_g_l,_v_l,_a_l) = sample
                #_v_shape = _visual.shape
                #_a_shape = _acoustic.shape
                
                _visual = F.avg_pool1d(torch.tensor(_visual).T, kernel_size=4, stride=4).T
                _acoustic = F.avg_pool1d(torch.tensor(_acoustic).T, kernel_size=5, stride=5).T
                #_visual = torch.tensor(_visual[::4,:])
                #_acoustic = torch.tensor(_acoustic[::5,:])
                
                _visual = _visual
                _acoustic = _acoustic
                label = torch.tensor(label).squeeze()
                _v_l,_a_l = _visual.shape[0], _acoustic.shape[0]

            else:
                (_words, _glove, _visual, _acoustic), label, (_g_l,_v_l,_a_l) = sample
            
            if _v_l <= 0 or _a_l <= 0:
                non_sample += 1
                #print(sample)
                continue 
                
            
            if text_type == "word":
                sent = " ". join(_words)
                self.text.append(sent)
                #self.text_len.append(_g_l)
            elif text_type == "glove":
                self.text.append(_glove)
                self.text_len.append(_g_l)
            
            self.visual.append(_visual)
            self.audio.append(_acoustic)
            self.label.append(label)


            self.visual_len.append(_v_l)
            self.audio_len.append(_a_l)


        #print("Khong co:",non_sample)
        self.len = len(self.text)
    
    def random_visual_feat(self, index):
        new_index = random.randint(0, self.__len__()-1)
        while new_index == index:
            new_index = random.randint(0, self.__len__()-1)
        
        return self.visual[new_index], self.visual_len[new_index]

    def random_audio_feat(self, index):
        new_index = random.randint(0, self.__len__()-1)
        while new_index == index:
            new_index = random.randint(0, self.__len__()-1)
        
        return self.audio[new_index], self.audio_len[new_index]

    def mask_visual_feat(self, visual, v_l, index=None):
        # rand_masked_len = random.randint(5, 15)
        # rand_visual_index = random.randint(0,v_l-1-rand_masked_len)
        idx = 0
        mask_feats = torch.clone(torch.tensor(visual))
        feat_mask = torch.zeros(v_l, dtype=torch.float32)

        while idx < v_l - 15:
            prob = random.random()
            if prob < 0.1:
                rand_masked_len = random.randint(5, 15)
                mask_feats[idx:idx+rand_masked_len] = 0.
                feat_mask[idx:idx+rand_masked_len] = 1.
                idx += rand_masked_len
        return mask_feats, feat_mask

    def mask_audio_feat(self, audio, a_l, index=None):
        # rand_masked_len = random.randint(5, 15)
        # rand_visual_index = random.randint(0,v_l-1-rand_masked_len)
        idx = 0
        mask_feats = torch.clone(torch.tensor(audio))
        feat_mask = torch.zeros(a_l, dtype=torch.float32)

        while idx < a_l - 15:
            prob = random.random()
            if prob < 0.1:
                rand_masked_len = random.randint(5, 15)
                mask_feats[idx:idx+rand_masked_len] = 0.
                feat_mask[idx:idx+rand_masked_len] = 1.
                idx += rand_masked_len
        return mask_feats, feat_mask

    def mask_langugae(self, input_ids):
        masked_encoding = self.collator([input_ids])
        return masked_encoding['input_ids'].squeeze(), masked_encoding['labels'].squeeze()

    def __getitem__(self, index):
        text = self.text[index]
        visual = self.visual[index]
        audio = self.audio[index]
        label = self.label[index]
        
        if self.text_type == "word":
            t_l = None
        elif self.text_type == "glove":
            t_l = self.text_len[index]
        
        v_l = self.visual_len[index]
        a_l = self.audio_len[index]
        
        if self.text_type == "word":
            encoded_sent = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                # pad_to_max_length=True,
                padding='max_length',
                return_tensors='pt'     # Return PyTorch (pt) tensors
            )
            input_ids = encoded_sent['input_ids'].squeeze()
            attention_mask = encoded_sent['attention_mask'].squeeze()
            token_type_ids = encoded_sent['attention_mask'].squeeze()
        
        #Masked language
        masked_input_ids, masked_label = self.mask_langugae(input_ids)

        #Masked visual
        masked_visual_feat, masked_visual_feat_mask = self.mask_visual_feat(visual, v_l)
        
        #Masked audio
        masked_audio_feat, masked_audio_feat_mask = self.mask_audio_feat(audio, a_l)

        #Negative sampling
        negative_visual_feat, negative_visual_mask = self.random_visual_feat(index)
        negative_audio_feat, negative_audio_mask = self.random_audio_feat(index)
        
        return InputExample(
                sent=text,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                visual_feat=torch.tensor(visual),
                v_len=v_l,
                audio_feat=torch.tensor(audio),
                a_len=a_l,
                masked_input_ids=masked_input_ids,
                masked_label=masked_label,
                masked_visual=torch.tensor(masked_visual_feat),
                masked_visual_label=torch.tensor(masked_visual_feat_mask),
                masked_audio=torch.tensor(masked_audio_feat),
                masked_audio_label=torch.tensor(masked_audio_feat_mask),
                negative_visual=torch.tensor(negative_visual_feat),
                negativae_visual_len=torch.tensor(negative_visual_mask),
                negative_audio=torch.tensor(negative_audio_feat),
                negativae_audio_len=torch.tensor(negative_audio_mask),
        )

    def __len__(self):
        return self.len