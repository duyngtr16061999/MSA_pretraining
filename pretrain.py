from ast import arg
from re import S
from modeiling import *
from datasets import MSA_data
from param import Args
from transformers import BertTokenizer,BertConfig,BertModel,DataCollatorForWholeWordMask

import collections
import os
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle

import wandb
wandb.login()

data_path = {
    "iemocap":"./checking/my_normalize_iemocap.pkl",
    "mosei":"./checking/my_normalize_mosei.pkl",
    "mosi":"./checking/my_normalize_mosi.pkl",
}

def get_loss_name(args):
    loss_name = []
    if args.task_mask_lm:
        loss_name.append("MaskLM")
    if args.task_mask_v:
        loss_name.append("MaskV")
    if args.task_mask_a:
        loss_name.append("MaskA")
    if args.matching_task:
        loss_name.append("Matching")
    if args.matching_visual:
        loss_name.append("MatchingV")
    if args.matching_audio:
        loss_name.append("MatchingA")
    return loss_name

VisualConfig.set_input_dim(
            Args.visual_feat_dim,
            Args.audio_feat_dim
        )
        
VisualConfig.set_layers(
            Args.vlayers,
            Args.alayers,
            Args.clayers
        )
        
VisualConfig.set_visual_config(
            Args.visual_hidden_dim,
            Args.visual_num_attention_heads,
            Args.visual_intermediate_size
        )
        
VisualConfig.set_audio_config(
            Args.audio_hidden_dim,
            Args.audio_num_attention_heads,
            Args.audio_intermediate_size
        )

VisualConfig.set_cross_config(
            Args.cross_hidden_dim,
            Args.cross_num_attention_heads,
            Args.cross_intermediate_size
        )

class InputFeatures(object):
    """A single training/test example for the language model."""
    def __init__(self,sent,input_ids,attention_mask,token_type_ids,
                visual_feat,visual_mask,
                audio_feat,audio_mask,
                masked_input_ids,masked_label,
                masked_visual,masked_visual_label,
                masked_audio,masked_audio_label,
                negative_visual,negativae_visual_mask,
                negative_audio,negativae_audio_mask):

        self.batch_sent = sent
        self.batch_input_ids = input_ids
        self.batch_attention_mask = attention_mask
        self.batch_token_type_ids = token_type_ids
        self.batch_visual_feat = visual_feat
        self.batch_audio_feat = audio_feat

        self.batch_visual_feat_am = visual_mask
        self.batch_audio_feat_am = audio_mask
        
        # For masked language modeling task
        self.batch_masked_input_ids = masked_input_ids
        self.batch_masked_label = masked_label

        self.batch_masked_visual = masked_visual
        self.batch_masked_visual_label = masked_visual_label

        self.batch_masked_audio = masked_audio
        self.batch_masked_audio_label = masked_audio_label

        # For matching task
        self.batch_negative_visual = negative_visual
        self.batch_negativae_visual_am = negativae_visual_mask

        self.batch_negative_audio = negative_audio
        self.batch_negativae_audio_am = negativae_audio_mask

def convert_example_to_features(examples,is_cuda=False)->InputFeatures:
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    batch_len = len(examples)
    visual_size = 35
    audio_size = 74

    batch_sent = []
    batch_input_ids = []
    batch_attention_mask = []
    batch_token_type_ids = []
    batch_visual_feat = []
    batch_audio_feat = []
    batch_v_len = []
    batch_a_len = []
    # For masked language modeling task
    batch_masked_input_ids = []
    batch_masked_label = []
    batch_masked_visual = []
    batch_masked_visual_label = []
    batch_masked_audio = []
    batch_masked_audio_label = []
    # For matching task
    batch_negative_visual = []
    batch_negativae_visual_len = []
    batch_negative_audio = []
    batch_negativae_audio_len = []

    for ie in examples:
        batch_sent.append(ie.sent)
        batch_input_ids.append(ie.input_ids)
        batch_attention_mask.append(ie.attention_mask)
        batch_token_type_ids.append(ie.token_type_ids)
        batch_visual_feat.append(ie.visual_feat)
        batch_audio_feat.append(ie.audio_feat)

        batch_v_len.append(ie.v_len)
        batch_a_len.append(ie.a_len)
        # For masked language modeling task
        batch_masked_input_ids.append(ie.masked_input_ids)
        batch_masked_label.append(ie.masked_label)
        batch_masked_visual.append(ie.masked_visual)
        batch_masked_visual_label.append(ie.masked_visual_label)
        batch_masked_audio.append(ie.masked_audio)
        batch_masked_audio_label.append(ie.masked_audio_label)
        # For matching task
        batch_negative_visual.append(ie.negative_visual)
        batch_negativae_visual_len.append(ie.negativae_visual_len)
        batch_negative_audio.append(ie.negative_audio)
        batch_negativae_audio_len.append(ie.negativae_audio_len)
    
    ### For text modality ###
    # For input
    batch_input_ids = torch.stack(batch_input_ids)
    batch_attention_mask = torch.stack(batch_attention_mask)
    batch_token_type_ids = torch.stack(batch_token_type_ids)
    # For masked input
    batch_masked_input_ids = torch.stack(batch_masked_input_ids)
    batch_masked_label = torch.stack(batch_masked_label)

    ### For visual and audio modalities 
    batch_v_len = torch.tensor(batch_v_len)
    batch_a_len = torch.tensor(batch_a_len)
    batch_negativae_visual_len = torch.tensor(batch_negativae_visual_len)
    batch_negativae_audio_len = torch.tensor(batch_negativae_audio_len)

    max_v_len = torch.max(batch_v_len)
    max_a_len = torch.max(batch_a_len)
    max_negative_v_len = torch.max(batch_negativae_visual_len)
    max_negative_a_len = torch.max(batch_negativae_audio_len)

    torch_batch_visual = torch.zeros((batch_len,max_v_len,visual_size), dtype=torch.float32)
    torch_batch_visual_am = torch.zeros((batch_len,max_v_len), dtype=torch.long)
    torch_batch_audio = torch.zeros((batch_len,max_a_len,audio_size), dtype=torch.float32)
    torch_batch_audio_am = torch.zeros((batch_len,max_a_len), dtype=torch.long)

    torch_batch_masked_visual = torch.zeros((batch_len,max_v_len,visual_size), dtype=torch.float32)
    torch_batch_masked_visual_label = torch.zeros((batch_len,max_v_len), dtype=torch.float32)
    torch_batch_masked_audio = torch.zeros((batch_len,max_a_len,audio_size), dtype=torch.float32)
    torch_batch_masked_audio_label = torch.zeros((batch_len,max_a_len), dtype=torch.float32)

    torch_batch_negative_visual = torch.zeros((batch_len,max_negative_v_len,visual_size), dtype=torch.float32)
    torch_batch_negative_visual_am = torch.zeros((batch_len,max_negative_v_len), dtype=torch.long)
    torch_batch_negative_audio = torch.zeros((batch_len,max_negative_a_len,audio_size), dtype=torch.float32)
    torch_batch_negative_audio_am = torch.zeros((batch_len,max_negative_a_len), dtype=torch.long)

    for i in range(batch_len):

        torch_batch_visual[i,:batch_v_len[i],:]=batch_visual_feat[i]
        torch_batch_visual_am[i,:batch_v_len[i]]=1
        torch_batch_audio[i,:batch_a_len[i],:]=batch_audio_feat[i]
        torch_batch_audio_am[i,:batch_a_len[i]]=1

        torch_batch_masked_visual[i,:batch_v_len[i],:]=batch_masked_visual[i]
        torch_batch_masked_visual_label[i,:batch_v_len[i]]=batch_masked_visual_label[i]

        torch_batch_masked_audio[i,:batch_a_len[i],:]=batch_masked_audio[i]
        torch_batch_masked_audio_label[i,:batch_a_len[i]]=batch_masked_audio_label[i]

        torch_batch_negative_visual[i,:batch_negativae_visual_len[i],:]=batch_negative_visual[i]
        torch_batch_negative_visual_am[i,:batch_negativae_visual_len[i]]=1
        torch_batch_negative_audio[i,:batch_negativae_audio_len[i],:]=batch_negative_audio[i]
        torch_batch_negative_audio_am[i,:batch_negativae_audio_len[i]]=1

    if is_cuda: 
        features = InputFeatures(
            sent=batch_sent,input_ids=batch_input_ids.cuda(),attention_mask=batch_attention_mask.cuda(),token_type_ids=batch_token_type_ids.cuda(),
            visual_feat=torch_batch_visual.cuda(),visual_mask=torch_batch_visual_am.cuda(),
            audio_feat=torch_batch_audio.cuda(),audio_mask=torch_batch_audio_am.cuda(),
            masked_input_ids=batch_masked_input_ids.cuda(),masked_label=batch_masked_label.cuda(),
            masked_visual=torch_batch_masked_visual.cuda(),masked_visual_label=torch_batch_masked_visual_label.cuda(),
            masked_audio=torch_batch_masked_audio.cuda(),masked_audio_label=torch_batch_masked_audio_label.cuda(),
            negative_visual=torch_batch_negative_visual.cuda(),negativae_visual_mask=torch_batch_negative_visual_am.cuda(),
            negative_audio=torch_batch_negative_audio.cuda(),negativae_audio_mask=torch_batch_negative_audio_am.cuda()
        )
    else:
        features = InputFeatures(
            sent=batch_sent,input_ids=batch_input_ids,attention_mask=batch_attention_mask,token_type_ids=batch_token_type_ids,
            visual_feat=torch_batch_visual,visual_mask=torch_batch_visual_am,
            audio_feat=torch_batch_audio,audio_mask=torch_batch_audio_am,
            masked_input_ids=batch_masked_input_ids,masked_label=batch_masked_label,
            masked_visual=torch_batch_masked_visual,masked_visual_label=torch_batch_masked_visual_label,
            masked_audio=torch_batch_masked_audio,masked_audio_label=torch_batch_masked_audio_label,
            negative_visual=torch_batch_negative_visual,negativae_visual_mask=torch_batch_negative_visual_am,
            negative_audio=torch_batch_negative_audio,negativae_audio_mask=torch_batch_negative_audio_am
        )
    
    return features

class MSA:
    def __init__(self, args) -> None:
        
        self.args = args
        wandb.init(project="msa_pretraining", name=args.wandb_name, config=args)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.col = DataCollatorForWholeWordMask(tokenizer=self.tokenizer,mlm=True,mlm_probability=0.3)

        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.LOSSES_NAME = get_loss_name(args)
        self.model = MMBERTPretraining(
            config=self.config,
            task_mask_lm=args.task_mask_lm,
            task_mask_v=args.task_mask_v,
            task_mask_a=args.task_mask_a,
            matching_task=args.matching_task,
            matching_visual=args.matching_visual,
            matching_audio=args.matching_audio
        )
        
        
        wandb.watch(self.model)
        # Loading datasets
        all_dataset = []
        for i in args.train.split(','):
            with open(data_path[i], 'rb') as f:
                all_dataset.append(pickle.load(f))
        # Train
        self.train_dataset  = MSA_data([i["train"] for i in all_dataset],True,self.tokenizer,self.col)
        # Valid
        self.val_dataset  = MSA_data([i["valid"] for i in all_dataset],True,self.tokenizer,self.col)

        self.train_dataloader = DataLoader(
                self.train_dataset, batch_size=args.batch_size,
                shuffle=True, num_workers=args.num_workers,
                collate_fn=lambda x: x,
                drop_last=True, pin_memory=True
            )
        self.val_dataloader = DataLoader(
                self.val_dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                collate_fn=lambda x: x,
                drop_last=False, pin_memory=True
            )

        self.accum_iter = None if args.accum_iter == -1 else args.accum_iter

        # Save config
        if args.save:
            self.save_config(args.save)
        
        # Weight initialization and loading
        if args.from_scratch:
            self.model.apply(self.model.init_bert_weights)
            self.model.bert.encoder.bertmodel = BertModel.from_pretrained('bert-base-uncased')
        
        if args.load is not None:
            if args.load_name is not None:
                self.load(args.load, args.load_name)
            else:
                self.load(args.load)

        
        # GPU Options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model = nn.DataParallel(self.model, device_ids=[0, 1])

    def train_batch(self, optim, batch, grad=False):
        loss, losses = self.model(batch)
        if self.args.multiGPU:
            loss = loss.mean()
            losses = losses.mean(0)

        if self.accum_iter is not None:
        # normalize loss to account for batch accumulation
            loss = loss / self.accum_iter

        loss.backward()
        
        if self.accum_iter is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            if grad:
                optim.step()
                optim.zero_grad()
        else:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            optim.step()
            optim.zero_grad()
        
        return loss.item(), losses.cpu()
    
    def valid_batch(self, batch):
        with torch.no_grad():
            loss, losses = self.model(batch)
            if self.args.multiGPU:
                loss = loss.mean()
                losses = losses.mean(0)
        return loss.item(), losses.cpu()

    def train(self):
        train_ld = self.train_dataloader

        # Optimizer
        from optimization import BertAdam
        batch_per_epoch = len(train_ld)
        t_total = int(batch_per_epoch * self.args.epochs)
        warmup_ratio = 0.05
        warmup_iters = int(t_total * warmup_ratio)
        # print("Batch per epoch: %d" % batch_per_epoch)
        # print("Total Iters: %d" % t_total)
        # print("Warm up Iters: %d" % warmup_iters)
        optim = BertAdam(self.model.parameters(), lr=self.args.lr, warmup=warmup_ratio, t_total=t_total)

        # Train
        best_eval_loss = 9595.
        for epoch in range(self.args.epochs):
            # Train
            self.model.train()
            total_loss = 0.
            total_losses = None
            
            step = 1
            for batch in tqdm(train_ld, total=len(train_ld)):
                if self.accum_iter is not None:
                    batch = convert_example_to_features(batch, False)
                else:
                    batch = convert_example_to_features(batch, True)
            
                if self.accum_iter is not None and ((step % self.accum_iter == 0) or step == len(train_ld)):
                    loss, losses = self.train_batch(optim, batch, True)
                    total_loss += loss
                else:
                    loss, losses = self.train_batch(optim, batch)
                    total_loss += loss
                if total_losses is None:
                    total_losses = torch.zeros_like(losses)
                total_losses += losses
                
                step += 1

            wandb.log({
                    "epoch_train_loss": total_loss / batch_per_epoch
                })
            print("The training loss for Epoch %d is %0.4f" % (epoch, total_loss / batch_per_epoch))
            losses_str = "The losses are "
            for name, loss in zip(self.LOSSES_NAME, total_losses.tolist()):
                wandb.log({
                    "epoch_train_" + name: loss / batch_per_epoch
                })
                losses_str += "%s: %0.4f " % (name, loss / batch_per_epoch)
            print(losses_str)
            print("----------------------------------------------------------")
            # Eval
            avg_eval_loss = self.evaluate_epoch(iters=-1)

            # Save
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                self.save(self.args.save_path,"BEST_EVAL_LOSS")
            self.save(self.args.save_path,"Epoch%02d" % (epoch+1))
            print("==========================================================") 
        
    def evaluate_epoch(self, iters: int=-1):
        self.model.eval()
        eval_ld = self.val_dataloader
        total_loss = 0.
        total_losses = None
        for i, batch in enumerate(eval_ld):
            if self.accum_iter is not None:
                batch = convert_example_to_features(batch, False)
            else:
                batch = convert_example_to_features(batch, True)
            loss, losses = self.valid_batch(batch)
            total_loss += loss
            if total_losses is None:
                total_losses = torch.zeros_like(losses)
            total_losses += losses
            if i == iters:
                break
        
        wandb.log({
                    "epoch_val_loss": total_loss / len(eval_ld)
                })
        print("The valid loss is %0.4f" % (total_loss / len(eval_ld)))
        losses_str = "The losses are "
        for name, loss in zip(self.LOSSES_NAME, total_losses.tolist()):
            wandb.log({
                    "epoch_val_" + name: loss / len(eval_ld)
                })
            losses_str += "%s: %0.4f " % (name, loss / len(eval_ld))
        print(losses_str)
        
        return total_loss / len(eval_ld)       
        
    def load(self, path, name=None):
        if name is not None:
            state_dict = torch.load(os.path.join(path, "%s.pth" % name))
        else:
            state_dict = torch.load(os.path.join(path, "model.pth"))
        self.model.load_state_dict(state_dict)
    
    def load_msa(self, path):
        pass
    
    def save(self, path, name=None):
        if not os.path.isdir(path):
            os.makedirs(path)
        if name is not None:
            torch.save(self.model.state_dict(),
                   os.path.join(path, "%s.pth" % name))
        else:
            torch.save(self.model.state_dict(),
                   os.path.join(path, "model.pth"))
    
    def save_config(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)
        print(self.args.__dict__)
        with open(os.path.join(path, 'args.txt'), 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)
                      
        with open(os.path.join(path, 'visual_config.json'), 'w') as f:
            json.dump(VisualConfig.__dict__, f, indent=2)
    
    def save_msa(self, path, name=None):
        self.save(path, name)
        self.save_config(path)
        
        

if __name__ == "__main__":
    
    msa = MSA(Args)
    msa.train()