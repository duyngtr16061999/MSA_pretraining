# coding=utf-8
# Copyright 2019 project LXRT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch LXRT model."""

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open


import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from transformers import BertModel, BertConfig
from file_utils import cached_path

import wandb

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
TF_WEIGHTS_NAME = 'model.ckpt'

def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except Importtokenization:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class VisualAudioConfig(object):
    def __init__(self,
                 v_layers=3,
                 a_layers=3,
                 cross_layers=3):
        self.v_layers = v_layers
        self.a_layers = a_layers
        self.cross_layers = cross_layers

        self.visual_feat_dim = 35
        self.audio_feat_dim = 74

        self.visual_hidden_dim = 288
        self.visual_num_attention_heads = 6
        self.visual_intermediate_size = 576

        self.audio_hidden_dim = 288
        self.audio_num_attention_heads = 6
        self.audio_intermediate_size = 576

        self.cross_hidden_dim = 288
        self.cross_num_attention_heads = 6
        self.cross_intermediate_size = 576

    def set_input_dim(self, visual_feat_dim, audio_feat_dim):
        self.visual_feat_dim = visual_feat_dim
        self.audio_feat_dim = audio_feat_dim

    def set_layers(self, v_layers, a_layers, cross_layers):
        self.v_layers = v_layers
        self.a_layers = a_layers
        self.cross_layers = cross_layers

    def set_visual_config(self, visual_hidden_dim, visual_num_attention_heads, visual_intermediate_size):
        self.visual_hidden_dim = visual_hidden_dim
        self.visual_num_attention_heads = visual_num_attention_heads
        self.visual_intermediate_size = visual_intermediate_size
    
    def set_audio_config(self, audio_hidden_dim, audio_num_attention_heads, audio_intermediate_size):
        self.audio_hidden_dim = audio_hidden_dim
        self.audio_num_attention_heads = audio_num_attention_heads
        self.audio_intermediate_size = audio_intermediate_size

    def set_cross_config(self, cross_hidden_dim, cross_num_attention_heads, cross_intermediate_size):
        self.cross_hidden_dim = cross_hidden_dim
        self.cross_num_attention_heads = cross_num_attention_heads
        self.cross_intermediate_size = cross_intermediate_size

VisualConfig = VisualAudioConfig()


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertBiAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.key1 = nn.Linear(ctx_dim, self.all_head_size)
        self.value1 = nn.Linear(ctx_dim, self.all_head_size)

        self.key2 = nn.Linear(ctx_dim, self.all_head_size)
        self.value2 = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context1, context2, attention_mask1=None, attention_mask2=None):
        mixed_query_layer = self.query(hidden_states)

        mixed_key1_layer = self.key1(context1)
        mixed_value1_layer = self.value1(context1)

        mixed_key2_layer = self.key2(context2)
        mixed_value2_layer = self.value2(context2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        
        key1_layer = self.transpose_for_scores(mixed_key1_layer)
        value1_layer = self.transpose_for_scores(mixed_value1_layer)
        key2_layer = self.transpose_for_scores(mixed_key2_layer)
        value2_layer = self.transpose_for_scores(mixed_value2_layer)

        #For context 1
        attention_scores1 = torch.matmul(query_layer, key1_layer.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask1 is not None:
            attention_scores1 = attention_scores1 + attention_mask1

        # Normalize the attention scores to probabilities.
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout(attention_probs1)
        
        context_layer1 = torch.matmul(attention_probs1, value1_layer)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)
        
        #For context 2
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores2 = torch.matmul(query_layer, key2_layer.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask2 is not None:
            attention_scores2 = attention_scores2 + attention_mask2

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value2_layer)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        context_layer_sum = (context_layer1 + context_layer2)/2
        return context_layer_sum

class BertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            #import pdb;pdb.set_trace()
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output

class BertBiCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertBiAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor1, ctx_tensor2, ctx_att_mask1=None, ctx_att_mask2=None):
        output = self.att(input_tensor, ctx_tensor1, ctx_tensor2, ctx_att_mask1, ctx_att_mask2)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, is_audio=False, is_visual=False):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


"""
---------------------------------------------------------------------------------------
      Above modules are copied from BERT (pytorch-transformer) with modifications.
---------------------------------------------------------------------------------------
"""
class MMBERTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # The cross-biattention layer
        self.language_attention = BertBiCrossattLayer(config)
        self.visual_attention = BertBiCrossattLayer(config)
        self.audio_attention = BertBiCrossattLayer(config)
        
        # Self-attention Layers
        self.lang_self_att = BertSelfattLayer(config)
        self.visn_self_att = BertSelfattLayer(config)
        self.audio_self_att = BertSelfattLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)
        self.audio_inter = BertIntermediate(config)
        self.audio_output = BertOutput(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask, audio_input, audio_attention_mask):
        # Cross Attention
        lang_att_output = self.language_attention(lang_input, visn_input, audio_input, ctx_att_mask1=visn_attention_mask, ctx_att_mask2=audio_attention_mask)
        visn_att_output = self.visual_attention(visn_input, audio_input, lang_input, ctx_att_mask1=audio_attention_mask, ctx_att_mask2=lang_attention_mask)
        audio_att_output = self.audio_attention(audio_input, lang_input, visn_input, ctx_att_mask1=lang_attention_mask, ctx_att_mask2=visn_attention_mask)
        return lang_att_output, visn_att_output, audio_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask, audio_input, audio_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        audio_att_output = self.audio_self_att(audio_input, audio_attention_mask)
        
        return lang_att_output, visn_att_output, audio_att_output

    def output_fc(self, lang_input, visn_input, audio_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)
        audio_att_output = self.audio_inter(audio_input)
        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        audio_output = self.audio_output(audio_att_output, audio_input)
        return lang_output, visn_output, audio_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask,
                      audio_feats, audio_attention_mask):

        lang_att_output, visn_att_output, audio_att_output = self.cross_att(
                                                        lang_feats, lang_attention_mask,
                                                        visn_feats, visn_attention_mask,
                                                        audio_feats, audio_attention_mask)

        lang_att_output, visn_att_output, audio_att_output = self.self_att(lang_att_output, lang_attention_mask,
                                                        visn_att_output, visn_attention_mask,
                                                        audio_att_output, audio_attention_mask)
        
        lang_att_output, visn_att_output, audio_att_output = self.output_fc(lang_att_output, visn_att_output, audio_att_output)

        return lang_att_output, visn_att_output, audio_att_output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class VisualEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = VisualConfig.visual_feat_dim

        # Object feature encoding
        self.visual_fc = nn.Linear(feat_dim, VisualConfig.visual_hidden_dim)
        
        self.pos_embedding = PositionalEncoding(VisualConfig.visual_hidden_dim)
        #self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        
        self.visual_layer_norm = BertLayerNorm(VisualConfig.visual_hidden_dim, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visual_input):
        x = self.visual_fc(visual_input)
        _, n, _ = visual_input.shape
        x = self.pos_embedding(x)
        x = self.visual_layer_norm(x)
        output = self.dropout(x)
        return output

class AudioEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        feat_dim = VisualConfig.audio_feat_dim

        # Object feature encoding
        self.audio_fc = nn.Linear(feat_dim, VisualConfig.audio_hidden_dim)
        
        #self.pos_embedding = nn.Parameter(torch.randn(1, VisualConfig.max_audio_len + 1, VisualConfig.audio_hidden_dim))
        self.pos_embedding = PositionalEncoding(VisualConfig.audio_hidden_dim)
        #self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        
        self.audio_layer_norm = BertLayerNorm(VisualConfig.audio_hidden_dim, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, audio_input):
        x = self.audio_fc(audio_input)
        
        _, n, _ = audio_input.shape
        x = self.pos_embedding(x)
        x = self.audio_layer_norm(x)

        output = self.dropout(x)
        return output

class MMBERTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Language encoder
        self.config = config
        self.visual_config = copy.deepcopy(self.config)
        self.audio_config = copy.deepcopy(self.config)
        self.cross_config = copy.deepcopy(self.config)

        self.visual_config.hidden_size = VisualConfig.visual_hidden_dim
        self.visual_config.num_attention_heads = VisualConfig.visual_num_attention_heads
        self.visual_config.intermediate_size = VisualConfig.visual_intermediate_size

        self.audio_config.hidden_size = VisualConfig.audio_hidden_dim
        self.audio_config.num_attention_heads = VisualConfig.audio_num_attention_heads
        self.audio_config.intermediate_size = VisualConfig.audio_intermediate_size

        self.cross_config.hidden_size = VisualConfig.cross_hidden_dim
        self.cross_config.num_attention_heads = VisualConfig.cross_num_attention_heads
        self.cross_config.intermediate_size = VisualConfig.cross_intermediate_size

        self.bertmodel = BertModel.from_pretrained('bert-base-uncased')
        self.language_project = nn.Linear(config.hidden_size, self.cross_config.hidden_size)
        # Visual encoder
        self.visn_fc = VisualEncoder(self.visual_config)
        # Audio encoder
        self.audio_fc = AudioEncoder(self.audio_config)
        
        # Number of layers
        self.num_v_layers = VisualConfig.v_layers
        self.num_a_layers = VisualConfig.a_layers
        self.num_cross_layers = VisualConfig.cross_layers
        
        print("MMBERT encoder with %d v_layers, %d a_layers, and %d cross_layers." %
              (self.num_v_layers, self.num_a_layers, self.num_cross_layers))

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.v_layers = nn.ModuleList(
            [BertLayer(self.visual_config) for _ in range(self.num_v_layers)]
        )
        self.a_layers = nn.ModuleList(
            [BertLayer(self.audio_config) for _ in range(self.num_a_layers)]
        )
        self.cross_layers = nn.ModuleList(
            [MMBERTLayer(self.cross_config) for _ in range(self.num_cross_layers)]
        )

        # self.cross_layers[0].language_attention.att.query = nn.Linear(config.hidden_size, self.cross_layers[0].language_attention.att.all_head_size)
        # self.cross_layers[0].visual_attention.att.key2 = nn.Linear(config.hidden_size, self.cross_layers[0].visual_attention.att.all_head_size)
        # self.cross_layers[0].visual_attention.att.value2 = nn.Linear(config.hidden_size, self.cross_layers[0].visual_attention.att.all_head_size)
        # self.cross_layers[0].audio_attention.att.key1 = nn.Linear(config.hidden_size, self.cross_layers[0].audio_attention.att.all_head_size)
        # self.cross_layers[0].audio_attention.att.value1 = nn.Linear(config.hidden_size, self.cross_layers[0].audio_attention.att.all_head_size)
        self.type = next(self.parameters()).dtype
    
    def forward(self, input_ids,attention_mask,token_type_ids,
                visn_feats, visn_attention_mask,
                audio_feats, audio_attention_mask):
        # Run visual embedding layer
        # Note: Word embedding layer was executed outside this module.
        #       Keep this design to allow loading BERT weights.
        visn_feats = self.visn_fc(visn_feats)
        audio_feats = self.audio_fc(audio_feats)

        # Run language layers
        bert_output = self.bertmodel(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

        #lang_feats = bert_output[0]
        for layer_module in self.v_layers:
            visn_feats = layer_module(visn_feats, visn_attention_mask)
            
        for layer_module in self.a_layers:
            audio_feats = layer_module(audio_feats, audio_attention_mask)

        # Run cross-modality layers
        lang_feats = bert_output["last_hidden_state"]
        lang_feats = self.language_project(lang_feats)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = extended_attention_mask.type(self.type)#.to(input_ids.device)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        for layer_module in self.cross_layers:
            lang_feats, visn_feats, audio_feats = layer_module(lang_feats, extended_attention_mask,
                                                  visn_feats, visn_attention_mask,
                                                  audio_feats, audio_attention_mask)

        return lang_feats, visn_feats, audio_feats
    
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()

        #For masked language modeling
        #self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        #self.transform = BertPredictionHeadTransform(config)
        self.dense = nn.Linear(config.hidden_size, bert_model_embedding_weights.size(1))
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(bert_model_embedding_weights.size(1), eps=1e-12)
        
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

        #For matching
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        #prediction_scores = self.predictions(sequence_output)
        #hidden_states = self.transform(hidden_states)
        if sequence_output is not None:
            hidden_states = self.dense(sequence_output)
            hidden_states = self.transform_act_fn(hidden_states)
            hidden_states = self.LayerNorm(hidden_states)

            prediction_scores = self.decoder(hidden_states) + self.bias
        else:
            prediction_scores = None

        if pooled_output is not None:
            seq_relationship_score = self.seq_relationship(pooled_output)
        else:
            seq_relationship_score = None

        return prediction_scores, seq_relationship_score


class BertVisualObjHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.decoder = nn.Linear(config.hidden_size, VisualConfig.visual_feat_dim)


    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output_visual_feats = self.decoder(hidden_states)
        return output_visual_feats

class BertAudioObjHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.decoder = nn.Linear(config.hidden_size, VisualConfig.audio_feat_dim)


    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output_audio_feats = self.decoder(hidden_states)
        return output_audio_feats

class MMBERTModel(nn.Module):
    """LXRT Model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = MMBERTEncoder(config)
        self.pooler = BertPooler(self.encoder.cross_config)
        
        self.type = next(self.parameters()).dtype
        self.apply(self.init_bert_weights)
    
    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                visn_feats=None, visn_attention_mask=None,
                audio_feats=None, audio_attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        #extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = extended_attention_mask.type(self.type)#.to(input_ids.device)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Process the visual attention mask
        if visn_attention_mask is not None:
            extended_visual_attention_mask = visn_attention_mask.unsqueeze(1).unsqueeze(2)
            #extended_visual_attention_mask = extended_visual_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_visual_attention_mask = extended_visual_attention_mask.type(self.type)#.to(input_ids.device)
            extended_visual_attention_mask = (1.0 - extended_visual_attention_mask) * -10000.0
        else:
            extended_visual_attention_mask = None
            
        if audio_attention_mask is not None:
            extended_audio_attention_mask = audio_attention_mask.unsqueeze(1).unsqueeze(2)
            #extended_audio_attention_mask = extended_audio_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_audio_attention_mask = extended_audio_attention_mask.type(self.type)#.to(input_ids.device)
            extended_audio_attention_mask = (1.0 - extended_audio_attention_mask) * -10000.0
        else:
            extended_audio_attention_mask = None

        # Run LXRT backbone
        lang_feats, visn_feats, audio_feats = self.encoder(
            input_ids,attention_mask,token_type_ids,
            visn_feats, extended_visual_attention_mask,
            audio_feats, extended_audio_attention_mask)

        pooled_output = self.pooler(lang_feats)

        return (lang_feats, visn_feats, audio_feats), pooled_output


class MMBERTPretraining(nn.Module):
    def __init__(self,
                 config,
                 task_mask_lm=True,
                 task_mask_v=True,
                 task_mask_a=True,
                 matching_task=True,
                 matching_visual=True,
                 matching_audio=True,
                 ):
        super().__init__()
        # Configuration
        self.config = config

        # Use of pre-training tasks
        self.task_mask_lm = task_mask_lm
        self.task_mask_v = task_mask_v
        self.task_mask_a = task_mask_a
        
        self.matching = matching_task
        self.matching_visual = matching_visual
        self.matching_audio = matching_audio

        # LXRT backbone
        self.bert = MMBERTModel(config)

        # Pre-training heads
        if self.task_mask_lm or self.matching or self.matching_visual or self.matching_audio:
            self.cls = BertPreTrainingHeads(self.bert.encoder.cross_config, self.bert.encoder.bertmodel.embeddings.word_embeddings.weight)
        if self.task_mask_v:
            self.visual_head = BertVisualObjHead(self.bert.encoder.cross_config)
        if self.task_mask_a:
            self.audio_head = BertAudioObjHead(self.bert.encoder.cross_config)

        # BaryCenter
        ### TODO ###

        # Weight initialization
        self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, batch):
        total_loss = 0.
        losses = ()
        
        if self.task_mask_lm:
            masked_input_ids = batch.batch_masked_input_ids
            attention_mask = batch.batch_attention_mask
            token_type_ids = batch.batch_token_type_ids
            masked_label = batch.batch_masked_label
            
            visual_feat = batch.batch_visual_feat
            visual_feat_am = batch.batch_visual_feat_am
            
            audio_feat = batch.batch_audio_feat
            audio_feat_am = batch.batch_audio_feat_am
            
            (lang_feats, _, _), _ = self.bert(
                masked_input_ids, token_type_ids, attention_mask,
                visual_feat, visual_feat_am,
                audio_feat, audio_feat_am,
            )
            lang_prediction_scores, _ = self.cls(lang_feats, None)
            
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            masked_lm_loss = loss_fct(
                lang_prediction_scores.view(-1, self.config.vocab_size),
                masked_label.view(-1)
            )
            
            total_loss += masked_lm_loss
            wandb.log({
                    "step_task_mask_lm_loss":masked_lm_loss
                })
            #losses["task_mask_lm"] = masked_lm_loss.detach()
            losses += (masked_lm_loss.detach(),)
            
        if self.task_mask_v:
            input_ids = batch.batch_input_ids
            attention_mask = batch.batch_attention_mask
            token_type_ids = batch.batch_token_type_ids
            
            visual_feat = batch.batch_masked_visual
            visual_feat_am = batch.batch_visual_feat_am
            label_visual_feat = batch.batch_masked_visual_label
            raw_visual_feat = batch.batch_visual_feat
            
            audio_feat = batch.batch_audio_feat
            audio_feat_am = batch.batch_audio_feat_am
            
            (_, visual_feat, _), _ = self.bert(
                input_ids, token_type_ids, attention_mask,
                visual_feat, visual_feat_am,
                audio_feat, audio_feat_am,
            )
           
            loss_fct = SmoothL1Loss(reduction='none')
            
            visn_prediction_scores = self.visual_head(visual_feat)
            output_dim = VisualConfig.visual_feat_dim
            
            visn_loss = loss_fct(
                visn_prediction_scores.view(-1, output_dim),
                raw_visual_feat.view(-1, output_dim)
            )
            if visn_loss.dim() > 1:     # Regression Losses
                visn_loss = visn_loss.mean(1)
            visn_loss = (visn_loss * label_visual_feat.view(-1)).mean()
            
            total_loss += visn_loss
            wandb.log({
                    "step_task_mask_v_loss":visn_loss
                })
            #losses["task_mask_v"] = visn_loss.detach()
            losses += (visn_loss.detach(),)
        
        if self.task_mask_a:
            input_ids = batch.batch_input_ids
            attention_mask = batch.batch_attention_mask
            token_type_ids = batch.batch_token_type_ids

            visual_feat = batch.batch_visual_feat
            visual_feat_am = batch.batch_visual_feat_am
            
            audio_feat = batch.batch_masked_audio
            audio_feat_am = batch.batch_audio_feat_am
            label_audio_feat = batch.batch_masked_audio_label
            raw_audio_feat = batch.batch_audio_feat
            
            (_, _, audio_feat), _ = self.bert(
                input_ids, token_type_ids, attention_mask,
                visual_feat, visual_feat_am,
                audio_feat, audio_feat_am,
            )
           
            loss_fct = SmoothL1Loss(reduction='none')
            
            aud_prediction_scores = self.audio_head(audio_feat)
            output_dim = VisualConfig.audio_feat_dim
            
            
            aud_loss = loss_fct(
                aud_prediction_scores.view(-1, output_dim),
                raw_audio_feat.view(-1, output_dim)
            )
            if aud_loss.dim() > 1:     # Regression Losses
                aud_loss = aud_loss.mean(1)
            aud_loss = (aud_loss * label_audio_feat.view(-1)).mean()
            
            total_loss += aud_loss
            wandb.log({
                    "step_task_mask_a_loss":aud_loss
                })
            #losses["task_mask_a"] = aud_loss.detach()
            losses += (aud_loss.detach(),)
        
        if self.matching:
            input_ids = batch.batch_input_ids
            attention_mask = batch.batch_attention_mask
            token_type_ids = batch.batch_token_type_ids

            visual_feat = batch.batch_visual_feat
            visual_feat_am = batch.batch_visual_feat_am
            
            audio_feat = batch.batch_audio_feat
            audio_feat_am = batch.batch_audio_feat_am
            
            (_, _, _), pooled_output = self.bert(
                input_ids, token_type_ids, attention_mask,
                visual_feat, visual_feat_am,
                audio_feat, audio_feat_am,
            )
            _, cross_relationship_score = self.cls(None, pooled_output)
            loss_fct = CrossEntropyLoss(ignore_index=-1)

            matched_label = torch.ones((cross_relationship_score.shape[0]), dtype=torch.int64).to(cross_relationship_score.device)
            
            matched_loss = loss_fct(
                cross_relationship_score.view(-1, 2),
                matched_label.view(-1)
            )
            total_loss += matched_loss
            wandb.log({
                    "step_matching_loss":matched_loss
                })
            #losses["matching"] = matched_loss.detach()
            losses += (matched_loss.detach(),)
        
        if self.matching_visual:
            input_ids = batch.batch_input_ids
            attention_mask = batch.batch_attention_mask
            token_type_ids = batch.batch_token_type_ids

            visual_feat = batch.batch_negative_visual
            visual_feat_am = batch.batch_negativae_visual_am
            
            audio_feat = batch.batch_audio_feat
            audio_feat_am = batch.batch_audio_feat_am
            
            (_, _, _), pooled_output = self.bert(
                input_ids, token_type_ids, attention_mask,
                visual_feat, visual_feat_am,
                audio_feat, audio_feat_am,
            )
            _, cross_relationship_score = self.cls(None, pooled_output)
            
            loss_fct = CrossEntropyLoss(ignore_index=-1)

            matched_label = torch.zeros((cross_relationship_score.shape[0]), dtype=torch.int64).to(cross_relationship_score.device)
            
            matched_loss = loss_fct(
                cross_relationship_score.view(-1, 2),
                matched_label.view(-1)
            )
            total_loss += matched_loss
            wandb.log({
                    "step_matching_visual_loss":matched_loss
                })
            #losses["matching_visual"] = matched_loss.detach()
            losses += (matched_loss.detach(),)
        
        if self.matching_audio:
            input_ids = batch.batch_input_ids
            attention_mask = batch.batch_attention_mask
            token_type_ids = batch.batch_token_type_ids

            visual_feat = batch.batch_visual_feat
            visual_feat_am = batch.batch_visual_feat_am
            
            audio_feat = batch.batch_negative_audio
            audio_feat_am = batch.batch_negativae_audio_am
            
            (_, _, _), pooled_output = self.bert(
                input_ids, token_type_ids, attention_mask,
                visual_feat, visual_feat_am,
                audio_feat, audio_feat_am,
            )
            _, cross_relationship_score = self.cls(None, pooled_output)
            
            loss_fct = CrossEntropyLoss(ignore_index=-1)

            matched_label = torch.zeros((cross_relationship_score.shape[0]), dtype=torch.int64).to(cross_relationship_score.device)
            
            matched_loss = loss_fct(
                cross_relationship_score.view(-1, 2),
                matched_label.view(-1)
            )
            total_loss += matched_loss
            wandb.log({
                    "step_matching_audio_loss":matched_loss
                })
            #losses["matching_audio"] = matched_loss.detach()
            losses += (matched_loss.detach(),)
        
        #return (lang_feats, visn_feats, audio_feats), (lang_prediction_scores, cross_relationship_score)
        wandb.log({
                    "step_loss":total_loss
                })
        return total_loss, torch.tensor(losses)
