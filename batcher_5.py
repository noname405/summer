# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
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
# ==============================================================================

"""This file contains code to process data into batches"""

import Queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import tensorflow as tf
#import data
import os

import numpy as np
import read_data_emb as read_data


class Example(object):
  """Class representing a train/val/test example for text summarization."""

  def __init__(self,start_enc,bucketing=True, truncate_input=True):
    """Initializes the Example, performing tokenization and truncation to produce the encoder, decoder and target sequences, which are stored in self.

    Args:
      article: source text; a string. each token is separated by a single space.
      abstract_sentences: list of strings, one per abstract sentence. In each sentence, each token is separated by a single space.
      vocab: Vocabulary object
      hps: hyperparameters
    """

 #   self.status=0
    self.start_enc=start_enc
    self.bucketing=bucketing
    self.truncate_input=truncate_input
#    self.hps = hps

  def prepare_data(self,article_sentences,sentence_label,words,word_label,abstract, vocab, hps):
    # Get ids of special tokens
    input_batch=[]
    file_no=0
    self.hps = hps
    self.vocab=vocab
    self.status=0
#    print("input",article_sentences,sentence_label,words,word_label,abstract)
    enc_inputs = [read_data.PAD_ID,read_data.GO_ID,read_data.EOS_ID,read_data.UNK_ID,read_data.NUL_ID]
    # Use the <s> as the <GO> symbol for decoder inputs.
    dec_labels=[0,0,0,0]
    dec_inputs = [read_data.GO_ID]
    sen_enc_inputs=[read_data.PAD_ID,read_data.GO_ID,read_data.EOS_ID,read_data.UNK_ID,read_data.NUL_ID]
    #start_enc=len(sen_enc_inputs)
    sen_dec_inputs=[read_data.GO_ID]
    target=[]
    switch=[]
    word_weight=[]
    article=[]
    sentence_ind=[read_data.PAD_ID,read_data.GO_ID,read_data.EOS_ID,read_data.UNK_ID,read_data.NUL_ID]
    sent_decwords=[]
    word_decode_count=[]
    #sent_enc_timesteps=max_article_sentences+start_enc
    #print('here')
    # Convert first N sentences to word IDs, stripping existing <s> and </s>.
    if len(article_sentences) < self.hps.min_input_len:
      #continue
      self.status=1
      return

    if not self.truncate_input:
      flag=0
      for i in xrange(self.hps.max_article_sentences):
        if (len(article_sentences[i]) > self.hps.enc_timesteps): # or
            #len(dec_inputs) > dec_timesteps):
          tf.logging.warning('Drop an example - too long.')
          flag=1
          break

      if flag==1:
        self.status=1
        return
        #continue
    # If we are truncating input, do so if necessary
    else:
      # for i in xrange(self.hps.max_article_sentences):
      #   if (len(article_sentences[i]) > self.hps.enc_timesteps):
      #       article_sentences[i] = article_sentences[i][:self.hps.enc_timesteps]
      for art_sentence in zip(article_sentences):
        article_sentence=art_sentence[0].split()
        if (len(article_sentence) > self.hps.enc_timesteps):
            article_sentence = article_sentence[:self.hps.enc_timesteps]
      # if len(dec_inputs) > dec_timesteps:
      #   dec_inputs = dec_inputs[:dec_timesteps]

    # for i in xrange(min(self.hps.max_article_sentences,
    #                     len(article_sentences))):
    i=0
    for art_sentence in zip(article_sentences):
    #for i in xrange(self.hps.max_article_sentences):
      article_sentence=art_sentence[0].split()
      if(i < self.hps.max_article_sentences):

        if(len(enc_inputs)<(self.hps.enc_timesteps-self.hps.min_input_len)):

          if(len(enc_inputs)+len(article_sentence) > (self.hps.enc_timesteps-self.hps.min_input_len)):
            article_sentence=article_sentence[:(self.hps.enc_timesteps-len(enc_inputs))]
            word_label[i]=word_label[i][:(self.hps.enc_timesteps-len(enc_inputs))]
          input_sent=read_data.sentence_to_token_ids(article_sentence, self.vocab)
          #print(article_sentences[i],input_sent)
          t=[len(enc_inputs)+k for k,j in enumerate(word_label[i])if j== 1]
          #print(t)
	  if len(t)+len(target) > self.hps.dec_timesteps-3:
		break
          if len(t)>0:

            switch.extend(np.zeros(len(t),dtype=np.int32))
            word_weight.extend(np.ones(len(t),dtype=np.int32))
	    sen_dec_inputs.extend(np.ones(len(t),dtype=np.int32)*read_data.NUL_ID)
            target+=t 
	    dec_inputs.extend(t)
          enc_inputs += input_sent
          sen_enc_inputs+=[len(enc_inputs)-1]
          dec_labels+=word_label[i]
	  #sentence_ind.extend(np.ones(len(input_sent),dtype=np.int32)*sen_enc_inputs[-1])
	  sentence_ind.extend(np.ones(len(input_sent),dtype=np.int32)*(i+1+read_data.NUL_ID))
          if sentence_label[i]=='1':
            switch+=[1]
            word_weight+=[0]
            #print(switch)
            sen_dec_inputs +=[i+self.start_enc]
            target+=[i+self.start_enc]
	    dec_inputs.extend([read_data.NUL_ID])
          article.append(article_sentence)
        i+=1

#    dec_inputs.extend([k for k,j in enumerate(dec_labels)if j==1] )
    switch+=[1] 
    switch+=[0] 

    # print(enc_inputs)
    # print(switch)
    # Filter out too-short input
    if (len(enc_inputs) < self.hps.min_input_len ):
        #len(dec_inputs) < min_input_len):
      tf.logging.warning('Drop an example - too short.\nenc:%d\ndec:%d',
                         len(enc_inputs), len(dec_inputs))
      #continue
      self.status=1
      return


    if (len(switch) > self.hps.dec_timesteps ):
        #len(dec_inputs) < min_input_len):
      tf.logging.warning('Drop an example - too long.\ndec:%d',
                         len(switch))
      #continue
      self.status=1
      return


    #If we're not truncating input, throw out too-long input
    if not self.truncate_input:
      if (len(enc_inputs) > self.hps.enc_timesteps or
          len(dec_inputs) > self.hps.dec_timesteps+1):
        tf.logging.warning('Drop an example - too long.\nenc:%d\ndec:%d',
                           len(enc_inputs), len(dec_inputs))
        #continue
        self.status=1
        return
    # If we are truncating input, do so if necessary
    else:
      if len(enc_inputs) > self.hps.enc_timesteps:
        enc_inputs = enc_inputs[:self.hps.enc_timesteps]
      if len(dec_inputs) > self.hps.dec_timesteps+1:
        dec_inputs = dec_inputs[:self.hps.dec_timesteps+1]
      if len(sen_enc_inputs) > self.hps.sent_enc_timesteps:
        sen_enc_inputs = sen_enc_inputs[:self.hps.sent_enc_timesteps]
      if len(sen_dec_inputs) > self.hps.dec_timesteps+1:
        sen_dec_inputs = sen_dec_inputs[:self.hps.dec_timesteps+1]

    # targets is dec_inputs without <s> at beginning, plus </s> at end
    # targets = dec_inputs[1:]
    # targets.append(end_id)

    # Now len(enc_inputs) should be <= enc_timesteps, and
    # len(targets) = len(dec_inputs) should be <= dec_timesteps

    enc_input_len = len(enc_inputs)
    dec_output_len = len(target)+1
    sent_enc_input_len=self.hps.sent_enc_timesteps

    

    # Pad if necessary
    while len(enc_inputs) < self.hps.enc_timesteps:
      enc_inputs.append(read_data.PAD_ID)
    while len(dec_inputs) < self.hps.dec_timesteps+1:
      dec_inputs.append(read_data.EOS_ID)
    while len(sen_dec_inputs) < self.hps.dec_timesteps+1:
      sen_dec_inputs.append(read_data.EOS_ID)
    while len(target) < self.hps.dec_timesteps:
      target.append(read_data.EOS_ID)
    while len(switch) < self.hps.dec_timesteps:
      switch.append(read_data.PAD_ID)
    while len(word_weight) < self.hps.dec_timesteps:
      word_weight.append(read_data.PAD_ID)
    while len(sen_enc_inputs) < self.hps.sent_enc_timesteps:
      sen_enc_inputs.append(read_data.PAD_ID)

   # words_decsent=[enc_inputs[sen_enc_inputs[s-1]+1:sen_enc_inputs[s]+1 ] if s>read_data.NUL_ID else [0] for s in (sen_dec_inputs) ]
    words_decsent=[range(sen_enc_inputs[s-1]+1,sen_enc_inputs[s]+1 ) if s>read_data.NUL_ID else [0] for s in (sen_dec_inputs) ]
    #sent_decwords=np.zeros([len(sen_dec_inputs)])

    for k,d in enumerate(dec_inputs): 
      if d>read_data.NUL_ID:
        sent_decwords.append([sentence_ind[i] for i,e in enumerate(enc_inputs) if e==enc_inputs[d]] )
        word_decode_count.append(enc_inputs.count(enc_inputs[d]))
      else:
        sent_decwords.append([0])
 	word_decode_count.append(1)

    self.enc_inputs=enc_inputs
    self.sen_enc_inputs=sen_enc_inputs
    self.dec_inputs=dec_inputs
    self.sen_dec_inputs=sen_dec_inputs
    self.target=target
    self.switch=switch
#    self.word_weight=word_weight
    self.word_weight=[ww / float(wc) for ww,wc in zip(word_weight,word_decode_count)]
    self.enc_input_len=enc_input_len
    self.sent_enc_input_len=sent_enc_input_len
    self.dec_output_len=dec_output_len
    self.article=article
    self.abstract=abstract
    self.words_decsent=words_decsent
    self.sent_decwords=sent_decwords
    #print("output",self.article,self.abstract)
    # start_decoding = vocab.word2id(data.START_DECODING)
    # stop_decoding = vocab.word2id(data.STOP_DECODING)

    # # Process the article
    # article_words = article.split()
    # if len(article_words) > hps.max_enc_steps:
    #   article_words = article_words[:hps.max_enc_steps]
    # self.enc_len = len(article_words) # store the length after truncation but before padding
    # self.enc_input = [vocab.word2id(w) for w in article_words] # list of word ids; OOVs are represented by the id for UNK token

    # # Process the abstract
    # abstract = ' '.join(abstract_sentences) # string
    # abstract_words = abstract.split() # list of strings
    # abs_ids = [vocab.word2id(w) for w in abstract_words] # list of word ids; OOVs are represented by the id for UNK token

    # # Get the decoder input sequence and target sequence
    # self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, hps.max_dec_steps, start_decoding, stop_decoding)
    # self.dec_len = len(self.dec_input)

    # # If using pointer-generator mode, we need to store some extra info
    # if hps.pointer_gen:
    #   # Store a version of the enc_input where in-article OOVs are represented by their temporary OOV id; also store the in-article OOVs words themselves
    #   self.enc_input_extend_vocab, self.article_oovs = data.article2ids(article_words, vocab)

    #   # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
    #   abs_ids_extend_vocab = data.abstract2ids(abstract_words, vocab, self.article_oovs)

    #   # Overwrite decoder target sequence so it uses the temp article OOV ids
    #   _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, hps.max_dec_steps, start_decoding, stop_decoding)

    # # Store the original strings
    # self.original_article = article
    # self.original_abstract = abstract
    # self.original_abstract_sents = abstract_sentences


  # def get_dec_inp_targ_seqs(self, sequence, max_len, start_id, stop_id):
  #   """Given the reference summary as a sequence of tokens, return the input sequence for the decoder, and the target sequence which we will use to calculate loss. The sequence will be truncated if it is longer than max_len. The input sequence must start with the start_id and the target sequence must end with the stop_id (but not if it's been truncated).

  #   Args:
  #     sequence: List of ids (integers)
  #     max_len: integer
  #     start_id: integer
  #     stop_id: integer

  #   Returns:
  #     inp: sequence length <=max_len starting with start_id
  #     target: sequence same length as input, ending with stop_id only if there was no truncation
  #   """
  #   inp = [start_id] + sequence[:]
  #   target = sequence[:]
  #   if len(inp) > max_len: # truncate
  #     inp = inp[:max_len]
  #     target = target[:max_len] # no end_token
  #   else: # no truncation
  #     target.append(stop_id) # end token
  #   assert len(inp) == len(target)
  #   return inp, target


  # def pad_decoder_inp_targ(self, max_len, pad_id):
  #   """Pad decoder input and target sequences with pad_id up to max_len."""
  #   while len(self.dec_input) < max_len:
  #     self.dec_input.append(pad_id)
  #   while len(self.target) < max_len:
  #     self.target.append(pad_id)


  # def pad_encoder_input(self, max_len, pad_id):
  #   """Pad the encoder input sequence with pad_id up to max_len."""
  #   while len(self.enc_input) < max_len:
  #     self.enc_input.append(pad_id)
  #   if self.hps.pointer_gen:
  #     while len(self.enc_input_extend_vocab) < max_len:
  #       self.enc_input_extend_vocab.append(pad_id)


class Batch(object):
  """Class representing a minibatch of train/val/test examples for text summarization."""

  def __init__(self, example_list, hps, vocab):
    """Turns the example_list into a Batch object.

    Args:
       example_list: List of Example objects
       hps: hyperparameters
       vocab: Vocabulary object
    """
    self.hps=hps

    # self.pad_id = vocab.word2id(data.PAD_TOKEN) # id of the PAD token used to pad sequences
    # self.init_encoder_seq(example_list, hps) # initialize the input to the encoder
    # self.init_decoder_seq(example_list, hps) # initialize the input and targets for the decoder
    # self.store_orig_strings(example_list) # store the original strings

    self.enc_batch = np.zeros(
        (self.hps.batch_size, self.hps.enc_timesteps), dtype=np.int32)
    self.enc_input_lens = np.zeros(
        (self.hps.batch_size), dtype=np.int32)
    self.dec_batch = np.zeros(
        (self.hps.batch_size, self.hps.dec_timesteps+1), dtype=np.int32)
    self.dec_output_lens = np.zeros(
        (self.hps.batch_size), dtype=np.int32)
    self.sent_enc_batch = np.zeros(
        (self.hps.batch_size, self.hps.sent_enc_timesteps), dtype=np.int32)
    self.sent_enc_input_lens = np.zeros(
        (self.hps.batch_size), dtype=np.int32)
    self.sent_dec_batch = np.zeros(
        (self.hps.batch_size, self.hps.dec_timesteps+1), dtype=np.int32)
    self.sent_dec_output_lens = np.zeros(
        (self.hps.batch_size), dtype=np.int32)
    self.target_batch = np.zeros(
        (self.hps.batch_size, self.hps.dec_timesteps), dtype=np.int32)
    self.extend_target_batch = np.zeros(
        (self.hps.batch_size, self.hps.dec_timesteps), dtype=np.int32)
    self.sent_target_batch = np.zeros(
        (self.hps.batch_size, self.hps.dec_timesteps), dtype=np.int32)
    self.switch_batch = np.zeros(
        (self.hps.batch_size, self.hps.dec_timesteps), dtype=np.float32)
    self.word_weights = np.zeros(
        (self.hps.batch_size, self.hps.dec_timesteps), dtype=np.float32)
    self.switch_weights = np.zeros(
        (self.hps.batch_size, self.hps.dec_timesteps), dtype=np.float32)
    self.sent_decwords=np.zeros(
        (self.hps.batch_size,self.hps.dec_timesteps,self.hps.max_article_sentences), dtype=np.int32)
    self.words_decsent=np.zeros(
        (self.hps.batch_size,self.hps.dec_timesteps,self.hps.enc_timesteps), dtype=np.int32)
    self.weights_sent_decwords=np.zeros(
        (self.hps.batch_size,self.hps.dec_timesteps,self.hps.max_article_sentences), dtype=np.float32)
    self.weights_words_decsent=np.zeros(
        (self.hps.batch_size,self.hps.dec_timesteps,self.hps.enc_timesteps), dtype=np.float32)
    self.origin_articles = ['None'] * self.hps.batch_size
    self.origin_abstracts = ['None'] * self.hps.batch_size

    #buckets = prepare_data()
    #print(np.shape(buckets))
    for i in xrange(len(example_list)):
      ex=example_list[i]

      self.origin_articles[i] = ex.article
      self.origin_abstracts[i]=ex.abstract
      self.sent_dec_batch[i,:] = ex.sen_dec_inputs
      self.sent_enc_batch[i,:] = ex.sen_enc_inputs
      self.enc_input_lens[i] = ex.enc_input_len
      self.sent_enc_input_lens[i] = ex.sent_enc_input_len
      self.dec_output_lens[i] = ex.dec_output_len
      self.enc_batch[i, :] = ex.enc_inputs[:]
      self.dec_batch[i, :] = ex.dec_inputs[:]
#      self.target_batch[i, :] = ex.target[:]
      #sent_target_batch[i, :] = target[:]
      self.switch_batch[i, :] = ex.switch[:]
      self.word_weights[i, :] = ex.word_weight[:]
      
      for j in xrange(self.hps.dec_timesteps):
        if j < self.hps.dec_timesteps-1 :
          self.sent_target_batch[i][j] = ex.sen_dec_inputs[j+1]
          self.target_batch[i][j] = ex.dec_inputs[j+1]
	  for k in xrange(len(ex.sent_decwords[j+1])):
            self.sent_decwords[i][j][k]=ex.sent_decwords[j+1][k]
	    if ex.sent_decwords[j+1][k] >0:
              self.weights_sent_decwords[i][j][k]=1
          for k in xrange(len(ex.words_decsent[j+1])):
            self.words_decsent[i][j][k]=ex.words_decsent[j+1][k]
	    if ex.words_decsent[j+1][k] >0:
              self.weights_words_decsent[i][j][k]=1
        if(ex.switch[j]>0):
#          self.sent_target_batch[i][j] = ex.target[j]
          self.extend_target_batch[i][j]=ex.target[j]+self.hps.enc_timesteps
        else:
	  self.extend_target_batch[i][j]=ex.target[j]
#	print('target',self.extend_target_batch[i],ex.target)
      for j in xrange((self.extend_target_batch[i].tolist().index(read_data.EOS_ID))+1):
         self.switch_weights[i][j] = 1
   #   print("ex_out",ex.sent_decwords,ex.words_decsent)


  # def init_encoder_seq(self, example_list, hps):
  #   """Initializes the following:
  #       self.enc_batch:
  #         numpy array of shape (batch_size, <=max_enc_steps) containing integer ids (all OOVs represented by UNK id), padded to length of longest sequence in the batch
  #       self.enc_lens:
  #         numpy array of shape (batch_size) containing integers. The (truncated) length of each encoder input sequence (pre-padding).

  #     If hps.pointer_gen, additionally initializes the following:
  #       self.max_art_oovs:
  #         maximum number of in-article OOVs in the batch
  #       self.art_oovs:
  #         list of list of in-article OOVs (strings), for each example in the batch
  #       self.enc_batch_extend_vocab:
  #         Same as self.enc_batch, but in-article OOVs are represented by their temporary article OOV number.
  #   """
  #   # Determine the maximum length of the encoder input sequence in this batch
  #   max_enc_seq_len = max([ex.enc_len for ex in example_list])

  #   # Pad the encoder input sequences up to the length of the longest sequence
  #   for ex in example_list:
  #     ex.pad_encoder_input(max_enc_seq_len, self.pad_id)

  #   # Initialize the numpy arrays
  #   # Note: our enc_batch can have different length (second dimension) for each batch because we use dynamic_rnn for the encoder.
  #   self.enc_batch = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
  #   self.enc_lens = np.zeros((hps.batch_size), dtype=np.int32)

  #   # Fill in the numpy arrays
  #   for i, ex in enumerate(example_list):
  #     self.enc_batch[i, :] = ex.enc_input[:]
  #     self.enc_lens[i] = ex.enc_len

  #   # For pointer-generator mode, need to store some extra info
  #   if hps.pointer_gen:
  #     # Determine the max number of in-article OOVs in this batch
  #     self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
  #     # Store the in-article OOVs themselves
  #     self.art_oovs = [ex.article_oovs for ex in example_list]
  #     # Store the version of the enc_batch that uses the article OOV ids
  #     self.enc_batch_extend_vocab = np.zeros((hps.batch_size, max_enc_seq_len), dtype=np.int32)
  #     for i, ex in enumerate(example_list):
  #       self.enc_batch_extend_vocab[i, :] = ex.enc_input_extend_vocab[:]

  # def init_decoder_seq(self, example_list, hps):
  #   """Initializes the following:
  #       self.dec_batch:
  #         numpy array of shape (batch_size, max_dec_steps), containing integer ids as input for the decoder, padded to max_dec_steps length.
  #       self.target_batch:
  #         numpy array of shape (batch_size, max_dec_steps), containing integer ids for the target sequence, padded to max_dec_steps length.
  #       self.padding_mask:
  #         numpy array of shape (batch_size, max_dec_steps), containing 1s and 0s. 1s correspond to real tokens in dec_batch and target_batch; 0s correspond to padding.
  #       """
  #   # Pad the inputs and targets
  #   for ex in example_list:
  #     ex.pad_decoder_inp_targ(hps.max_dec_steps, self.pad_id)

  #   # Initialize the numpy arrays.
  #   # Note: our decoder inputs and targets must be the same length for each batch (second dimension = max_dec_steps) because we do not use a dynamic_rnn for decoding. However I believe this is possible, or will soon be possible, with Tensorflow 1.0, in which case it may be best to upgrade to that.
  #   self.dec_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
  #   self.target_batch = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.int32)
  #   self.padding_mask = np.zeros((hps.batch_size, hps.max_dec_steps), dtype=np.float32)

  #   # Fill in the numpy arrays
  #   for i, ex in enumerate(example_list):
  #     self.dec_batch[i, :] = ex.dec_input[:]
  #     self.target_batch[i, :] = ex.target[:]
  #     for j in xrange(ex.dec_len):
  #       self.padding_mask[i][j] = 1

  # def store_orig_strings(self, example_list):
  #   """Store the original article and abstract strings in the Batch object"""
  #   self.original_articles = [ex.original_article for ex in example_list] # list of lists
  #   self.original_abstracts = [ex.original_abstract for ex in example_list] # list of lists
  #   self.original_abstracts_sents = [ex.original_abstract_sents for ex in example_list] # list of list of lists


class Batcher(object):
  """A class to generate minibatches of data. Buckets examples together based on length of the encoder sequence."""

  BATCH_QUEUE_MAX = 100 # max number of batches the batch_queue can hold

  def __init__(self, data_path, vocab, hps, start_enc,single_pass,bucketing=True, truncate_input=True):
    """Initialize the batcher. Start threads that process the data into batches.

    Args:
      data_path: tf.Example filepattern.
      vocab: Vocabulary object
      hps: hyperparameters
      single_pass: If True, run through the dataset exactly once (useful for when you want to run evaluation on the dev or test set). Otherwise generate random batches indefinitely (useful for training).
    """
    self._data_path = data_path
    self._vocab = vocab
    self._hps = hps
    self._single_pass = single_pass
    self.start_enc=start_enc
    self.bucketing=bucketing
    self.truncate_input=truncate_input
    # Initialize a queue of Batches waiting to be used, and a queue of Examples waiting to be batched
    self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
    self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)

    # Different settings depending on whether we're in single_pass mode or not
    if single_pass:
      self._num_example_q_threads = 1 # just one thread, so we read through the dataset just once
      self._num_batch_q_threads = 1  # just one thread to batch examples
      self._bucketing_cache_size = 1 # only load one batch's worth of examples before bucketing; this essentially means no bucketing
      self._finished_reading = False # this will tell us when we're finished reading the dataset
    else:
      self._num_example_q_threads = 16 # num threads to fill example queue
      self._num_batch_q_threads = 4  # num threads to fill batch queue
      self._bucketing_cache_size = 100 # how many batches-worth of examples to load into cache before bucketing

    # Start the threads that load the queues
    self._example_q_threads = []
    for _ in xrange(self._num_example_q_threads):
      self._example_q_threads.append(Thread(target=self.fill_example_queue))
      self._example_q_threads[-1].daemon = True
      self._example_q_threads[-1].start()
    self._batch_q_threads = []
    for _ in xrange(self._num_batch_q_threads):
      self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
      self._batch_q_threads[-1].daemon = True
      self._batch_q_threads[-1].start()

    # Start a thread that watches the other threads and restarts them if they're dead
    if not single_pass: # We don't want a watcher in single_pass mode because the threads shouldn't run forever
      self._watch_thread = Thread(target=self.watch_threads)
      self._watch_thread.daemon = True
      self._watch_thread.start()


  def next_batch(self):
    """Return a Batch from the batch queue.

    If mode='decode' then each batch contains a single example repeated beam_size-many times; this is necessary for beam search.

    Returns:
      batch: a Batch object, or None if we're in single_pass mode and we've exhausted the dataset.
    """
    # If the batch queue is empty, print a warning
    if self._batch_queue.qsize() == 0:
      tf.logging.warning('Bucket input queue is empty when calling next_batch. Bucket queue size: %i, Input queue size: %i', self._batch_queue.qsize(), self._example_queue.qsize())
      if self._single_pass and self._finished_reading:
        tf.logging.info("Finished reading dataset in single_pass mode.")
        return None

    batch = self._batch_queue.get() # get the next Batch
    return batch

  def fill_example_queue(self):
    """Reads data from file and processes into Examples which are then placed into the example queue."""

    input_gen = self.text_generator(read_data.example_generator(self._data_path, self._single_pass))

    while True:
      try:
        (article_text,sent_label_text,word_text,word_label_text,abstract) = input_gen.next() # read the next example from file. article and abstract are both strings.
      except StopIteration: # if there are no more examples:
        tf.logging.info("The example generator for this example queue filling thread has exhausted data.")
        if self._single_pass:
          tf.logging.info("single_pass mode is on, so we've finished reading dataset. This thread is stopping.")
          self._finished_reading = True
          break
        else:
          raise Exception("single_pass mode is off but the example generator is out of data; error.")

      #abstract_sentences = [sent.strip() for sent in data.abstract2sents(abstract)] # Use the <s> and </s> tags in abstract to get a list of sentences.
      article_sentences = [sent.strip() for sent in read_data.article2sents(article_text)] # Use the <s> and </s> tags in abstract to get a list of sentences.
    #  print('before',abstract)
    #  if self._single_pass:
      abstract=abstract.split('<eos>')
      word_label = read_data.text2labels(word_label_text)# Use the
      #print(article, abstract_sentences)
      #sent_label=[int(i) for i in sent_label_text.split()]
      sent_label=['1' if int(i)==1 else '0' for i in sent_label_text.split()]
      word=word_text.split()

#      example = Example(article, abstract_sentences, self._vocab, self._hps) # Process into an Example.
      example = Example(self.start_enc,self.bucketing, self.truncate_input)
      example.prepare_data(article_sentences,sent_label,word,word_label,abstract, self._vocab, self._hps)
      #print('status',example.status)
      if example.status==0:
        self._example_queue.put(example)
      #self._example_queue.put(example) # place the Example in the example queue.


  def fill_batch_queue(self):
    """Takes Examples out of example queue, sorts them by encoder sequence length, processes into Batches and places them in the batch queue.

    In decode mode, makes batches that each contain a single example repeated.
    """
    while True:
      if self._hps.mode != 'test':
        # Get bucketing_cache_size-many batches of Examples into a list, then sort
        inputs = []
        for _ in xrange(self._hps.batch_size * self._bucketing_cache_size):
          inputs.append(self._example_queue.get())
       # inputs = sorted(inputs, key=lambda inp: inp.enc_len) # sort by length of encoder sequence

        # Group the sorted Examples into batches, optionally shuffle the batches, and place in the batch queue.
        batches = []
        for i in xrange(0, len(inputs), self._hps.batch_size):
          batches.append(inputs[i:i + self._hps.batch_size])
        if not self._single_pass:
          shuffle(batches)
        for b in batches:  # each b is a list of Example objects
          self._batch_queue.put(Batch(b, self._hps, self._vocab))

      else: # beam search decode mode
        ex = self._example_queue.get()
        b = [ex for _ in xrange(self._hps.batch_size)]
        self._batch_queue.put(Batch(b, self._hps, self._vocab))


  def watch_threads(self):
    """Watch example queue and batch queue threads and restart if dead."""
    while True:
      time.sleep(60)
      for idx,t in enumerate(self._example_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found example queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_example_queue)
          self._example_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()
      for idx,t in enumerate(self._batch_q_threads):
        if not t.is_alive(): # if the thread is dead
          tf.logging.error('Found batch queue thread dead. Restarting.')
          new_t = Thread(target=self.fill_batch_queue)
          self._batch_q_threads[idx] = new_t
          new_t.daemon = True
          new_t.start()


  def text_generator(self, example_generator):
    """Generates article and abstract text from tf.Example.

    Args:
      example_generator: a generator of tf.Examples from file. See data.example_generator"""
    while True:
      e = example_generator.next() # e is a tf.Example
      try:
        article_text = e.features.feature['article_text'].bytes_list.value[0] # the article text was saved under the key 'article' in the data files
        sent_label_text = e.features.feature['sent_label_text'].bytes_list.value[0] # the abstract text was saved under the key 'abstract' in the data files
        word_text = e.features.feature['word_text'].bytes_list.value[0] 
        word_label_text=e.features.feature['word_label_text'].bytes_list.value[0] 
        abstract_text = e.features.feature['abstract_text'].bytes_list.value[0] 
      except ValueError:
        tf.logging.error('Failed to get article or abstract from example: %s', text_format.MessageToString(e))
        continue
      if len(article_text)==0: # See https://github.com/abisee/pointer-generator/issues/1
        tf.logging.warning('Found an example with empty article text. Skipping it.')
      else:
        yield (article_text,sent_label_text,word_text,word_label_text,abstract_text)
