# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Sequence-to-Sequence with attention model for text summarization.
"""
from collections import namedtuple

import numpy as np
import tensorflow as tf
#import seq2seq_lib
import seq2seq_17 as seq2seq
#from wlstm_cell_1 import wlstm_cell


HParams = namedtuple('HParams',
                     'mode, min_lr, lr, batch_size, '
                     'enc_layers, enc_timesteps, dec_timesteps,sent_enc_timesteps,max_article_sentences '
                     'min_input_len, num_hidden, emb_dim,vocabulary_size, max_grad_norm, '
                     'num_softmax_samples')


# def _extract_argmax_and_embed(embedding, output_projection=None,
#                               update_embedding=True):
#   """Get a loop_function that extracts the previous symbol and embeds it.

#   Args:
#     embedding: embedding tensor for symbols.
#     output_projection: None or a pair (W, B). If provided, each fed previous
#       output will first be multiplied by W and added B.
#     update_embedding: Boolean; if False, the gradients will not propagate
#       through the embeddings.

#   Returns:
#     A loop function.
#   """
#   def loop_function(prev, _):
#     """function that feed previous model output rather than ground truth."""
#     if output_projection is not None:
#       prev = tf.nn.xw_plus_b(
#           prev, output_projection[0], output_projection[1])
#     prev_symbol = tf.argmax(prev, 1)
#     # Note that gradients will not propagate through the second parameter of
#     # embedding_lookup.
#     emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
#     if not update_embedding:
#       emb_prev = tf.stop_gradient(emb_prev)
#     return emb_prev
#   return loop_function


class Seq2SeqAttentionModel(object):
  """Wrapper for Tensorflow model graph for text sum vectors."""

  def __init__(self, hps, vocab, num_gpus=0):
    self.hps = hps
    self.vocab = vocab
    self.num_gpus = num_gpus
    self.cur_gpu = 0
    self.loss_ind=3#tf.get_variable("loss_ind",initializer=2,trainable=False)

  def make_feed(self,batch):
    return {self.enc_batch:batch.enc_batch, self.dec_batch:batch.dec_batch,
                                       self.sent_enc_batch:batch.sent_enc_batch,self.sent_dec_batch:batch.sent_dec_batch,
                                       self.target_batch:batch.target_batch,self.extend_target_batch:batch.extend_target_batch,self.sent_target_batch:batch.sent_target_batch,self.switch_batch:batch.switch_batch,
                                       self.enc_input_lens:batch.enc_input_lens,self.sent_enc_input_lens:batch.sent_enc_input_lens,self.dec_output_lens:batch.dec_output_lens, 
                                       self.word_weights_batch:batch.word_weights,self.switch_weights_batch:batch.switch_weights,self.sent_decwords_batch:batch.sent_decwords,self.words_decsent_batch:batch.words_decsent,
                                       self.weights_sent_decwords_batch:batch.weights_sent_decwords,self.weights_words_decsent_batch:batch.weights_words_decsent}


  def run_train_step(self, sess, batch,loss_ind=0):
    self.loss_ind=loss_ind
    to_return =[self.train_op, self.summaries, self.loss_to_minimize, self.global_step]
    out=[self.decoder_outputs_dists, self.sent_decoder_outputs_dists,self.final_log_dists]
    loss=[self.loss,self.word_loss,self.sent_loss,self.word_loss_null,self.sent_loss_null,self.switch_loss]
    return sess.run([to_return,out,loss],
                    feed_dict=self.make_feed(batch))

  def run_eval_step(self, sess,batch): 
    to_return = [self.summaries, self.total_loss, self.global_step]
   # out=[self.decoder_outputs,self.sent_decoder_outputs,self.switch_pred,self.word_switch_pred]
#    out=[self.decoder_outputs_dists, self.sent_decoder_outputs_dists,self.switch_pred,self.word_switch_pred]
#    out=[self.decoder_outputs_dists, self.sent_decoder_outputs_dists,self.final_log_dists]
    out=[self.decoder_outputs, self.sent_decoder_outputs,self.final_log_dists]
    loss=[self.loss,self.word_loss,self.sent_loss,self.word_loss_null,self.sent_loss_null,self.switch_loss]
    return sess.run([to_return,out,loss],feed_dict=self.make_feed(batch))

  def run_test_step(self, sess, next_enc_batch, next_dec_batch,next_sent_enc_batch, next_sent_dec_batch,next_target_batch,next_sent_target_batch,next_switch_batch, next_enc_input_lens, next_sent_enc_input_lens,next_dec_output_lens,
          next_word_weights_batch,next_switch_weights_batch):
    to_return = [self.summaries, self.loss, self.global_step]
    out=[self.decoder_outputs,self.sent_decoder_outputs,self.switch_pred,self.word_switch_pred]
    data=[self.switch_prob,self.switch_pred,self.sent_weights,self.weights]
    return sess.run([to_return,out,data],
                    feed_dict={self.enc_batch:next_enc_batch, self.dec_batch:next_dec_batch,
                                       self.sent_enc_batch:next_sent_enc_batch,self.sent_dec_batch:next_sent_dec_batch,
                                       self.target_batch:next_target_batch,self.sent_target_batch:next_sent_target_batch,self.switch_batch:next_switch_batch,
                                       self.enc_input_lens:next_enc_input_lens,self.sent_enc_input_lens:next_sent_enc_input_lens,self.dec_output_lens:next_dec_output_lens, 
                                       self.word_weights_batch:next_word_weights_batch,self.switch_weights_batch:next_switch_weights_batch})

  def run_decode_step(self, sess, next_enc_batch, next_dec_batch,next_sent_enc_batch, next_sent_dec_batch,next_target_batch,next_sent_target_batch,next_switch_batch, next_enc_input_lens, next_sent_enc_input_lens,next_dec_output_lens,
          next_word_weights_batch,next_switch_weights_batch):
    to_return = [self.decoder_outputs,self.sent_decoder_outputs,self.switch_pred,self.word_switch_pred, self.global_step]
    return sess.run(to_return,
                    feed_dict={self.enc_batch:next_enc_batch, self.dec_batch:next_dec_batch,
                                       self.sent_enc_batch:next_sent_enc_batch,self.sent_dec_batch:next_sent_dec_batch,
                                       self.target_batch:next_target_batch,self.sent_target_batch:next_sent_target_batch,self.switch_batch:next_switch_batch,
                                       self.enc_input_lens:next_enc_input_lens,self.sent_enc_input_lens:next_sent_enc_input_lens,self.dec_output_lens:next_dec_output_lens, 
                                       self.word_weights_batch:next_word_weights_batch,self.switch_weights_batch:next_switch_weights_batch})


  def next_device(self):
    """Round robin the gpu device. (Reserve last gpu for expensive op)."""
    if self.num_gpus == 0:
      return ''
    dev = '/gpu:%d' % self.cur_gpu
    if self.num_gpus > 1:
      self.cur_gpu = (self.cur_gpu + 1) % (self.num_gpus-1)
    return dev

  def get_gpu(self, gpu_id):
    if self.num_gpus <= 0 or gpu_id >= self.num_gpus:
      return ''
    return '/gpu:%d' % gpu_id

  def mask_and_avg(self,values,weights):
    """Applies mask to values then returns overall average (a scalar)
    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and 0s.
    Returns:
      a scalar"""

   # dec_lens = tf.reduce_sum(self.switch_weights_batch, axis=1) # shape batch_size. float32
    dec_lens = tf.reduce_sum(weights, axis=1) # shape batch_size. float32
    dec_lens+= 1e-12
    values_per_step = [v * weights[:,dec_step] for dec_step,v in enumerate(values)]
    values_per_ex = sum(values_per_step)/dec_lens # shape (batch_size); normalized value for each batch member
    return tf.reduce_mean(values_per_ex) # overall average

  def _coverage_loss(self,attn_dists, padding_mask):
    """Calculates the coverage loss from the attention distributions.
    Args:
      attn_dists: The attention distributions for each decoder timestep. A list length max_dec_steps containing shape (batch_size, attn_length)
      padding_mask: shape (batch_size, max_dec_steps).
    Returns:
      coverage_loss: scalar
    """
    coverage = tf.zeros_like(attn_dists[0]) # shape (batch_size, attn_length). Initial coverage is zero.
    covlosses = [] # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
    for a in attn_dists:
      covloss = tf.reduce_sum(tf.minimum(a, coverage), [1]) # calculate the coverage loss for this step
      covlosses.append(covloss)
      coverage += a # update the coverage vector
    coverage_loss = self.mask_and_avg(covlosses, padding_mask)
    return coverage_loss


  def _calc_final_dist(self, vocab_dists, attn_dists):
    """Calculate the final distribution, for the pointer-generator model
    Args:
      vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays. The words are in the order they appear in the vocabulary file.
      attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
    Returns:
      final_dists: The final distributions. List length max-dec_steps of (batch_size, extended_vsize) arrays."""
    with tf.variable_scope('final_distribution'):
      # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
      # vocab_dists = [(1-p_gen) * dist for (p_gen,dist) in zip(p_gens, vocab_dists)]
      # attn_dists = [p_gen * dist for (p_gen,dist) in zip(p_gens, attn_dists)]

      # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
      extras=self.hps.sent_enc_timesteps-self.hps.max_article_sentences 
      extended_vsize = self.hps.enc_timesteps + self.hps.sent_enc_timesteps # the maximum (over the batch) size of the extended vocabulary
      extra_zeros = tf.zeros((self.hps.batch_size, self.hps.sent_enc_timesteps))
      vocab_dists_extended = [tf.concat(1,[dist, extra_zeros]) for dist in vocab_dists] # list length max_dec_steps of shape (batch_size, extended_vsize)

      # Project the values in the attention distributions onto the appropriate entries in the final distributions
      # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary, then we add 0.1 onto the 500th entry of the final distribution
      # This is done for each decoder timestep.
      # This is fiddly; we use tf.scatter_nd to do the projection
      batch_nums = tf.range(0, limit=self.hps.batch_size) # shape (batch_size)
      batch_nums = tf.expand_dims(batch_nums, 1) # shape (batch_size, 1)
      attn_len = self.hps.sent_enc_timesteps #tf.shape(self._enc_batch_extend_vocab)[1] # number of states we attend over
      batch_nums = tf.tile(batch_nums, [1, attn_len]) # shape (batch_size, attn_len)
      sent_ind=tf.range(self.hps.enc_timesteps, extended_vsize)
      sent_ind=tf.tile(tf.expand_dims(sent_ind,1),[self.hps.batch_size,1])
      indices = tf.stack( (batch_nums,tf.reshape(sent_ind,[self.hps.batch_size,self.hps.sent_enc_timesteps]) ), axis=2)
#      indices = tf.stack( (batch_nums,self.sent_enc_batch ), axis=2) # shape (batch_size, enc_t, 2)
      shape = [self.hps.batch_size, extended_vsize]
      attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists] # list length max_dec_steps (batch_size, extended_vsize)

      # Add the vocab distributions and the copy distributions together to get the final distributions
      # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving the final distribution for that decoder timestep
      # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
      final_dists = [vocab_dist + copy_dist for (vocab_dist,copy_dist) in zip(vocab_dists_extended, attn_dists_projected)]

      return final_dists  


  def get_loss(self,final_dists,targets,weights):
	  log_dists = [tf.log(dist+1e-12) for dist in final_dists]
          loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
          batch_nums = tf.range(0, limit=self.hps.batch_size) # shape (batch_size)
          for dec_step, log_dist in enumerate(log_dists):
            target = targets[dec_step] # The indices of the target words. shape (batch_size)
            indices = tf.stack( (batch_nums, target), axis=1) # shape (batch_size, 2)
            losses = tf.gather_nd(-log_dist, indices) # shape (batch_size). loss on this step for each batch
            loss_per_step.append(losses)
	  loss=self.mask_and_avg(loss_per_step,weights)
	  return loss


  def get_null_loss(self,final_dists,targets,nweights,weights):
        log_dists = [tf.log(dist+1e-12) for dist in final_dists]
        loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
        multiples=list(map(int, targets[0].get_shape()))[1]
       # print("multiples",multiples)
        batch_range = tf.range(0, limit=self.hps.batch_size) # shape (batch_size)
        batch_range=tf.reshape(batch_range,[-1,1])
        batch_nums=tf.tile(batch_range,[1,multiples])
       # print("range",batch_range)
       # print("nums",batch_nums)
        for dec_step, log_dist in enumerate(log_dists):
          target = targets[dec_step] # The indices of the target words. shape (batch_size)
         # print("target",target)
          indices = tf.stack( (batch_nums, target), axis=2) # shape (batch_size, 2)
         # print("ind",indices)
          losses = tf.gather_nd(-log_dist, indices)*nweights[dec_step] # shape (batch_size). loss on this step for each batch
	  dec_lens = tf.reduce_sum(nweights[dec_step],1)
          dec_lens+= 1e-12
         # print("losses",losses)
	  loss_per_step.append((tf.reduce_sum(losses,1))/dec_lens)
        loss=self.mask_and_avg(loss_per_step,weights)
        return loss



  def add_placeholders(self):
    """Inputs to be fed to the graph."""
    hps = self.hps

    self.enc_batch = tf.placeholder(tf.int32,
                                  [hps.batch_size, hps.enc_timesteps],
                                  name='enc_batch')
    self.dec_batch = tf.placeholder(tf.int32,
                                     [hps.batch_size, hps.dec_timesteps+1],
                                     name='dec_batch')
    self.sent_enc_batch = tf.placeholder(tf.int32,
                                      [hps.batch_size, hps.sent_enc_timesteps],
                                      name='sent_enc_batch')
    self.sent_dec_batch = tf.placeholder(tf.int32,
                                     [hps.batch_size, hps.dec_timesteps+1],
                                     name='sent_dec_batch')
    self.target_batch = tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps],
                                   name='target_batch')
    self.extend_target_batch = tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps],
                                   name='extend_target_batch')
    self.sent_target_batch = tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps],
                                   name='sent_target_batch')
    self.switch_batch = tf.placeholder(tf.float32,
                                   [hps.batch_size, hps.dec_timesteps],
                                   name='switch_batch')
    self.sent_decwords_batch=tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps,hps.max_article_sentences],name='sent_decwords')
    self.words_decsent_batch=tf.placeholder(tf.int32,
                                   [hps.batch_size, hps.dec_timesteps,hps.enc_timesteps],name='words_decsent')
    self.weights_sent_decwords_batch=tf.placeholder(tf.float32,
                                   [hps.batch_size, hps.dec_timesteps,hps.max_article_sentences],name='weights_sent_decwords')
    self.weights_words_decsent_batch=tf.placeholder(tf.float32,
                                   [hps.batch_size, hps.dec_timesteps,hps.enc_timesteps],name='weights_words_decsent')

    self.enc_input_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                        name='enc_input_lens')
    self.sent_enc_input_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                        name='sent_enc_input_lens')
    self.dec_output_lens = tf.placeholder(tf.int32, [hps.batch_size],
                                         name='dec_output_lens')
    self.word_weights_batch = tf.placeholder(tf.float32,
                                        [hps.batch_size, hps.dec_timesteps],
                                        name='word_weights_batch')
    self.switch_weights_batch = tf.placeholder(tf.float32,
                                        [hps.batch_size, hps.dec_timesteps],
                                          name='switch_weights_batch')
    #self.mode=tf.placeholder(tf.int32,name='mode')


  def add_seq2seq(self):
    hps = self.hps
    vsize = hps.vocabulary_size
    threshold=0.5

    with tf.variable_scope('seq2seq'):
      encoder_inputs = tf.unpack(tf.transpose(self.enc_batch))
      decoder_inputs = tf.unpack(tf.transpose(self.dec_batch))
      sent_encoder_inputs = tf.unpack(tf.transpose(self.sent_enc_batch))
      sent_decoder_inputs = tf.unpack(tf.transpose(self.sent_dec_batch))
      targets = tf.unpack(tf.transpose(self.target_batch))
      extend_targets = tf.unpack(tf.transpose(self.extend_target_batch))
      sent_targets = tf.unpack(tf.transpose(self.sent_target_batch))
      switch = tf.unpack(tf.transpose(self.switch_batch))
      word_weights = tf.unpack(tf.transpose(self.word_weights_batch))
      switch_weights = tf.unpack(tf.transpose(self.switch_weights_batch))
      sent_decwords=tf.unpack(tf.transpose(self.sent_decwords_batch,perm=[1,0,2]))
      words_decsent=tf.unpack(tf.transpose(self.words_decsent_batch,perm=[1,0,2]))
      weights_sent_decwords=tf.unpack(tf.transpose(self.weights_sent_decwords_batch,perm=[1,0,2]))
      weights_words_decsent=tf.unpack(tf.transpose(self.weights_words_decsent_batch,perm=[1,0,2]))
      enc_lens = self.enc_input_lens
      sent_enc_lens = self.sent_enc_input_lens
      #int_mode=self.mode
      # Embedding shared by the input and outputs.
      with tf.variable_scope('embedding'): #, tf.device('/cpu:0'):
        embedding = tf.get_variable(
            'word_embedding',dtype=tf.float32,
            initializer=self.embed)
        emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                              for x in encoder_inputs]
        emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, x)
                              for x in decoder_inputs]


      with tf.variable_scope('sent_embedding'):#, tf.device('/cpu:0'):
        sent_embedding = tf.get_variable(
            'sent_embedding', [hps.sent_enc_timesteps, hps.emb_dim], dtype=tf.float32)
            #initializer=tf.truncated_normal_initializer(stddev=1e-4))
        sent_emb_decoder_inputs = [tf.nn.embedding_lookup(sent_embedding, x)
                              for x in sent_decoder_inputs]

      for layer_i in xrange(hps.enc_layers):
        with tf.variable_scope('encoder%d'%layer_i):
	  emb_encoder_inputs=tf.unpack(tf.nn.dropout(emb_encoder_inputs,0.5))
          cell_fw = tf.nn.rnn_cell.LSTMCell(
              hps.num_hidden/2,
              initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=123),#tf.random_uniform_initializer(-0.1, 0.1, seed=123),
              state_is_tuple=False)
          cell_bw = tf.nn.rnn_cell.LSTMCell(
              hps.num_hidden/2,
              initializer=tf.contrib.layers.xavier_initializer(uniform=True,seed=123),#tf.random_uniform_initializer(-0.1, 0.1, seed=113),
              state_is_tuple=False)
          (emb_encoder_inputs, fw_state, bw_state) = tf.nn.bidirectional_rnn(
              cell_fw, cell_bw, emb_encoder_inputs, dtype=tf.float32,
              sequence_length=enc_lens)


          # cell_word = tf.nn.rnn_cell.LSTMCell(
          #     num_hidden,
          #     initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
          #     state_is_tuple=False)
          
          # (emb_encoder_inputs, fw_state) = tf.nn.rnn(
          #     cell_word, emb_encoder_inputs, dtype=tf.float32,
          #     sequence_length=enc_input_lens)
      encoder_outputs = emb_encoder_inputs

      #sent_ip=tf.pack([tf.gather(sent_i[l],index[l]) for l in xrange(batch_size)])
      sent_i=tf.transpose(encoder_outputs,perm=[1,0,2])
      #print(sent_i)
      index=tf.transpose(sent_encoder_inputs,perm=[1,0])
      #print(index)
      sent_ip=tf.pack([tf.gather(sent_i[l],index[l]) for l in xrange(hps.batch_size)])
      sent_input=tf.unpack(tf.transpose(sent_ip,perm=[1,0,2]))





      for layer_i in xrange(hps.enc_layers):
        with tf.variable_scope('sent_encoder%d'%layer_i):
	  sent_input=tf.unpack(tf.nn.dropout(sent_input,0.5))
          cell_sent = tf.nn.rnn_cell.LSTMCell(
              hps.num_hidden,
              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
              state_is_tuple=False)
          
          (sent_input, sent_fw_state) = tf.nn.rnn(
              cell_sent, sent_input, dtype=tf.float32,
              sequence_length=sent_enc_lens)
	  #cell_sent=wlstm_cell(hps.num_hidden)
	  #sent_input,sent_fw_state=cell_sent(sent_input)
      sent_encoder_outputs = sent_input
      #return sent_encoder_outputs


      with tf.variable_scope('decoder'):
        # When decoding, use model output from the previous step
        # for the next step.
        loop_function = None
        sent_loop_function = None
        #loop_function = seq2seq._extract_argmax_and_embed(
        #  embedding,hps.batch_size)
        #sent_loop_function=seq2seq.sent_extract_argmax_and_embed(
        #  sent_embedding)
        # if mode == 'decode':
        #   loop_function = _extract_argmax_and_embed(
        #       embedding, (w, v), update_embedding=False)

        self.cell = tf.nn.rnn_cell.LSTMCell(
            hps.num_hidden,
            initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
            state_is_tuple=False)
	#self.cell=wlstm_cell(hps.num_hidden)
        encoder_outputs = [tf.reshape(x, [hps.batch_size, 1, hps.num_hidden])
                           for x in encoder_outputs]
        enc_top_states = tf.concat(1, encoder_outputs)
        #dec_in_state = fw_state
        dec_in_state=tf.concat(1,[fw_state,bw_state])


        #print(enc_top_states)
        # During decoding, follow up _dec_in_state are fed from beam_search.
        # dec_out_state are stored by beam_search for next step feeding.

      #with tf.variable_scope('sent_decoder'):
        # When decoding, use model output from the previous step
        # for the next step.
        #loop_function = None
        # if mode == 'decode':
        #   loop_function = _extract_argmax_and_embed(
        #       embedding, (w, v), update_embedding=False)

        with tf.variable_scope('sent_decoder'):
          self.sent_cell = tf.nn.rnn_cell.LSTMCell(
              hps.num_hidden,
              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
              state_is_tuple=False)
        #print(sent_encoder_outputs)
	  #self.sent_cell=wlstm_cell(hps.num_hidden)
        sent_encoder_outputs = [tf.reshape(x, [hps.batch_size, 1, hps.num_hidden])
                           for x in sent_encoder_outputs]
        sent_enc_top_states = tf.concat(1, sent_encoder_outputs)

        #print(sent_encoder_outputs,sent_enc_top_states)
        if hps.mode== 'train':
          mode=True
        else:
          mode=False

        sent_dec_in_state = sent_fw_state
        sent_initial_state_attention = True #(mode == 'decode')
	self.decoder_outputs, self.dec_out_state,self.sent_decoder_outputs, self.sent_dec_out_state,self.switch_output,self.switch_prob,self.decoder_outputs_dists,self.sent_decoder_outputs_dists = seq2seq.attention_decoder(
            emb_decoder_inputs,encoder_inputs, dec_in_state, enc_top_states,self.cell,
            sent_emb_decoder_inputs, sent_input,sent_dec_in_state, sent_enc_top_states,
            self.sent_cell,hps.dec_timesteps,switch=switch,word_weights=word_weights, mode_train=mode,num_heads=1, loop_function=loop_function,sent_loop_function=sent_loop_function,
            initial_state_attention=sent_initial_state_attention)
        # switch_out=[tf.reshape(x, [hps.batch_size]) for x in self.switch_prob]
        # #self.switch_pred=tf.round(switch_out)
        # self.switch_pred=tf.to_float(tf.greater_equal(switch_out,threshold))
        # self.word_switch_pred=tf.add(1.0, tf.mul(-1.0,self.switch_pred))
      
        # sent_weights=[x*y for x,y in zip(switch_out, switch )]
        # weights =[(1-x)*(y) for x,y in zip(switch_out,word_weights)]

        #switch_out=[tf.reshape(x, [hps.batch_size,2]) for x in self.switch_prob]

        #self.switch_pred=tf.round(switch_out)
#        self.switch_pred=[x[:,1] for x in switch_out]
#	self.switch_pred=[tf.to_float(tf.greater_equal(x[:,1],threshold)) for x in switch_out]
       # self.word_switch_pred=tf.add(1.0, tf.mul(-1.0,self.switch_pred))
        #print(switch_out)

        #self.sent_weights=[x[:,0]*y for x,y in zip(switch_out, switch )]
        #self.weights =[x[:,1]*(y) for x,y in zip(switch_out,word_weights)]
        switch_target=[tf.to_int32(tf.greater_equal(x,1)) for x in switch]


        # t_decoder_outputs=tf.transpose(self.decoder_outputs,perm=[1,0,2])
        # t_sent_decoder_outputs=tf.transpose(self.sent_decoder_outputs,perm=[1,0,2])
        # t_switch_output=tf.transpose(self.switch_output,perm=[1,0,2])

        final_dists = self._calc_final_dist(self.decoder_outputs_dists, self.sent_decoder_outputs_dists)
        # Take log of final distribution

        log_dists = [tf.log(dist+1e-12) for dist in final_dists]
        with tf.variable_scope('loss'):
          #if FLAGS.pointer_gen: # calculate loss from log_dists
            # Calculate the loss per step
            # This is fiddly; we use tf.gather_nd to pick out the log probabilities of the target words
          loss_per_step = [] # will be list length max_dec_steps containing shape (batch_size)
          batch_nums = tf.range(0, limit=hps.batch_size) # shape (batch_size)
#	  sent_lens = tf.reduce_sum(self.switch_batch, axis=1)+1e-12
#          word_lens = tf.reduce_sum(self.word_weights_batch, axis=1)+1e-12
	  sent_lens=1
	  word_lens=1
          for dec_step, log_dist in enumerate(log_dists):
            target = extend_targets[dec_step] # The indices of the target words. shape (batch_size)
            indices = tf.stack( (batch_nums, target), axis=1) # shape (batch_size, 2)
            losses = tf.gather_nd(-log_dist, indices) # shape (batch_size). loss on this step for each batch
	    w=(word_weights[dec_step]/word_lens)+(switch[dec_step]/sent_lens)
            loss_per_step.append(losses*w)
         # print("loss_per_step",loss_per_step)
        self.loss =tf.reduce_mean(sum(loss_per_step))# self.mask_and_avg(loss_per_step,self.switch_weights_batch)
	self.final_log_dists=final_dists #log_dists

      if  hps.mode!='decode':
        with tf.variable_scope('word_loss'):
          self.word_loss=self.get_loss(
                self.decoder_outputs, targets,self.word_weights_batch)  
          self.word_loss_null=self.get_null_loss(
                self.decoder_outputs,words_decsent,weights_words_decsent,self.switch_batch)  
	  self.coverage_loss=  0.0 #self._coverage_loss(self.decoder_outputs_dists,self.word_weights_batch)

        with tf.variable_scope('sent_loss'):
          self.sent_loss=self.get_loss(
                self.sent_decoder_outputs_dists, sent_targets,self.switch_batch) 
          self.sent_loss_null=self.get_null_loss(
                self.sent_decoder_outputs,sent_decwords,weights_sent_decwords,self.word_weights_batch) 

	with tf.variable_scope('switch_loss'):
          self.switch_loss=seq2seq.sequence_loss(
                self.switch_output,switch_target, switch_weights,
                softmax_loss_function=None)
        #sent_loss=2*sent_loss        
        #sent_loss=2*sent_loss
        #self.loss=word_loss+sent_loss+switch_loss
	self.total_loss=self.loss+self.word_loss+self.sent_loss+self.sent_loss_null+self.word_loss_null +self.switch_loss #+self.coverage_loss
#	self.total_loss=self.loss + self.sent_loss_null+self.word_loss_null +self.switch_loss
        tf.scalar_summary('loss',tf.minimum(12.0,  self.loss))
 #       tf.scalar_summary('total_loss',  self.total_loss)
       # tf.scalar_summary('word_loss',  tf.minimum(12.0,self.word_loss))
        #tf.scalar_summary('sent_loss', tf.minimum(12.0, self.sent_loss))
       # tf.scalar_summary('sent_loss_null', tf.minimum(12.0, self.sent_loss_null))
       # tf.scalar_summary('word_loss_null', tf.minimum(120.0, self.word_loss_null))
       # tf.scalar_summary('switch_loss',  self.switch_loss)
       # tf.histogram_summary('switch_prob',switch_out)
   #     tf.histogram_summary('switch',switch)
   #     tf.histogram_summary('decoder_outputs',tf.nn.softmax(self.decoder_outputs,dim=1))
   #     tf.histogram_summary('sent_decoder_outputs',tf.nn.softmax(self.sent_decoder_outputs,dim=1))
#        tf.histogram_summary('sent_embedding',sent_embedding)
#        tf.histogram_summary('embedding',embedding)
        # with tf.variable_scope('output'), tf.device(self.next_device()):
        #   model_outputs = []
        #   for i in xrange(len(decoder_outputs)):
        #     if i > 0:
        #       tf.get_variable_scope().reuse_variables()
        #     model_outputs.append(
        #         tf.nn.xw_plus_b(decoder_outputs[i], w, v))

        # if hps.mode == 'decode':
        #   with tf.variable_scope('decode_output'), tf.device('/cpu:0'):
        #     best_outputs = [tf.argmax(x, 1) for x in model_outputs]
        #     tf.logging.info('best_outputs%s', best_outputs[0].get_shape())
        #     self.outputs = tf.concat(
        #         1, [tf.reshape(x, [hps.batch_size, 1]) for x in best_outputs])

        #     self.topk_log_probs, self.topk_ids = tf.nn.top_k(
        #         tf.log(tf.nn.softmax(model_outputs[-1])), hps.batch_size*2)

        # with tf.variable_scope('loss'), tf.device(self.next_device()):
        #   def sampled_loss_func(inputs, labels):
        #     with tf.device('/cpu:0'):  # Try gpu.
        #       labels = tf.reshape(labels, [-1, 1])
        #       return tf.nn.sampled_softmax_loss(w_t, v, inputs, labels,
        #                                         hps.num_softmax_samples, vsize)

        #   if hps.num_softmax_samples != 0 and hps.mode == 'train':
        #     self.loss = seq2seq_lib.sampled_sequence_loss(
        #         decoder_outputs, targets, loss_weights, sampled_loss_func)
        #   else:
        #     self.loss = tf.nn.seq2seq.sequence_loss(
        #         model_outputs, targets, loss_weights)
        #   tf.scalar_summary('loss', tf.minimum(12.0, self.loss))

  def add_train_op(self):
    """Sets self.train_op, op to run for training."""
    hps = self.hps
    loss_ind=self.loss_ind
    #self.loss_to_minimize=self.loss+self.word_loss_null+self.sent_loss_null

   # loss_to_minimize=tf.cond(tf.mod(self.global_step,400)>0,lambda:self.sent_loss+ self.word_loss_null,lambda:self.word_loss+self.sent_loss_null)
   # self.loss_to_minimize=self.loss + self.sent_loss_null+self.word_loss_null +self.switch_loss+self.word_loss
    self.loss_to_minimize=self.loss +self.switch_loss+self.word_loss

    self.lr_rate = tf.maximum(
        hps.min_lr,  # min_lr_rate.
        tf.train.exponential_decay(hps.lr, self.global_step, 30000, 0.98))

    tvars = tf.trainable_variables()
   # gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
   # with tf.device(self.get_gpu(self.num_gpus-1)):
    with tf.device('/gpu:1'):
       grads, global_norm = tf.clip_by_global_norm(
          tf.gradients(self.loss_to_minimize, tvars), hps.max_grad_norm)


#      grads, global_norm = tf.clip_by_global_norm(
#          gradients, hps.max_grad_norm)
   # tf.scalar_summary('global_norm', global_norm)
   # tf.scalar_summary('loss_to_minimize',  self.loss_to_minimize)
    self.grad_norms,self.grads,self.tvars=self.get_grad_norm(grads,tvars)
    #optimizer = tf.train.GradientDescentOptimizer(self.lr_rate)
    optimizer=tf.train.AdamOptimizer()

#    optimizer = tf.train.AdagradOptimizer(0.15, initial_accumulator_value=0.1)
   # tf.scalar_summary('learning rate', self.lr_rate)
    with tf.device('/gpu:1'):
          self.train_op = optimizer.apply_gradients(
            zip(grads, tvars), global_step=self.global_step, name='train_step')

  def get_grad_norm(self,grads,tvars):
    grad_norm=[]
    out_grads=[]
    out_tvars=[]
    for i,g in enumerate(grads):
      try:
      	if i <2:
        	t=tf.convert_to_tensor(g.values)
      	else:
        	t=tf.convert_to_tensor(g)
  	out_grads.append(g)
  	out_tvars.append(tvars[i])
        grad=tf.sqrt(tf.reduce_sum(t * t))
        grad_norm.append(grad)

      except:	
        print("name",i,tvars[i].name)
#      tf.scalar_summary(tvars[i].name, grad)
    return grad_norm, out_grads, out_tvars

  def encode_top_state(self, sess, enc_inputs, enc_len):
    """Return the top states from encoder for decoder.

    Args:
      sess: tensorflow session.
      enc_inputs: encoder inputs of shape [batch_size, enc_timesteps].
      enc_len: encoder input length of shape [batch_size]
    Returns:
      enc_top_states: The top level encoder states.
      dec_in_state: The decoder layer initial state.
    """
    results = sess.run([self.enc_top_states, self.dec_in_state],
                       feed_dict={self.articles: enc_inputs,
                                  self.article_lens: enc_len})
    return results[0], results[1][0]

  def decode_topk(self, sess, latest_tokens, enc_top_states, dec_init_states):
    """Return the topK results and new decoder states."""
    feed = {
        self.enc_top_states: enc_top_states,
        self.dec_in_state:
            np.squeeze(np.array(dec_init_states)),
        self.abstracts:
            np.transpose(np.array([latest_tokens])),
        self.abstract_lens: np.ones([len(dec_init_states)], np.int32)}

    results = sess.run(
        [self.topk_ids, self.topk_log_probs, self.dec_out_state],
        feed_dict=feed)

    ids, probs, states = results[0], results[1], results[2]
    new_states = [s for s in states]
    return ids, probs, new_states

  def build_graph(self,embed):
    # g = tf.Graph()
    # with g.as_default():
    self.embed=tf.convert_to_tensor(embed,name="embed",dtype=tf.float32)
    self.add_placeholders()
    with tf.device('/gpu:1'):
        self.add_seq2seq()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
#    if self.hps.mode == 'train':
    self.add_train_op()
    self.summaries = tf.merge_all_summaries()

# def build_eval_graph(self):
#   g1 = tf.Graph()
#   with g1.as_default():
#     self.add_placeholders()
#     self.add_seq2seq()
#     self.global_step = tf.Variable(0, name='global_step', trainable=False)
#     if self.hps.mode == 'train':
#       self.add_train_op()
#     self.summaries = tf.merge_all_summaries()
