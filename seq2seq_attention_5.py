# Copyri gh2016 The TensorFlow Authors. All Rights Reserved.
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

"""This is the top-level file to train, evaluate or test your summarization model"""
import sys
import time

import tensorflow as tf
from tensorflow.python.platform import gfile
import tensorflow.python.debug as tf_debug
import batcher_5 as batch_reader
#import data
import seq2seq_attention_model_13 as seq2seq_attention_model
#import seq2seq_attention_model_graph as seq2seq_attention_model
import random
import os
import numpy as np

import read_data_emb as read_data
#import seq2seq_6 as seq2seq
import util
sys.path.insert(0, '/datadrive/lstm_codes/textsum/codes/rouge/pyrouge')
from pyrouge import Rouge155



FLAGS = tf.app.flags.FLAGS

# Where to find data
tf.app.flags.DEFINE_string('data_path', '/datadrive/lstm_codes/textsum/codes/dataset/finished_files/chunked/train*', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path expression to text vocabulary file.')
tf.app.flags.DEFINE_string('train_path', '/datadrive/lstm_codes/textsum/codes/new_codes/wn_2', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('rouge_dir', '/datadrive/lstm_codes/textsum/codes/rouge/ROUGE-1.5.5/', 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')

# Important settings
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_boolean('single_pass', False, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Where to save output
tf.app.flags.DEFINE_string('log_root', '/datadrive/lstm_codes/textsum/codes/new_codes/wn_nocvg', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', 'exp_1', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')
tf.app.flags.DEFINE_string('model_dir',
                           'model_dir_rough', 'Path expression to tf.Example.')
tf.app.flags.DEFINE_string('system_dir',
                           'system_dir_rough', 'Path expression to tf.Example.')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 200, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 100, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 800, 'max timesteps of encoder (max source text tokens)')
tf.app.flags.DEFINE_integer('max_dec_steps', 50, 'max timesteps of decoder (max summary tokens)')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for beam search decoding.')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'Minimum sequence length of generated summary. Applies only for beam search decoding mode')
tf.app.flags.DEFINE_integer('max_article_sentences', 50,
                            'Max number of first sentences to use from the '
                            'article')
tf.app.flags.DEFINE_integer('vocabulary_size', 150000, 'Size of vocabulary. These will be read from the vocabulary file in order. If the vocabulary file contains fewer words than this number, or if this number is set to 0, will take all words in the vocabulary file.')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value for Adagrad')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'magnitude for lstm cells random uniform inititalization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init, used for initializing everything else')
tf.app.flags.DEFINE_float('max_grad_norm', 1.5, 'for gradient clipping')
tf.app.flags.DEFINE_bool('use_bucketing', False,
                         'Whether bucket articles of similar length.')
tf.app.flags.DEFINE_bool('truncate_input', True,
                         'Truncate inputs that are too long. If False, '
                         'examples that are too long are discarded.')

thresh=3
print('Parameters:',tf.flags.FLAGS.__flags)
print('Notes: Copied from wn_3, switch is present,added word and sent null losses,changed NUL_ID is  4,embedding=googlenews')
# Pointer-generator or baseline model
def prepare_vocab():
    """Fill input queue with ModelInput."""
    vocab_path = os.path.join(FLAGS.train_path, "vocab%d.txt" % FLAGS.vocabulary_size)
    key_path = os.path.join(FLAGS.train_path, "keyword%d.txt" % FLAGS.vocabulary_size)
  #create_vocabulary(fr_vocab_path, train_path + "test_target.txt", fr_vocabulary_size, tokenizer)
    if not (gfile.Exists(vocab_path) and gfile.Exists(key_path)):
      print('No Vocabulary and keywords exist')
      #read_data.create_vocab(vocab_path, key_path,FLAGS.data_path, FLAGS.vocabulary_size,FLAGS.file_list)
    print('Reading Vocabulary and keywords')
    vocab, re_vocab = read_data.initialize_vocabulary(vocab_path)
    key=read_data.initialize_keywords(key_path)
    embed=read_data.get_embedding(vocab,FLAGS.emb_dim)
    return vocab,re_vocab,key,embed

def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
  """Calculate the running average loss via exponential decay.
  This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

  Args:
    loss: loss on the most recent eval step
    running_avg_loss: running_avg_loss so far
    summary_writer: FileWriter object to write for tensorboard
    step: training iteration step
    decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

  Returns:
    running_avg_loss: new running average loss
  """
  if running_avg_loss == 0:  # on the first iteration just take the loss
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  #summary_writer.add_summary(loss_sum, step)
  tf.logging.info('running_avg_loss: %f', running_avg_loss)
  return running_avg_loss

def get_rouge():

  rouge_args = '-e /home/aishwarya/rouge/ROUGE-1.5.5/data -n 4 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -x -l 100'
  r=Rouge155(FLAGS.rouge_dir)
  r.model_dir=FLAGS.model_dir
  r.system_dir=FLAGS.system_dir
  r.model_filename_pattern='model.#ID#.txt'
  r.system_filename_pattern='system.(\d+).txt'
  output = r.convert_and_evaluate()
  print("Rouge score:")
  print(output)


def switch_fscore(final_dists,target_labels,re_vocab,next_enc_batch,articles,abstracts,start=0):

#print(next_batch())
        trueNegative = 0
        truePositive = 0
        falseNegative = 0
        falsePositive = 0
        count=start

        #print((next_enc_batch))
        final_dist=np.transpose(final_dists,axes=(1, 0, 2))
        #sent_output=np.transpose(sent_decoder_outputs,axes=(1, 0, 2))
        #et_input=np.transpose(next_enc_batch,axes=(1, 0, 2))
        #target=np.transpose(target_labels)
        #switch=np.transpose(switch_labels)

  #      s_sent=np.transpose(switch_pred)
        # s_word=np.transpose(word_switch_pred)
        # s_sent=switch_labels #"To be deleted"

        for f_output,f_target, ee_input,article,abstract in zip(final_dist,target_labels,next_enc_batch,articles,abstracts):
                p=0
                f=0
		if count >0:
			break
                if np.count_nonzero(ee_input)==0:
                        continue
                pred_sent_ind=[]
                pred_word_ind=[]
                gnd_sent_ind=[]
                gnd_word_ind=[]
                pred_out=[]
		pred_switch=[]
                gnd_out=[]
                sent_prob=[]
                for  ff_output,ff_target in zip(f_output, f_target):
                                #print("Before",groundTruth)
                                #print("predict",np.shape(ff_output))
                                #predict=sess.run(softmax,feed_dict={pred_place:predict})
                                #print("After",predict)
                        #print("predict", np.shape(predict), np.shape(f_output),np.shape(ft_output))
                        #iff_sent=int(ff_sent)

                        max_ind=np.argmax(ff_output)
                        max_ind_sent=np.argmax(ff_output[FLAGS.max_enc_steps:])
                        max_ind_word=np.argmax(ff_output[:FLAGS.max_enc_steps])
                        #if (ff_sent > 0):
                       # if (max_ind > FLAGS.max_enc_steps-1 ):
                         # max_ind=max_ind-FLAGS.max_enc_steps
			 # if max_ind == read_data.EOS_ID:
			#	break
                         # pred_sent_ind.append(max_ind)
                      #    pred_out.append(['sent:',max_ind])
                          #sent_prob.append(ff_output[max_ind])
                        #else:
			if max_ind == read_data.EOS_ID:
				break
                        #  pred_word_ind.append(max_ind)
                        pred_out.append(max_ind)
                        #if(ff_switch >0):
                        pred_sent_ind.append(max_ind_sent)
                       # gnd_out.append(['sent:',ff_target])
                       # else:
                        pred_word_ind.append(max_ind_word)
                        gnd_out.append(ff_target)
		print('gnd:',gnd_out)
		print('pred:',pred_out)
                print('sentence:',pred_sent_ind)
                print('word:',pred_word_ind)
               # print('switch',f_sprob,f_switch)
               # print('pred_output',pred_out)
               # print('gnd_output',gnd_out)


def get_rouge_dir(ee_input,sent_prob,pred_sent_ind,article,count):

                model_file=open(os.path.join(FLAGS.model_dir, "model.%d.txt" % count),"w+")
                system_file=open(os.path.join(FLAGS.system_dir, "system.%d.txt" % count),"w+")
    
#                count=count+1
                #print(ee_input)
               # print("max_ind,groundTruth",pred_ind,gnd_ind)
                #print("sentence")
                ee_input=ee_input[ee_input.tolist().index(read_data.UNK_ID)+1:]
                if read_data.PAD_ID in ee_input:
                        ee_input=ee_input[:ee_input.tolist().index(read_data.PAD_ID)]
                #print(" ".join([tf.compat.as_str(re_vocab[output]) for output in ee_input]))
                #print("article")
                if read_data.EOS_ID in pred_sent_ind:
                        sent_prob=sent_prob[:pred_sent_ind.index(read_data.EOS_ID)]
                        pred_sent_ind=pred_sent_ind[:pred_sent_ind.index(read_data.EOS_ID)]

                sent_prob=np.array(sent_prob).argsort()
                max_sent_prob=list(reversed(sent_prob))
                if len(max_sent_prob) > thresh:
                        max_sent_prob=max_sent_prob[:thresh]
                out_ind=sorted(set([pred_sent_ind[i] for i in max_sent_prob]))
                #out_ind=[ out for out in output_ind if out < len(article)+4]
                #print(out_ind,gnd_sent_ind,article)
                if out_ind != []:
                        #system_file.write(" ".join([tf.compat.as_str(re_vocab[output]) for output in ee_input]) 
                        system_words=[ word for output in out_ind for word in article[output-5]]
                        #system_words=system_words[:100]
                        system_file.write(" ".join([tf.compat.as_str(word) for word in system_words]))
                        model_file.write("".join([tf.compat.as_str(word) for word in abstract ]))
      # print("write")
# def convert_to_coverage_model():
#   """Load non-coverage checkpoint, add initialized extra variables for coverage, and save as new checkpoint"""
#   tf.logging.info("converting non-coverage model to coverage model..")

#   # initialize an entire coverage model from scratch
#   sess = tf.Session(config=util.get_config())
#   print "initializing everything..."
#   sess.run(tf.global_variables_initializer())

#   # load all non-coverage weights from checkpoint
#   saver = tf.train.Saver([v for v in tf.global_variables() if "coverage" not in v.name and "Adagrad" not in v.name])
#   print "restoring non-coverage variables..."
#   curr_ckpt = util.load_ckpt(saver, sess)
#   print "restored."

#   # save this model and quit
#   new_fname = curr_ckpt + '_cov_init'
#   print "saving model to %s..." % (new_fname)
#   new_saver = tf.train.Saver() # this one will save all variables that now exist
#   new_saver.save(sess, new_fname)
#   print "saved."
#   exit()


def setup_training(model, batcher,re_vocab,embed):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  default_device = tf.device('/cpu:0')
  #default_device = tf.device('/gpu:1')
  with default_device:
    model.build_graph(embed) # build the graph
    # if FLAGS.convert_to_coverage_model:
    #   assert FLAGS.coverage, "To convert your non-coverage model to a coverage model, run with convert_to_coverage_model=True and coverage=True"
    #   convert_to_coverage_model()
  saver = tf.train.Saver(max_to_keep=10) # only keep 1 checkpoint at a time
#  saver=tf.train.import_meta_graph(os.path.join(train_dir,'model.ckpt-279720.meta'))


  sv = tf.train.Supervisor(logdir=train_dir,
                     is_chief=True,
                     saver=saver,
                     summary_op=None,
                     save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                     save_model_secs=10, # checkpoint every 60 secs
                     global_step=model.global_step)
  summary_writer = sv.summary_writer
  tf.logging.info("Preparing or waiting for session...")
  sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
#  model.hps._replace(lr=FLAGS.lr)
  tf.logging.info("Created session.")
  try:
    run_training(model, batcher,re_vocab,embed, sess_context_manager, sv, summary_writer,saver) # this is an infinite loop until interrupted
  except KeyboardInterrupt:
    tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
    sv.stop()

def get_gradients(model,batch,sess):
	nan_results=[]
	inf_results=[]
	out_grads,variables,grad_norms=sess.run([model.grads,model.tvars,model.grad_norms],feed_dict=model.make_feed(batch))
	names=[v.name for v in model.tvars]
	for g,v,n,gn in zip(out_grads,variables,names,grad_norms):
	   if gn > 0:
		print("varaible norm: ", n,gn)
		try:
			g_array=np.asarray(g.values)
		except AttributeError:
			g_array=np.asarray(g)
		if g_array.any == np.nan:
			nan_results.append(n)
		if g_array.any ==np.inf:
			inf_results.append(n)
	print("nan variables: ",nan_results)
	print("inf variables: ",inf_results)

def run_training(model, batcher,re_vocab,embed, sess_context_manager, sv, summary_writer,saver):
  """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
  tf.logging.info("starting run_training")
  loss_ind=0
  with sess_context_manager as sess:
    while True: # repeats until interrupted
      batch = batcher.next_batch()
#      print("Input data",batch.enc_batch[0], batch.dec_batch[0],batch.sent_enc_batch[0], batch.sent_dec_batch[0],batch.target_batch[0],batch.extend_target_batch[0],batch.sent_target_batch[0],batch.switch_batch[0], batch.enc_input_lens[0], batch.sent_enc_input_lens[0],batch.dec_output_lens[0],batch.word_weights[0],batch.switch_weights[0],batch.sent_decwords[0],batch.words_decsent[0],batch.weights_sent_decwords[0],batch.weights_words_decsent[0])

#      tf.logging.info('running training step...')
      t0=time.time()
      step_output= model.run_train_step(sess,batch)
      #    sess, batch.enc_batch, batch.dec_batch,batch.sent_enc_batch, batch.sent_dec_batch,batch.target_batch,batch.extend_target_batch,batch.sent_target_batch,batch.switch_batch, batch.enc_input_lens, batch.sent_enc_input_lens,batch.dec_output_lens,
       #   batch.word_weights,batch.switch_weights,loss_ind)
      t1=time.time()
     # tf.logging.info('seconds for training step: %.3f', t1-t0)
 #     get_gradients(model,batch,sess)
      (_, summaries, loss, train_step)=step_output[0]
      (out_decoder_outputs,out_sent_decoder_outputs,final_dists)=step_output[1]
      # print("article_batch",article_batch)
      (step_loss,word_loss,sent_loss,word_loss_null,sent_loss_null,switch_loss)=step_output[2]
      #loss = results['loss']
     # tf.logging.info('loss: %f', loss) # print the loss to screen
     # tf.logging.info('step_loss: %f word_loss: %f ,sent_loss: %f ,word_loss_null: %f,sent_loss_null: %f',step_loss,word_loss,sent_loss,word_loss_null,sent_loss_null)
      # if FLAGS.coverage:
      #   coverage_loss = results['coverage_loss']
      #   tf.logging.info("coverage_loss: %f", coverage_loss) # print the coverage loss to screen

      # get the summaries and iteration number so we can write summaries to tensorboard
      # summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
      # train_step = results['global_step'] # we need this to update our running average loss
      
  #    summary_writer.add_summary(summaries, train_step) # write the summaries
      if train_step % 50 == 0: # flush the summary writer every so often
  #    if train_step == 21490: # flush the summary writer every so often
#	_ = util.load_ckpt(saver, sess)
#	tf.logging.info('loss_ind: %d',loss_ind)
#	loss_ind=(loss_ind+1)%4
        tf.logging.info('train_step: %d', train_step) # print the loss to screen
        tf.logging.info('loss: %f', loss) # print the loss to screen

	tf.logging.info('step_loss: %f word_loss: %f ,sent_loss: %f ,word_loss_null: %f,sent_loss_null: %f ,switch_loss: %f',step_loss,word_loss,sent_loss,word_loss_null,sent_loss_null,switch_loss)
#	switch_fscore(final_dists,batch.extend_target_batch,re_vocab,batch.enc_batch,batch.origin_articles,batch.origin_abstracts)
#        print("Input data",batch.enc_batch[0], batch.dec_batch[0],batch.sent_enc_batch[0], batch.sent_dec_batch[0],batch.target_batch[0],batch.extend_target_batch[0],batch.sent_target_batch[0],batch.switch_batch[0], batch.enc_input_lens[0], batch.sent_enc_input_lens[0],batch.dec_output_lens[0],batch.word_weights[0],batch.switch_weights[0])
        summary_writer.flush()



def run_eval(model, batcher,re_vocab,embed):
  """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
  model.build_graph(embed) # build the graph
  saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
  sess = tf.Session(config=util.get_config())
  eval_dir = os.path.join(FLAGS.log_root, "eval") # make a subdir of the root dir for eval data
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
  summary_writer = tf.summary.FileWriter(eval_dir)
  running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
  best_loss = None  # will hold the best loss achieved so far

  while True:
    _ = util.load_ckpt(saver, sess) # load a new checkpoint
    batch = batcher.next_batch() # get the next batch

    # run eval on the batch
    t0=time.time()
#    results = model.run_eval_step(sess, batch)
    step_output= model.run_eval_step(sess,batch)
        #  sess, batch.enc_batch, batch.dec_batch,batch.sent_enc_batch, batch.sent_dec_batch,batch.target_batch,batch.extend_target_batch,batch.sent_target_batch,batch.switch_batch, batch.enc_input_lens, batch.sent_enc_input_lens,batch.dec_output_lens,
         # batch.word_weights,batch.switch_weights)
    t1=time.time()

    tf.logging.info('seconds for batch: %.2f', t1-t0)
    (summaries, loss, train_step)=step_output[0]
    (out_decoder_outputs,out_sent_decoder_outputs,out_switch_pred,out_word_switch_pred)=step_output[1]
      # print("article_batch",article_batch)
    (step_loss,word_loss,sent_loss,switch_loss)=step_output[2]
      #loss = results['loss']
    tf.logging.info('loss: %f', loss) # print the loss to screen
    tf.logging.info('step_loss: %f ,word_loss: %f ,sent_loss: %f ,switch_loss: %f',step_loss,word_loss,sent_loss,switch_loss)

    # print the loss and coverage loss to screen
    summary_writer.add_summary(summaries, train_step)

    # calculate running avg loss
    running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)

    # If running_avg_loss is best so far, save this checkpoint (early stopping).
    # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
    if best_loss is None or running_avg_loss < best_loss:
      tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
      saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
      best_loss = running_avg_loss

    #fscore(out_decoder_outputs,out_sent_decoder_outputs,out_switch_pred,out_word_switch_pred,batch.target_batch,batch.switch_batch,re_vocab,batch.enc_batch,batch.origin_articles,batch.origin_abstracts)
    switch_fscore(final_dists,out_switch_pred,batch.extend_target_batch,batch.switch_batch,re_vocab,batch.enc_batch,batch.origin_articles,batch.origin_abstracts)
    get_rouge()

    # flush the summary writer every so often
    if train_step % 100 == 0:
      summary_writer.flush()

def run_decode(model, batcher, re_vocab,embed):
  """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
  model.build_graph(embed) # build the graph
  saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
  sess = tf.Session(config=util.get_config())
  eval_dir = os.path.join(FLAGS.log_root, "eval") # make a subdir of the root dir for eval data
  #bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
  #print(saver)
  summary_writer = tf.summary.FileWriter(eval_dir)
  running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
  best_loss = None  # will hold the best loss achieved so far
  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)
    os.makedirs(FLAGS.system_dir)
  _ = util.load_ckpt(saver, sess) # load a new checkpoint
  ckpt_state = tf.train.get_checkpoint_state(eval_dir,latest_filename="checkpoint_best")
  tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
  saver.restore(sess, ckpt_state.model_checkpoint_path)
  count=0
  while True:
          _ = util.load_ckpt(saver, sess) # load a new checkpoint
          batch = batcher.next_batch() # get the next batch
          if batch is None:
                get_rouge()
                break
          # run eval on the batch
          t0=time.time()
          step_output= model.run_eval_step(
                sess, batch.enc_batch, batch.dec_batch,batch.sent_enc_batch, batch.sent_dec_batch,batch.target_batch,batch.extend_target_batch,batch.sent_target_batch,batch.switch_batch, batch.enc_input_lens, batch.sent_enc_input_lens,batch.dec_output_lens,batch.word_weights,batch.switch_weights)
          t1=time.time()
          tf.logging.info('seconds for batch: %.2f', t1-t0)
         # print("original",count,batch.origin_articles)
          # print the loss and coverage loss to screen

          ( summaries, loss, train_step)=step_output[0]
          (out_decoder_outputs,out_sent_decoder_outputs,out_switch_pred,out_word_switch_pred)=step_output[1]
          tf.logging.info('loss: %f', loss)
          # if FLAGS.coverage:
          #   coverage_loss = results['coverage_loss']
          #   tf.logging.info("coverage_loss: %f", coverage_loss)

          # add summaries
          # summaries = results['summaries']
          # train_step = results['global_step']
                                                                                                                                                       
          summary_writer.add_summary(summaries, train_step)

          # calculate running avg loss
          running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)

          # If running_avg_loss is best so far, save this checkpoint (early stopping).
          # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
         # if best_loss is None or running_avg_loss < best_loss:
         #   tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
          #  saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
          print("running_avg_loss",running_avg_loss)
	  switch_fscore(final_dists,out_switch_pred,batch.extend_target_batch,batch.switch_batch,re_vocab,batch.enc_batch,batch.origin_articles,batch.origin_abstracts,count)
#          fscore(out_decoder_outputs,out_sent_decoder_outputs,out_switch_pred,out_word_switch_pred,batch.target_batch,batch.switch_batch,re_vocab,batch.enc_batch,batch.origin_articles,batch.origin_abstracts,count)
          count=count+1
          get_rouge()
          # flush the summary writer every so often
          #if train_step % 100 == 0:
          #summary_writer.flush()

                                                                                                                                                                                  

def main(unused_argv):
  if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
    raise Exception("Problem with flags: %s" % unused_argv)

  tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
  tf.logging.info('Starting seq2seq_attention in %s mode...', (FLAGS.mode))
  print('Initialising')
  
  vocab,re_vocab,key,embed = prepare_vocab()
  voacb_start=[read_data.PAD_ID,read_data.GO_ID,read_data.EOS_ID,read_data.UNK_ID,read_data.NUL_ID]
  #max_article_sentences=5
  start_enc=len(voacb_start)
  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
  FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
  if not os.path.exists(FLAGS.log_root):
    if FLAGS.mode=="train":
      os.makedirs(FLAGS.log_root)
    else:
      raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

  #vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size) # create a vocabulary
  if not os.path.exists(FLAGS.model_dir):
    os.makedirs(FLAGS.model_dir)
    os.makedirs(FLAGS.system_dir)
  # If in decode mode, set batch_size = beam_size
  # Reason: in decode mode, we decode one example at a time.
  # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.

  # If single_pass=True, check we're in decode mode
  if FLAGS.single_pass and FLAGS.mode!='decode':
    raise Exception("The single_pass flag should only be True in decode mode")

  # # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
  # hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm', 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt', 'pointer_gen']
  # hps_dict = {}
  # for key,val in FLAGS.__flags.iteritems(): # for each flag
  #   if key in hparam_list: # if it's in the list
  #     hps_dict[key] = val # add it to the dict
  # hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

  # # Create a batcher object that will create minibatches of data
  # batcher = Batcher(FLAGS.data_path, vocab, hps, single_pass=FLAGS.single_pass)
  hps = seq2seq_attention_model.HParams(
      mode=FLAGS.mode,  # train, eval, decode
      min_lr=0.01,  # min learning rate.
      lr=FLAGS.lr ,#0.15,  # learning rate
      batch_size=FLAGS.batch_size,
      enc_layers=1,
      enc_timesteps=FLAGS.max_enc_steps,#40,
      dec_timesteps=FLAGS.max_dec_steps, #10,
      sent_enc_timesteps=FLAGS.max_article_sentences+start_enc,
      max_article_sentences=FLAGS.max_article_sentences,
      min_input_len=2,  # discard articles/summaries < than this
      num_hidden=FLAGS.hidden_dim,  # for rnn cell
      emb_dim=FLAGS.emb_dim,  # If 0, don't use embedding
      vocabulary_size=FLAGS.vocabulary_size,
      max_grad_norm=2,
      num_softmax_samples=40)  # If 0, no sampled softmax.


  # Create a batcher object that will create minibatches of data
  batcher = batch_reader.Batcher(
      FLAGS.data_path,vocab,hps,start_enc, single_pass=FLAGS.single_pass, bucketing=FLAGS.use_bucketing,
      truncate_input=FLAGS.truncate_input)

  tf.set_random_seed(111) # a seed value for randomness

  if hps.mode == 'train':
    print "creating model..."
#    model = SummarizationModel(hps, vocab)
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
				hps, vocab, num_gpus=1)
    setup_training(model, batcher,re_vocab,embed)
  elif hps.mode == 'eval':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=2)
    run_eval(model, batcher,re_vocab,embed)
  elif hps.mode == 'test':
    model = seq2seq_attention_model.Seq2SeqAttentionModel(
        hps, vocab, num_gpus=2)
    run_decode(model, batcher,re_vocab,embed)

  
  else:
    raise ValueError("The 'mode' flag must be one of train/eval/decode")

if __name__ == '__main__':
  tf.app.run()
