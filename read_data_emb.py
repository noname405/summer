from nltk.corpus import stopwords
import re
import gensim
from tensorflow.python.platform import gfile
from gensim.summarization import keywords
import sys
#sys.path.insert(0,'RAKE-tutorial')
#import rake
import operator
#stop = set(stopwords.words('english'))
import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2
import math
import pickle
import numpy as np
# data_path='/home/aishwarya/Documents/sum/models/textsum/neuralsum/neuralsum/cnn/training/'
# filenames=open('/home/aishwarya/Documents/sum/models/textsum/filenames.txt')
#stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','--']) # remove it if you need punctuation 

_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_NUL = b"_NUL"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK,_NUL]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
NUL_ID=4

_DIGIT_RE = re.compile(br"\s\d+\s|\s\d+$")
SYB_RE=re.compile(b"([.,!?\"':;)(])|--")
model_path='/datadrive/lstm_codes/textsum/codes/dataset/finished_files/dm/embed_model.bin'

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
SENTENCE_START={}
SENTENCE_END={}
SENTENCE_START['article'] = '<a>'
SENTENCE_END['article'] = '</a>'
SENTENCE_START['word'] = '<w>'
SENTENCE_END['word'] = '</w>'
SENTENCE_END['abstract'] = '<eos>'
#vocab_path='/home/aishwarya/Documents/sum/models/textsum/vocab.txt'
def read_files(data_path,key,file):
	d={}
	count=0
	
	sentence=[]
	label=[]
	f=open(data_path+file)
	s=f.readline()
	s=f.readline()
	s=f.readline()
	while s!='\n':
		sent=s.strip().split('\t')
		sent[0]=SYB_RE.sub(r'',sent[0])
		sent[0]=_DIGIT_RE.sub(b" 0",sent[0])
		#sent[0]=re.sub(b"([.,!?\"':;)(])|--",r'',sent[0])
		
		sentence.append(sent[0].strip())
		label.append(sent[3])
		s=f.readline()
	# d['sentences']=sentence
	# d['sent_labels']=label
	w=[]
	s=f.readline()
	while s!='\n':
		w+=[i for i in s.lower().split() if (i  in key or bool(re.match('entity@'r'\d',i)))]

		s=f.readline()
	w_set=set(w)

	word_label=[]
	for sent in sentence:
		wd=[1 if k in w else 0 for k in sent.strip().split()]
		word_label.append(wd)
	#print(sentence)
	return sentence,label,w,word_label

def create_vocab(vocabulary_path,keyword_path, data_path, max_vocabulary_size,file_list):
	vocab={}
	counter=0
	tokens=[]
	abstract=""
	filenames=open(file_list)
	for file in filenames:
		counter=counter+1
		#print('file counter',counter)
		f=open(data_path+file.strip())
		s=f.readline()
		s=f.readline()
		s=f.readline()
		while s!='\n':
			s=SYB_RE.sub(r'',s)
			tokens += s.strip().split()
			s=f.readline()
		s=f.readline()	
		while s!='\n':
			#print(s)
			abstract+=s
			s=f.readline()
	print('file counter',counter)
	abstract=abstract.replace("\n", "")
	abstract=SYB_RE.sub(r' ',abstract)
	print('abstract',len(abstract.split()))
	#keys=keywords(abstract,split='True',ratio=0.01)
	#keys=keywords(abstract,split='True',words=max_vocabulary_size)
	rake_object = rake.Rake("RAKE-tutorial/SmartStoplist.txt",1,1,1)
	rake_keywords=rake_object.run(abstract)
	keys=[k[0] for k in rake_keywords]
	keys=keys[:max_vocabulary_size]
	print('keys')
	tokens+=keys
	for w in tokens:
		word = _DIGIT_RE.sub(b" 0", w)
		if word in vocab:
			vocab[word] += 1
		else:
			vocab[word] = 1
	print("Creating vocabulary")
	vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
	if len(vocab_list) > max_vocabulary_size:
		vocab_list = vocab_list[:max_vocabulary_size]
	with open(vocabulary_path, mode="wb") as vocab_file:
		for w in vocab_list:
			vocab_file.write(w + b"\n")
	print("Creating Keywords")
	with open(keyword_path, mode="wb") as key_file:
		for w in keys:
			word=_DIGIT_RE.sub(b" 0", w)
			#if (word.encode('utf-8') in vocab_list):
			if (word in vocab_list):
				key_file.write(word + b"\n")

#create_vocab(vocab_path,data_path,40)

# for file in filenames:
# 	print(read_files(file))

def basic_tokenizer(sentence):
	"""Very basic tokenizer: split the sentence into a list of tokens."""
	words = []
	for space_separated_fragment in sentence.strip().split():
		words.extend(_WORD_SPLIT.split(space_separated_fragment))
	return [w for w in words if w]


def initialize_vocabulary(vocabulary_path):
	"""Initialize vocabulary from file.

	We assume the vocabulary is stored one-item-per-line, so a file:
		dog
		cat
	will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
	also return the reversed-vocabulary ["dog", "cat"].

	Args:
		vocabulary_path: path to the file containing the vocabulary.

	Returns:
		a pair: the vocabulary (a dictionary mapping string to integers), and
		the reversed vocabulary (a list, which reverses the vocabulary mapping).

	Raises:
		ValueError: if the provided vocabulary_path does not exist.
	"""
	if gfile.Exists(vocabulary_path):
		rev_vocab = []
		with gfile.GFile(vocabulary_path, mode="rb") as f:
			rev_vocab.extend(f.readlines())
		rev_vocab = [line.strip() for line in rev_vocab]
		vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
		return vocab, rev_vocab
	else:
		raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def initialize_keywords(key_path):
	
	if gfile.Exists(key_path):
		key = []
		with gfile.GFile(key_path, mode="rb") as f:
			key.extend(f.readlines())
		key = [line.strip() for line in key]
		#vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
		return key
	else:
		raise ValueError("keyword file %s not found.", key_path)


def sentence_to_token_ids(sentence, vocabulary,
													tokenizer=None, normalize_digits=True):
	"""Convert a string to list of integers representing token-ids.

	For example, a sentence "I have a dog" may become tokenized into
	["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
	"a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

	Args:
		sentence: the sentence in bytes format to convert to token-ids.
		vocabulary: a dictionary mapping tokens to integers.
		tokenizer: a function to use to tokenize each sentence;
			if None, basic_tokenizer will be used.
		normalize_digits: Boolean; if true, all digits are replaced by 0s.

	Returns:
		a list of integers, the token-ids for the sentence.
	"""

	if tokenizer:
		words = tokenizer(sentence)
	else:
		#words = basic_tokenizer(sentence)
		words = sentence#sentence.strip().split()
		#print([w for w in words])
	if not normalize_digits:
		return [vocabulary.get(w, UNK_ID) for w in words]
	# Normalize digits by 0 before looking words up in the vocabulary.
	return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
											tokenizer=None, normalize_digits=True):
	"""Tokenize data file and turn into token-ids using given vocabulary file.

	This function loads data line-by-line from data_path, calls the above
	sentence_to_token_ids, and saves the result to target_path. See comment
	for sentence_to_token_ids on the details of token-ids format.

	Args:
		data_path: path to the data file in one-sentence-per-line format.
		target_path: path where the file with token-ids will be created.
		vocabulary_path: path to the vocabulary file.
		tokenizer: a function to use to tokenize each sentence;
			if None, basic_tokenizer will be used.
		normalize_digits: Boolean; if true, all digits are replaced by 0s.
	"""
	if not gfile.Exists(target_path):
		print("Tokenizing data in %s" % data_path)
		vocab, _ = initialize_vocabulary(vocabulary_path)
		with gfile.GFile(data_path, mode="rb") as data_file:
			with gfile.GFile(target_path, mode="w") as tokens_file:
				counter = 0
				for line in data_file:
					counter += 1
					if counter % 100000 == 0:
						print("  tokenizing line %d" % counter)
					token_ids = sentence_to_token_ids(line, vocab, tokenizer,
																						normalize_digits)
					tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def get_embedding(vocab,embedding_size):
	
	
	sqrt3 = math.sqrt(3)
	#vocab, _ = initialize_vocabulary(vocabulary_path)
	size_vocab=len(vocab)
	embed=np.zeros((size_vocab,embedding_size))
	model = gensim.models.Word2Vec.load_word2vec_format(model_path, binary=True)  
	for (k, v) in vocab.iteritems() :
		if k in model.vocab:

			w=_DIGIT_RE.sub(b"0", k)
			embed[v]=model[w]
		else:
			embed[v]=np.random.uniform(-sqrt3,sqrt3,embedding_size)
	return embed


def get_pos(in_path, out_path):
	pos_num=pickle.load(open("/home/aishwarya/Documents/point/pos_num.p"))
	infile=open(in_path)
	outfile=open(out_path,"wb")

	st=[]
	for line in infile:
		
		l=line.strip().split()
		if l ==[]:
			#print(st)
			for i in st:
				outfile.write(str(i) + " ")
			outfile.write("\n")
			st=[]
		else:
			if l[1] in pos_num:
				st.append(pos_num[l[1]])
				
			else:
				st.append(3)

	infile.close()
	outfile.close()

def article2sents(abstract,mode='article'):
	"""Splits abstract text from datafile into list of sentences.

	Args:
		abstract: string containing <s> and </s> tags for starts and ends of sentences

	Returns:
		sents: List of sentence strings (no tags)"""
	cur = 0
	sents = []
	while True:
		try:
			start_p = abstract.index(SENTENCE_START[mode], cur)
			end_p = abstract.index(SENTENCE_END[mode], start_p + 1)
			cur = end_p + len(SENTENCE_END[mode])
			sents.append(abstract[start_p+len(SENTENCE_START[mode]):end_p])
		except ValueError as e: # no more sentences
			#print("sents",sents)
			return sents


def text2labels(abstract,mode='word'):
	"""Splits abstract text from datafile into list of sentences.

	Args:
		abstract: string containing <s> and </s> tags for starts and ends of sentences

	Returns:
		sents: List of sentence strings (no tags)"""
	cur = 0
	
	labels=[]
	while True:
		try:
			sents = []
			start_p = abstract.index(SENTENCE_START[mode], cur)
			end_p = abstract.index(SENTENCE_END[mode], start_p + 1)
			cur = end_p + len(SENTENCE_END[mode])
			sents=(abstract[start_p+len(SENTENCE_START[mode]):end_p])
			#print("sents1" ,sents)
			sents=[int(s) for s in sents.strip().split(' ')]
			#print("sents2" ,sents)
			labels.append(sents)
		except ValueError as e: # no more sentences
			return labels

def abstract2sents(abstract,mode='abstract'):
        """Splits abstract text from datafile into list of sentences.

        Args:
                abstract: string containing <s> and </s> tags for starts and ends of sentences

        Returns:
                sents: List of sentence strings (no tags)"""
        cur = 0
        sents = []
        while True:
                try:
                        start_p =cur
                        end_p = abstract.index(SENTENCE_END[mode], start_p + 1)
                        cur = end_p + len(SENTENCE_END[mode])
                        sents.append(abstract[start_p:end_p])
                except ValueError as e: # no more sentences
                        #print("sents",sents)
                        return sents.append(abstract[cur:])

def example_generator(data_path, single_pass):
	"""Generates tf.Examples from data files.

		Binary data format: <length><blob>. <length> represents the byte size
		of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
		the tokenized article text and summary.

	Args:
		data_path:
			Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
		single_pass:
			Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.

	Yields:
		Deserialized tf.Example.
	"""
	while True:
		filelist = glob.glob(data_path) # get the list of datafiles
		assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
		if single_pass:
			filelist = sorted(filelist)
		else:
			random.shuffle(filelist)
		for f in filelist:
			reader = open(f, 'rb')
			while True:
				len_bytes = reader.read(8)
				if not len_bytes: break # finished reading this file
				str_len = struct.unpack('q', len_bytes)[0]
				example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
				#print(example_str)
				yield example_pb2.Example.FromString(example_str)

		if single_pass:
			print "example_generator completed reading all datafiles. No more data."
			break

