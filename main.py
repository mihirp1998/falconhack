import itertools
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Input,Dense, LSTM
from keras.optimizers import Adam
from keras.layers import Lambda,Reshape
from keras.preprocessing import sequence
from keras.models import Model
from keras import backend as K
import cPickle
from keras.layers import RepeatVector
from keras.models import load_model
import tokenize
from os import listdir
from os.path import isfile, join
def main():
	
	# data = open('abc.txt','rb')
	# X = data.read().split('\n')
	# X = [i + ' EOS' for i in X]
	# token_docs= [i.split(' ') for i in X]

	# all_tokens = itertools.chain.from_iterable(token_docs)
	# word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
	# id_to_word = {token: idx for idx, token in enumerate(set(all_tokens))}

	word_to_id,id_to_word,token_docs = word_id()
	vocab_size= len(word_to_id.keys())

	token_ids = [[[word_to_id[token] ]for token in docs] for docs in token_docs]

	vec = OneHotEncoder(n_values=len(word_to_id))
	doc = []

	for i in token_ids:	
		doc.append(np.array(vec.fit_transform(i).toarray()))
	doc = np.array(doc)
	# return doc
	# X = [i[:(i.shape[0])] for i in doc]
	X = doc 
	# return X
	tx = [len(i) for i in X]

	Y = [i[1:,:] for i in doc]
	# return X,Y	
	Y = sequence.pad_sequences(Y,maxlen = max(tx), padding='post')
	# Y = [i.T for i in Y]
	X = sequence.pad_sequences(X,maxlen = max(tx), padding='post')
	# Y = np.transpose(Y,(1,0,2))


	# X = sequence.pad_sequence(X,)
	return X,Y,tx,vocab_size
def testMain(token_docs):
	
	# data = open('abc.txt','rb')
	# X = data.read().split('\n')
	# X = [i + ' EOS' for i in X]
	# token_docs= [i.split(' ') for i in X]

	# all_tokens = itertools.chain.from_iterable(token_docs)
	# word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
	# id_to_word = {token: idx for idx, token in enumerate(set(all_tokens))}

	word_to_id,id_to_word,_ = word_id()
	vocab_size= len(word_to_id.keys())

	token_ids = [[[word_to_id[token] ]for token in docs] for docs in token_docs]

	vec = OneHotEncoder(n_values=len(word_to_id))
	doc = []

	for i in token_ids:	
		doc.append(np.array(vec.fit_transform(i).toarray()))
	doc = np.array(doc)
	X = [i[:(i.shape[0]-1)] for i in doc] 
	# return X
	tx = [len(i) for i in X] 

	Y = [i[1:,:] for i in doc]	
	# return X
	Y = sequence.pad_sequences(Y,maxlen = 210, padding='post')
	# Y = [i.T for i in Y]
	X = sequence.pad_sequences(X,maxlen = 210, padding='post')
	# Y = np.transpose(Y,(1,0,2))


	# X = sequence.pad_sequence(X,)
	return X,Y,tx,vocab_size	
def code_pre():
	totalArr = []
	files = [join('./code/train',f) for f in listdir('./code/train') if isfile(join('./code/train',f)) and 'java' in f]
	files = ['./code/train/train1.java']
	# print(files)
	for f in files:
		fileArr = []
		print f
		def appendArr(a,b,c,d,e):
			fileArr.append(b)
		file  = open(f,'rb')
		tokenize.tokenize(file.readline,appendArr)
		totalArr.append(fileArr)
	# cPickle.dump(arr,open('vocab.p','wb'))
	return totalArr
def word_id():
	# data = open('abc.txt','rb')
	# X = data.read().split('\n')
	# X = [i + ' EOS' for i in X]
	# token_docs= [i.split(' ') for i in X]
	token_docs =  code_pre()
	# return token_docs
	all_tokens = itertools.chain.from_iterable(token_docs)
	word_to_id = {token: idx for idx, token in enumerate(set(all_tokens))}
	all_tokens = itertools.chain.from_iterable(token_docs)
	id_to_word = {idx : token for idx, token in enumerate(set(all_tokens))}
	return word_to_id,id_to_word,token_docs

def modela(LSTM_cell,densor,vocab__size,tx,n_a):
	reshapor = Reshape((1,vocab__size))
	# reshapora = Reshape((1,1))
	a0 = Input(shape = (n_a,))
	c0 = Input(shape = (n_a,))
	# tx = Input(shape = (1,1))
	a = a0
	c = c0
	X = Input(shape = (None,vocab__size))
	outputs = []
	# print(tx[0,0,0])
	# tf.constant(tx[0,0,0]	
	# return tx[0,0,0]
	for t in range(tx):
		x =  Lambda(lambda x: X[:,t,:])(X)
		# print('time step one',x)
		x = reshapor(x)
		print('x',x,'a',a,'c',c)
		a, _, c = LSTM_cell(x, initial_state=[a, c])
		out = densor(a)
		outputs.append(out)	
	model = Model(inputs=[X,a0,c0], outputs= outputs)
	return model


def predictModel(LSTM_cell, densor, n_values , n_a , Ty ,Tn):
	X = Input(shape=(None, n_values))
	reshapor = Reshape((1,n_values))
	a0 = Input(shape=(n_a,), name='a0')
	c0 = Input(shape=(n_a,), name='c0')
	a = a0
	c = c0
	# x = x0
	outputs = []
	newOutputs = []
	for t in range(Ty):
		x =  Lambda(lambda x: X[:,t,:])(X)
		x = reshapor(x)
		a, _, c = LSTM_cell(X, initial_state=[a, c])
		out = densor(a)
		outputs.append(out)
		if t is (Ty- 1):
			x = Lambda(one_hot,arguments={'vocab_size':n_values})(out)
	for t in range(Tn):
		a, _, c = LSTM_cell(x, initial_state=[a, c])
		out = densor(a)
		newOutputs.append(out)
		x = Lambda(one_hot,arguments={'vocab_size':n_values})(out)		
	inference_model = Model(inputs =[X,a0,c0],outputs= newOutputs)	
	return inference_model


def one_hot(x,vocab_size):
    x = K.argmax(x)
    x = tf.one_hot(x, vocab_size) 
    x = RepeatVector(1)(x)
    return x


def make():
	n_a =30
	X,Y,tx,vocab_size = main()
	time_step_size = max(tx)
	LSTM_cell = LSTM(n_a, return_state = True) 
	densor = Dense(vocab_size, activation='softmax')
	model  = modela(LSTM_cell,densor,vocab_size,time_step_size,n_a)
	# cPickle.dump(LSTM_cell,open('lstm.p','wb'))
	# return model(vocab_size)
	
	opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	m = X.shape[0]
	
	a0 = np.zeros((m, n_a))
	c0 = np.zeros((m, n_a))

	model.fit([X, a0, c0], list(Y), epochs=1000)
	predictedModel = predictModel(LSTM_cell,densor,vocab_size,n_a,1,5)
	predictedModel.save('predictCode.h5')
	
	
def predict():
	maxtx = 6
	word_to_id,id_to_word,token_docs = word_id()
	vocab_size = len(word_to_id.keys())
	model =load_model('predictCode.h5')
	x_initializer = np.zeros((1, 1, vocab_size))
	idwod = word_to_id['public'] 
	x_initializer[:,:,idwod] =  1
	# X = sequence.pad_sequences(x_initializer,maxlen = maxtx)
	a_initializer = np.zeros((1, 30))
	c_initializer = np.zeros((1, 30))
	newPreds = model.predict([x_initializer,a_initializer,c_initializer])
	# return newPreds
	pred = np.argmax(newPreds,axis = 2)
	pred = pred.flatten()
	predArray = [id_to_word[i] for i in pred]
	predWord = ' '.join(predArray)
	
	return predWord,newPreds
	
	