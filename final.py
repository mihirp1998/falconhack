

from __future__ import print_function

import tensorflow as tf
import random
import main
import numpy as np
from code import temp
# ====================
#  TOY DATA GENERATOR
# ====================
class ToySequenceData(object):
	""" Generate sequence of data with dynamic length.
	This class generate samples for training:
	- Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
	- Class 1: random sequences (i.e. [1, 3, 10, 7,...])
	NOTICE:
	We have to pad each sequence to reach 'max_seq_len' for TensorFlow
	consistency (we cannot feed a numpy array with inconsistent
	dimensions). The dynamic calculation will then be perform thanks to
	'seqlen' attribute that records every actual sequence length.
	"""
	def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,
				 max_value=1000):
		self.data = []
		self.labels = []
		self.seqlen = []
		for i in range(n_samples):
			# Random sequence length
			len = random.randint(min_seq_len, max_seq_len)
			# Monitor sequence length for TensorFlow dynamic calculation
			self.seqlen.append(len)
			# Add a random or linear int sequence (50% prob)
			if random.random() < .5:
				# Generate a linear sequence
				rand_start = random.randint(0, max_value - len)
				s = [[float(i)/max_value] for i in
					 range(rand_start, rand_start + len)]
				# Pad sequence for dimension consistency
				s += [[0.] for i in range(max_seq_len - len)]
				self.data.append(s)
				self.labels.append([1., 0.])
			else:
				# Generate a random sequence
				s = [[float(random.randint(0, max_value))/max_value]
					 for i in range(len)]
				# Pad sequence for dimension consistency
				s += [[0.] for i in range(max_seq_len - len)]
				self.data.append(s)
				self.labels.append([0., 1.])
		self.batch_id = 0

	def next(self, batch_size):
		""" Return a batch of data. When dataset end is reached, start over.
		"""
		if self.batch_id == len(self.data):
			self.batch_id = 0
		batch_data = (self.data[self.batch_id:min(self.batch_id +
												  batch_size, len(self.data))])
		batch_labels = (self.labels[self.batch_id:min(self.batch_id +
												  batch_size, len(self.data))])
		batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
												  batch_size, len(self.data))])
		self.batch_id = min(self.batch_id + batch_size, len(self.data))
		return batch_data, batch_labels, batch_seqlen



# learning_rate = 0.01
# training_steps = 10000
# batch_size = 128
# display_step = 200

# # Network Parameters
# seq_max_len = 20 # Sequence max length
# n_hidden = 64 # hidden layer num of features
# n_classes = 2 # linear sequence or not

# trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)
# testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)

# # tf Graph input
# x = tf.placeholder("float", [None, seq_max_len, 1])
# y = tf.placeholder("float", [None, n_classes])
# # A placeholder for indicating each sequence length
# seqlen = tf.placeholder(tf.int32, [None])

# # Define weights
# weights = {
#     'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
# }
# biases = {
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }


def dynamicRNN( x,y,seqlen,vocab_size,seq_max_len):
	n_hidden = 50
#     # Prepare data shape to match `rnn` function requirements
#     # Current data input shape: (batch_size, n_steps, n_input)
#     # Required shape: 'n_steps' tensors list of shape (batch_size, n_input) 
	# Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
	x_unstack = tf.unstack(x, seq_max_len, 1)
	weights = {
	    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]),name = 'w1')
	}

	biases = {
	    'out': tf.Variable(tf.zeros([vocab_size]),name='b1')
	}
#     # Define a lstm cell with tensorflow
	lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,name="lstm_cell")


#     # Get lstm cell output, providing 'sequence_length' will perform dynamic
#     # calculation.
	outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x_unstack, dtype=tf.float32,
								sequence_length=seqlen)
	# return outputs,states

# # When performing dynamic calculation, we must retrieve the last
# # dynamically computed output, i.e., if a sequence length is 10, we need
# # to retrieve the 10th output.
# # However TensorFlow doesn't support advanced indexing yet, so we build
# # a custom op that for each sample in batch size, get its length and
# # get the corresponding relevant output.
# # 'outputs' is a list of output at every timestep, we pack them in a Tensor
# # and change back dimension to [batch_size, n_step, n_input]


	# return outputs,outputsPred

	outputs = tf.stack(outputs)
	outputs = tf.transpose(outputs, [1, 0, 2])
	batch_size = tf.shape(outputs)[0] 
	outputs = tf.reshape(outputs, [-1, n_hidden])
	outputs  = tf.matmul(outputs, weights['out']) + biases['out']
	
	allOutputs = tf.nn.softmax(outputs)
	allOutputs = tf.reshape(allOutputs,[batch_size,seq_max_len,vocab_size],name="allOutputs")

	outputs = tf.reshape(outputs,[batch_size,seq_max_len,vocab_size])
	
	index = tf.range(0,batch_size)*seq_max_len + (seqlen -1)
	outputPred = tf.gather(tf.reshape(outputs,[-1,vocab_size]),index)
	outputPred = tf.nn.softmax(outputPred,name = "predictedOutputs")
	outputPred  = tf.argmax(outputPred,axis = -1)
	predictedOutputs = []
	outputPred =  tf.one_hot(outputPred,vocab_size,name ="one_hot")
	# states = tf.identity(states,name="states")
	predictedOutputs.append(outputPred)
	for i in range(20):
		outputPred, states = lstm_cell(outputPred,states)
		outputPred = tf.matmul(outputPred, weights['out']) + biases['out']
		# softmax_cross_entropy_with_logits
		outputPred = tf.nn.softmax(outputPred,name ="softymax")
		outputPred  = tf.argmax(outputPred,axis = -1,name="argymax")
		outputPred =  tf.one_hot(outputPred,vocab_size,name="one_hot_vec")
		predictedOutputs.append(outputPred)


	predictedOutputs = tf.stack(predictedOutputs,name="predictedOutputing")	


	return outputs,predictedOutputs

def mainRun():
	tf.reset_default_graph()
	learning_rate = 0.01
	x1,y1,t,vocab_size  = main.main()
	seq_max_len = max(t)

	x = tf.placeholder("float", [None,seq_max_len, vocab_size],name="x")
	y = tf.placeholder("float", [None,seq_max_len, vocab_size],name= "y")
	seqlen = tf.placeholder(tf.int32, [None],name ="seqlen")
	
	pred,b = dynamicRNN(x,y,seqlen,vocab_size,max(t))
	# return pred,b
	# return pred,b
	init = tf.global_variables_initializer()
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y),name="theloss")
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,name="the_gdo").minimize(cost,name="theoptimizer")
	saver = tf.train.Saver()
	sess = tf.Session()
	# saver = tf.train.import_meta_graph('saved/model-10000.meta')
	# saver.restore(sess,tf.train.latest_checkpoint('saved/.'))

	sess.run(init)
	# graph = tf.get_default_graph()
	# pred = graph.get_tensor_by_name("b1:0")
	# print(sess.run(pred))
	# return

	# l,m  = sess.run([pred,b],feed_dict={x:x1,y:y1,seqlen:t})
	# return l,m,i

	for i in range(10001):
		 l,o  = sess.run([cost,optimizer],feed_dict={x:x1,y:y1,seqlen:t})
		 if not(i % 500):
			print("saving model")
		 	saver.save(sess,'./saved/SMallmodel',global_step = i)
		 print("l {} o {} epoch {}".format(l,o,i)) 
def hello():
	print("life's not good good")
def mainContinue():
	tf.reset_default_graph()
	learning_rate = 0.01
	x1,y1,t,vocab_size  = main.main()
	sess = tf.Session()
	saver = tf.train.import_meta_graph('saved/SMallmodel-10000.meta')
	saver.restore(sess,tf.train.latest_checkpoint('saved/.'))
	graph = tf.get_default_graph()
	xP = graph.get_tensor_by_name("x:0")
	yP = graph.get_tensor_by_name("y:0")
	seqlenP = graph.get_tensor_by_name("seqlen:0")

	
	cost =graph.get_tensor_by_name("theloss:0")
	b = graph.get_tensor_by_name("b1:0")

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,name="the_gdo").minimize(cost,name="theoptimizer")
	
	
	for i in range(10000,50000):
		 l,_  = sess.run([cost,optimizer],feed_dict={xP:x1,yP:y1,seqlenP:t})
		 # print(l,b)
		 if not(i % 500):
			print("saving model")
		 	saver.save(sess,'./saved/SMallmodel',global_step = i)
		 print("l {}  epoch {}".format(l,i))		  
def restore():
	tf.reset_default_graph()
	token_code = temp.getCode()
	x1,y1,t,vocab_size = main.testMain(token_code)
	x1,y1,t,vocab_size = main.main()

	word_to_id,id_to_word,token_docs = main.word_id()
	sess = tf.Session()
	x1 = np.zeros((1, max(t), vocab_size))
	idwod = word_to_id['import']
 	x1[:,:,idwod] =  1
	t = [1]
	saver = tf.train.import_meta_graph('saved/SMallmodel-10000.meta')
	saver.restore(sess,tf.train.latest_checkpoint('saved/.'))
	graph = tf.get_default_graph()
	pred = graph.get_tensor_by_name("predictedOutputing:0")
	all_outputs  = graph.get_tensor_by_name("allOutputs:0")
	all_outputs  = tf.argmax(all_outputs,axis = -1,name="AllOargymax")
	all_outputs =  tf.one_hot(all_outputs,vocab_size,name="AllOone_hot_vec")
	predictedOutputs = []
	# for i in range(10):
	# 	outputPred, states = lstm_cell(outputsPred,states)
	# 	outputPred = tf.matmul(outputPred, weights['out']) + biases['out']
	# 	# softmax_cross_entropy_with_logits
	# 	outputPred = tf.nn.softmax(outputPred)
	# 	predictedOutputs.append(outputPred)
	# 	outputPred  = tf.argmax(outputPred)
	# 	outputPred =  tf.one_hot(outputPred,vocab_size)


	# predictedOutputs = tf.stack(predictedOutputs,name="predictedOutputs")	
	xP = graph.get_tensor_by_name("x:0")
	yP = graph.get_tensor_by_name("y:0")
	seqlenP = graph.get_tensor_by_name("seqlen:0")
	cost =graph.get_tensor_by_name("theloss:0")
	# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=yP))	

	p,cost,all_outputs  = sess.run([pred,cost,all_outputs],feed_dict={xP:x1,seqlenP:t,yP:y1})
	pred = np.argmax(p,axis = 2)
	pred = pred.flatten()
	predArray = [id_to_word[i] for i in pred]
	predWord = ' '.join(predArray)
	aText = np.argmax(all_outputs,axis = 2)
	aText = aText.flatten()
	aText = [id_to_word[i] for i in aText]
	aText = ' '.join(aText)
	return p,aText,predWord,cost


# pred = dynamicRNN(x, seqlen, weights, biases)

# # Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# # Evaluate model
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# # Initialize the variables (i.e. assign their default value)
# init = tf.global_variables_initializer()

# # Start training
# with tf.Session() as sess:

#     # Run the initializer
#     sess.run(init)

#     for step in range(1, training_steps + 1):
#         batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
#         # Run optimization op (backprop)
#         sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
#                                        seqlen: batch_seqlen})
#         if step % display_step == 0 or step == 1:
#             # Calculate batch accuracy & loss
#             acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,
#                                                 seqlen: batch_seqlen})
#             print("Step " + str(step*batch_size) + ", Minibatch Loss= " + \
#                   "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                   "{:.5f}".format(acc))

#     print("Optimization Finished!")

#     # Calculate accuracy
#     test_data = testset.data
#     test_label = testset.labels
#     test_seqlen = testset.seqlen
#     print("Testing Accuracy:", \
#         sess.run(accuracy, feed_dict={x: test_data, y: test_label,
#                                       seqlen: test_seqlen}))
if __name__ == "__main__":
	mainRun()
