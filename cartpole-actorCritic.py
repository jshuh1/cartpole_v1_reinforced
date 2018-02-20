import time
import tensorflow as tf
import gym
import numpy as np
import math

start = time.time()

hiddenSize = 300
hiddenSize2 = 800
learning_rate = 0.001
gamma = 0.99

#actor
state = tf.placeholder(shape=[None, 4], dtype=tf.float32)
W = tf.Variable(tf.truncated_normal([4,hiddenSize], stddev=0.1, dtype=tf.float32))
b1 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.1, dtype=tf.float32))
hidden = tf.nn.relu(tf.matmul(state,W)+b1)
O = tf.Variable(tf.truncated_normal([hiddenSize,2], stddev=0.1, dtype=tf.float32))
b2 = tf.Variable(tf.truncated_normal([2], stddev=0.1, dtype=tf.float32))
output = tf.nn.softmax(tf.matmul(hidden,O)+b2)

#critic
V1 = tf.Variable(tf.truncated_normal([4, hiddenSize2], stddev=0.1, dtype=tf.float32))
v1Out = tf.nn.relu(tf.matmul(state, V1))
V2 = tf.Variable(tf.truncated_normal([hiddenSize2,1], dtype=tf.float32, stddev=0.1))
vOut = tf.matmul(v1Out, V2)


rewards = tf.placeholder(shape=[None], dtype=tf.float32)
actions = tf.placeholder(shape=[None], dtype=tf.int32)
indices = tf.range(0, tf.shape(output)[0]) * 2 + actions
actProbs = tf.gather(tf.reshape(output, [-1]), indices)
vLoss = tf.reduce_mean(tf.square(rewards - vOut))
loss = -tf.reduce_mean(tf.log(actProbs) * rewards) + vLoss
optimizer = tf.train.AdamOptimizer(learning_rate)
trainOp = optimizer.minimize(loss)

init = tf.global_variables_initializer()
game =gym.make("CartPole-v1")
result = 0
numTrials = 3

for trial in range(numTrials):
	with tf.Session() as sess:
		sess.run(init)
		totRs = []
		currentBest = 0
		for i in range(1, 1001):
			hist = []
			disRs = []
			hist_temp = []
			st = game.reset()
			for j in range(1000):
				actDist = sess.run(output, feed_dict={state:[st]})
				act = np.random.choice(2, p=actDist[0])
				st1,rwd,dn,_ = game.step(act)
				#game.render()
				hist.append((st, act, rwd))
				hist_temp.append((st, act, rwd))
				st = st1
				if dn or (j != 0 and j % 50 == 0):
					for m in range(len(hist_temp)):
						temp = 0
						for k in range(len(hist_temp)-m):
							temp += math.pow(gamma,k) * hist_temp[m+k][2]
						critic = sess.run(vOut, feed_dict={state:[hist_temp[-1][0]]})
						disRs.append(temp - critic[0][0])
					sess.run(trainOp, feed_dict={state: [el[0] for el in hist], actions: [el[1] for el in hist], rewards: disRs})
					if dn:
						totRs.append(j)
						break
					else:
					 	hist_temp = []		 	
			if i % 100 == 0:
				if np.mean(totRs[-100:]) > 300:
					result += np.mean(totRs[-100:])
					print ("Trial: ", trial+1, " - intermediate result: ", np.mean(totRs[-100:]))
					break;	
			if i == 1000:
				result += np.mean(totRs[-100:])
				print ("Trial: ", trial+1, " - intermediate result: ", np.mean(totRs[-100:]))
end = time.time()

print ("The mean reward collected for the last 100 episodes of each trial: " )
print (result/numTrials)
print ("Time: ")
print (end-start)











				
