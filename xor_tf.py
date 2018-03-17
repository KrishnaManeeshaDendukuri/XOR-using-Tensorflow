
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np

#initialization
x=tf.placeholder(tf.float64,shape=[4,2],name='x')
y=tf.placeholder(tf.float64,shape=[4,1],name='y')
l_r=1

theta1=tf.cast(tf.Variable(tf.random_normal([3,2])),tf.float64)
theta1=0.25*(theta1-0.5)
theta2=tf.cast(tf.Variable(tf.random_normal([3,1])),tf.float64)
theta2=0.25*(theta2-0.5)

#forward propagation

a1=tf.concat([np.c_[np.ones(x.shape[0])],x],1)
z1=tf.matmul(a1,theta1)
a2=tf.concat([np.c_[np.ones(x.shape[0])],tf.sigmoid(z1)],1)
z2=tf.matmul(a2,theta2)
a3=tf.sigmoid(z2)

#cost=tf.reduce_sum((y*tf.log(a3))+((1-y)*tf.log(1-a3)),axis=1)
cost=tf.reduce_mean(tf.square(y-a3))
optim= tf.train.GradientDescentOptimizer(learning_rate=l_r).minimize(cost)
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[[0],[1],[1],[0]]


#init=tf.global_variables_initializer()
init=tf.initialize_all_variables()
s=tf.Session()
s.run(init)


for i in range(1000):
    s.run(optim,feed_dict={x:X,y:Y})
    if ((i%100)==0):
        print("epoch",i)
        print("Hyp:",s.run(a3,feed_dict={x:X,y:Y}))


