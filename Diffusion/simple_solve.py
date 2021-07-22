# coding: utf-8
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os,signal
from pyDOE import lhs 
import random
import matplotlib.pyplot as plt 
import util 


def ooops():
    print('ooops')


NT = 33
NX = 33

delta_x = 1./NX 
delta_t = 1./NT
is_sigint_up = False

alf = 1.
beta1 = 50.
beta2 = 50.
beta3 = 50.
beta4 = 50.

INTRACT = True
OPTIMIZER = tf.train.AdamOptimizer
ALL_DATA_NUM = 100000


TRAIN_STEP = 6*10000
hlayers_option = [i for i in range(5,6)]
nodes_option = [i for i in range(21,22)]
randNumOption = [i for i in range(1424,1425)]
bcsample_option = [i for i in range(233,234)]

num_hlayers = random.sample(hlayers_option, 1)[0]
num_nodes = random.sample(nodes_option, 1)[0]
bcSampleNum = random.sample(bcsample_option, 1)[0]
randEnable = random.sample(randNumOption, 1)[0]

print('Number of hidden layers: {}'.format(num_hlayers))
print('Number of nodes per hidden layer: {}'.format(num_nodes))
print('Sample numbers: {} {}'.format(bcSampleNum, randEnable))


# Build a list to contain nodes on each layers
# Example: [2, 5, 5, 5,1]
layers = [0]*(num_hlayers + 2)
for j in range(len(layers)):
    if j == 0:
        # Input 
        layers[j] = 2
    elif j == len(layers) - 1:
        # Output
        layers[j] = 1
    else:
        # Hidden
        layers[j] = num_nodes

# Assistive code 
LOG_NAME = 'log_'+'h_'+str(num_hlayers)+'n_'+str(num_nodes)+'r_'+str(randEnable)+'b_'+str(bcSampleNum)
if os.path.exists(LOG_NAME):
    print("Same Hyperameter exists. Exitting ...")
    exit()
else:
    os.makedirs(LOG_NAME)

MODEPATH = LOG_NAME+'/'

LAST_MODE_PATH = 'last_model/'
if os.path.exists(LAST_MODE_PATH):
    pass
else:
    os.makedirs(LAST_MODE_PATH) 




#################
# Build network 
#################
with tf.name_scope('input'):
    ts = tf.placeholder(tf.float32,shape=(None, 1))
    xs = tf.placeholder(tf.float32,shape=(None, 1))
    ts_s = tf.placeholder(tf.float32,shape=(None, 1))
    xs_s = tf.placeholder(tf.float32,shape=(None, 1))
    x_zeros = tf.placeholder(tf.float32,shape=(None,1))
    t_zeros = tf.placeholder(tf.float32,shape=(None,1))
    x_ones = tf.placeholder(tf.float32,shape=(None,1))
    t_ones = tf.placeholder(tf.float32,shape=(None,1))
    delx = tf.placeholder(tf.float32,shape=(None,1))
    one_p_delx = tf.placeholder(tf.float32,shape=(None,1))
    one_m_delx = tf.placeholder(tf.float32,shape=(None,1))
    m_delx = tf.placeholder(tf.float32,shape=(None,1))

    x_bc = tf.placeholder(tf.float32,shape=(None,1))
    x_p_one_bc = tf.placeholder(tf.float32,shape=(None,1))
    t_bc = tf.placeholder(tf.float32,shape=(None,1))
    x_m_one_bc = tf.placeholder(tf.float32,shape=(None,1))


num_layers = len(layers)
l = [0]*(num_layers-1)
Weights = [0]*(num_layers-1)
Biases = [0]*(num_layers-1)
w_rec = [0]*(num_layers-1)
b_rec = [0]*(num_layers-1)


# A function to handle a interrupt signal: Ctrl + C
def sigint_handler(signum, frame):
  global is_sigint_up
  is_sigint_up = True
  print ('Interrupted!')
signal.signal(signal.SIGINT, sigint_handler)


def init_NN():
    global Weights,Biases,w_rec,b_rec,layers,l,num_layers
    for i in range(len(layers)-1):
        # Try to load existing weights
        try:
            temp = np.loadtxt('w_rec'+str(i)+'.txt', dtype=np.float32)
            temp = temp[:,np.newaxis].reshape([layers[i],layers[i+1]])
            Weights[i] = tf.Variable(temp)
            print('Well Done')
        # if no existing weights, then give random wights
        except Exception as e:
            ooops()
            Weights[i] = tf.Variable(tf.random_normal([layers[i],layers[i+1]]))
  
        try:
            if i != num_layers - 2:
                temp = np.loadtxt('b_rec'+str(i)+'.txt', dtype=np.float32)
                temp = temp[:,np.newaxis].reshape([1,layers[i+1]])
                Biases[i] = tf.Variable(temp)
                print('Well Done')
            else:
                temp = np.loadtxt('b_rec'+str(i)+'.txt', dtype=np.float32)
                temp = temp.reshape([1,layers[i+1]])
                Biases[i] = tf.Variable(temp)
                print('Well Done')           
        except Exception as e:
            ooops()
            Biases[i] = tf.Variable(tf.zeros([1,layers[i+1]]) + 0.1)

# Forward calculation
def forward(t,x):
    global Weights,Biases,w_rec,b_rec,layers,l,num_layers
    inputs = tf.concat([t,x],1)
    h = inputs
    for l in range(num_layers-2):
        w = Weights[l]
        b = Biases[l]
        h = tf.sigmoid(tf.add(tf.matmul(h,w),b))
    w = Weights[-1]
    b = Biases[-1]
    Y = tf.add(tf.matmul(h,w),b)
    return Y

# PINN
def net_f(t,x):
    c = 7.
    pi = np.pi
    u = forward(t,x)
    u_t = tf.gradients(u,t)[0]
    u_x = tf.gradients(u,x)[0]
    u_xx = tf.gradients(u_x,x)[0]
    f = u_t - u_xx + c*(tf.sin(2*pi*x))*u
    return f


#################
# Prepare data
#################

# 生成10000对数据
all_data = lhs(2,ALL_DATA_NUM)

# 从10000个数据里面随机选num个数据，用来训练
def get_traing_data(all_data, num):
    idx = np.random.choice(all_data.shape[0], num, replace=False)
    traing_data = all_data[idx,:]
    t_data = traing_data[:,0]
    x_data = traing_data[:,1]
    t_data = t_data[:,np.newaxis]
    x_data = x_data[:,np.newaxis]
    return t_data, x_data

# t_data
temp_ = np.linspace(0,1,NT)
temp_ = temp_[0:]
sumv_ = temp_ 
for i in range(NX-1):
    sumv_ = np.append(sumv_,temp_)
t_data = sumv_.reshape(-1,1)

t_small_data = np.linspace(0,1,NT)
t_small_data = t_small_data[:, np.newaxis]
t_small_data = t_small_data.reshape([NT,1])


# x_data
temp = np.linspace(0,1,NX)
temp = temp[:,np.newaxis]
x_data = np.repeat(temp,NT,axis=0)

x_small_data = np.linspace(0,1,NX)
x_small_data = x_small_data[:, np.newaxis]
x_small_data = x_small_data.reshape([NX,1])

NT_zeros = np.zeros([NT,1])
NT_ones = np.ones([NT,1])

NX_zeros = np.zeros([NX,1])
NX_ones = np.ones([NX,1])

delx_data = delta_x * np.ones([NX,1])
one_p_delx_data = 1. + delx_data
one_m_delx_data = 1. - delx_data
m_delx_data = -1*delx_data

########################
# Define loss function
########################

init_NN()
f_pred = net_f(ts, xs)
U = forward(ts, xs)

E = forward(t_bc,x_bc)
F = forward(t_bc,x_p_one_bc)
F2 = forward(t_bc,x_m_one_bc)

G = forward(t_zeros, xs_s)

SSEu = alf*tf.reduce_mean(tf.square(f_pred))
SSEb1 = beta1*(tf.reduce_mean(tf.square(E-F)))
SSEb2 = beta2*(tf.reduce_mean(tf.square(G-1)))
SSEb3 = beta2*(tf.reduce_mean(tf.square(E-F2)))


with tf.name_scope('loss'):
    loss = SSEu + SSEb1 + SSEb2 + SSEb3
    tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    train_step = OPTIMIZER().minimize(loss)

########################
# train NN
########################
init = tf.global_variables_initializer()

# Get numeric solutions
u_num = util.get_num_solution(NX,NT,7.)
u_num = u_num.reshape(NX,NT,order='C')
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x_axis = np.linspace(0,1,NX)
for i in range(0,32,5):
    ax.scatter(x_axis,u_num[:,i], c='b', s=15, alpha = 1,marker='o')
plt.xlabel('x')
plt.ylabel('u')
plt.xlim(0,1)
plt.ylim(0.8,2.2)
lines = [0]*33
if INTRACT == True:
    plt.ion()
    plt.show()

# Color list is for showing the fixed color of plots
color_list = ['g', 'r', 'c', 'm', 'y','k', 'plum', 'deepskyblue', 'lightgreen','crimson', 'b']

with tf.Session() as sess:
    sess.run(init)
    t_bc_data, x_bc_data = get_traing_data(all_data,bcSampleNum)
    t_rand_data, x_rand_data = get_traing_data(all_data,randEnable)

    train_dic = {
                    ts: t_rand_data, xs: x_rand_data, ts_s: t_small_data,
                    t_zeros: NT_zeros, t_ones: NT_ones,
                    x_zeros: NX_zeros, xs_s: x_small_data,
                    x_bc: x_bc_data, t_bc: t_bc_data,
                    x_p_one_bc: x_bc_data+1,x_m_one_bc: x_bc_data-1
                }
    
    for i in range(TRAIN_STEP):  
        sess.run(train_step, feed_dict=train_dic)
        if i % 100 == 0:
            loss_value = sess.run(loss, feed_dict=train_dic)
            print(loss_value)

            for j in range(num_layers-1):
                w_rec[j] = Weights[j].eval()
                b_rec[j] = Biases[j].eval()
                np.savetxt(LAST_MODE_PATH+'w_rec'+str(j)+'.txt' , w_rec[j])
                np.savetxt(LAST_MODE_PATH+'b_rec'+str(j)+'.txt' , b_rec[j])
            np.savetxt(LAST_MODE_PATH+'loss.txt',np.reshape(loss_value,[1,1]))

            if is_sigint_up:
                # Save model
                for j in range(num_layers-1):
                    w_rec[j] = Weights[j].eval()
                    b_rec[j] = Biases[j].eval()
                    np.savetxt(MODEPATH+'w_rec'+str(j)+'.txt' , w_rec[j])
                    np.savetxt(MODEPATH+'b_rec'+str(j)+'.txt' , b_rec[j])
                print("Model saved")
                np.savetxt(MODEPATH+'loss.txt',np.reshape(loss_value,[1,1]))
                plt.savefig(MODEPATH+"result.png")
                print('Image saved.')
                break
            
            # Get the results from the NN
            u_out = sess.run(U, feed_dict={ts:t_data,xs:x_data})
            u_out = u_out.reshape(NX,NT,order='C')
            
            # Show lines interactively
            for i in range(0,32,5):
                try:
                    ax.lines.remove(lines[i][0])
                except Exception:
                    pass
            counter = 0
            for i in range(0,32,5):
                lines[i] = ax.plot(x_axis,u_out[:,i], lw=2,label='t={:.2f}'.format(i*delta_t), color=color_list[counter])
                counter += 1
            plt.title('Loss:{:.5f}'.format(loss_value) + ' R: '+str(randEnable)+' b: '+str(bcSampleNum) + ' ' +str(layers))
            if INTRACT == True:
                plt.pause(0.0001)
                plt.legend()
    
    
    # Save model when ending
    plt.savefig(MODEPATH+"result.png")
    print('Image saved.')
    for j in range(num_layers-1):
        w_rec[j] = Weights[j].eval()
        b_rec[j] = Biases[j].eval()
        np.savetxt(MODEPATH+'w_rec'+str(j)+'.txt' , w_rec[j])
        np.savetxt(MODEPATH+'b_rec'+str(j)+'.txt' , b_rec[j])
    np.savetxt(MODEPATH+'loss_value.txt', np.reshape(loss_value,[1,1]))
    print("Model saved.")     

 
