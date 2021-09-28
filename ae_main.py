from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import heatmap_seqmultiiter
import csv
kernel = 10
num = 31
def write_csv(lines,filename):
    with open(filename,'w') as f:
        writer = csv.writer(f,lineterminator='\n')
        writer.writerows(lines)
print("normal")

def testoutput(file_input,file_output,heatmap_pattern = 2):
    nes = []
    ne1 = [] 
    ne2 = []
    ne3 = []

    ht1 = heatmap_seqmultiiter.seqiter(file_input,dir_or_file = "file",pattern = heatmap_pattern,binary=True,flatten=False,stride_size = 5,kernel_size = 10,reset = False)
    for i in range(10000):
        try:
            tmp = ht1.next()
        except:
            break
        tmp = h.next()
        tmp = np.squeeze(np.asarray([tmp],np.float32))
        batch_x1 = tmp[:,[8,9,10,17]]
        batch_x2 = tmp[:,[0,1,2,3,4,5,6,7]]
        batch_x3 = tmp[:,[11,12,13,14,15,16]]
        batch_x1 = batch_x1.flatten()[np.newaxis,:]
        batch_x2 = batch_x2.flatten()[np.newaxis,:]
        batch_x3 = batch_x3.flatten()[np.newaxis,:]
  
        data = es.eval({x1: batch_x1,x2: batch_x2,x3:batch_x3})
        print('%6.3f %6.3f' % (data[0][0],data[0][1]))
        nes.append((data[0][0],data[0][1]))
        data = e1.eval({x1: batch_x1,x2: batch_x2,x3:batch_x3})
        ne1.append((data[0][0],data[0][1]))
        data = e2.eval({x1: batch_x1,x2: batch_x2,x3:batch_x3})
        ne2.append((data[0][0],data[0][1]))
        data = e3.eval({x1: batch_x1,x2: batch_x2,x3:batch_x3})
        ne3.append((data[0][0],data[0][1]))
    nes_file = file_output  + "_nes.csv"
    ne1_file = file_output  + "_e1.csv"
    ne2_file = file_output  + "_e2.csv"
    ne3_file = file_output  + "_e3.csv"
    write_csv(nes,nes_file)
    write_csv(ne1,ne1_file)
    write_csv(ne2,ne2_file)
    write_csv(ne3,ne3_file)


def reg_pattern(pattern = 'heart',kernel=10):
    if pattern == 'heart':
        #t1,t2 = np.random.uniform(0.8, 1.0),np.random.uniform(0.0, 0.2)
        embedding_data = np.array([[1.0,0.0]])
        batch_x1 = np.zeros((1,40))
        batch_x2 = np.ones((1,80))
        #batch_x1 = np.random.uniform(0.0, 0.2, (1, 40))
        #batch_x2 = np.random.uniform(0.8, 1.0, (1, 80))
        batch_x3 = np.random.random_sample((1,60))
        input_data = (batch_x1,batch_x2,batch_x3)
    if pattern == 'vessel':
#        t1,t2 = np.random.uniform(0.0, 0.2),np.random.uniform(0.8, 1.0)
        embedding_data = np.array([[0.0,1.0]])
        batch_x1 = np.ones((1,40))
        batch_x2 = np.zeros((1,80))
        #batch_x1 = np.random.uniform(0.8, 1.0, (1, 40))
        #batch_x2 = np.random.uniform(0.0, 0.2, (1, 80))
        batch_x3 = np.random.random_sample((1,60))
        input_data = (batch_x1,batch_x2,batch_x3)
    if pattern == 'blank':
        t1,t2 = np.random.uniform(0.0, 0.2),np.random.uniform(0.0, 0.2)
        embedding_data = np.array([[t1,t2]])
        #batch_x1 = np.ones((1,40))
        #batch_x2 = np.zeros((1,80))
        batch_x1 = np.random.uniform(0.0, 0.2, (1, 40))
        batch_x2 = np.random.uniform(0.0, 0.2, (1, 80))
        batch_x3 = np.random.random_sample((1,60))
        input_data = (batch_x1,batch_x2,batch_x3)
    return input_data,embedding_data 

v = 4*kernel
cal = 8*kernel
other = 6*kernel

# Variables
x1 = tf.placeholder("float", [None,v])
x2 = tf.placeholder("float", [None,cal])
x3 = tf.placeholder("float", [None,other])
reg123 = tf.placeholder("float",[None,2])

x123 = tf.concat([x1,x2,x3],axis = 1)

w_enc_e1 = tf.Variable(tf.random_normal([v, 2], mean=0.0, stddev=0.05))
w_enc_e2 = tf.Variable(tf.random_normal([cal, 2], mean=0.0, stddev=0.05))
w_enc_e3 =  tf.Variable(tf.random_normal([other, 2], mean=0.0, stddev=0.05))

w_dec_d1 = tf.Variable(tf.random_normal([2,v], mean=0.0, stddev=0.05))
w_dec_d2 = tf.Variable(tf.random_normal([2,cal], mean=0.0, stddev=0.05))
w_dec_d3 = tf.Variable(tf.random_normal([2,other], mean=0.0, stddev=0.05))
w_dec_ds =  tf.Variable(tf.random_normal([2, v + cal + other], mean=0.0, stddev=0.05))
b_dec_ds =  tf.Variable(tf.random_normal([v + cal + other], mean=0.0, stddev=0.05))

b_dec_d1 = tf.Variable(tf.zeros([v]))
b_dec_d2 = tf.Variable(tf.zeros([cal]))
b_dec_d3 = tf.Variable(tf.zeros([other]))

b_enc_e1 = tf.Variable(tf.zeros([2]))
b_enc_e2 = tf.Variable(tf.zeros([2]))
b_enc_e3 = tf.Variable(tf.zeros([2]))

w_enc_es =  tf.Variable(tf.random_normal([6, 2], mean=0.0, stddev=0.05))
b_enc_es =  tf.Variable(tf.random_normal([2], mean=0.0, stddev=0.05))


# Create the model
def model(X, w_e, b_e, w_d, b_d):
    encoded = tf.sigmoid(tf.matmul(X, w_e) + b_e)
    decoded = tf.sigmoid(tf.matmul(encoded, w_d) + b_d)

    return encoded, decoded

e1, d1 = model(x1, w_enc_e1, b_enc_e1, w_dec_d1, b_dec_d1)
e2, d2 = model(x2, w_enc_e2, b_enc_e2, w_dec_d2, b_dec_d2)
e3, d3 = model(x3, w_enc_e3, b_enc_e3, w_dec_d3, b_dec_d3)

es_inp = tf.concat([e1,e2,e3],axis = 1)
es,ds = model(es_inp,w_enc_es,b_enc_es,w_dec_ds,b_dec_ds)


 

# Cost Function basic term
cross_entropy1 = -1. * x1 * tf.log(d1) - (1. - x1) * tf.log(1. - d1)
cross_entropy2 = -1. * x2 * tf.log(d2) - (1. - x2) * tf.log(1. - d2)
# cross_entropy3 = -1. * x3 * tf.log(d3) - (1. - x3) * tf.log(1. - d3)
cross_entropyds =  -1. * x123 * tf.log(ds) - (1. - x123) * tf.log(1. - ds)

# reg term
# lossreg =-1. * reg123 * tf.log(es) - (1. - reg123) * tf.log(1. - es)               
lossreg = tf.square(es - reg123)

loss1 = tf.reduce_mean(cross_entropy1)
loss2 = tf.reduce_mean(cross_entropy2)
# loss3 = tf.reduce_mean(cross_entropy3) 
lossds = tf.reduce_mean(cross_entropyds)


loss = loss1 + loss2  + lossds
train_step = tf.train.AdagradOptimizer(0.7).minimize(loss)

loss_reg_step = loss1 + loss2  + lossds + lossreg
reg_train_step = tf.train.AdagradOptimizer(0.1).minimize(loss_reg_step)

# Train
init = tf.initialize_all_variables()
heatmap_pattern = 2
dirpath = "path_to_training"
testdirpath1 = "path_to_test_normal"
testdirpath2 = "path_to_test_chd"
outpath = "output_path"
h = heatmap_seqmultiiter.seqiter(dirpath,pattern = heatmap_pattern,binary=True,flatten=False,stride_size = 5,kernel_size = 10,reset = True)
sess = tf.InteractiveSession()
#num = 11
tf.set_random_seed(num)
sess.run(init)

print('Training...')
for i in range(100000):
    tmp = h.next()
    tmp = np.squeeze(np.asarray([tmp],np.float32))
    batch_x1 = tmp[:,[8,9,10,17]]
    batch_x2 = tmp[:,[0,1,2,3,4,5,6,7]]
    batch_x3 = tmp[:,[11,12,13,14,15,16]]
    batch_x1 = batch_x1.flatten()[np.newaxis,:]
    batch_x2 = batch_x2.flatten()[np.newaxis,:]
    batch_x3 = batch_x3.flatten()[np.newaxis,:]
    train_step.run({x1: batch_x1,x2: batch_x2,x3:batch_x3})
    if i % 10 == 0:
        id,em = reg_pattern(pattern = 'heart',kernel = 10)
        reg_train_step.run({x1: id[0],x2: id[1],x3:batch_x3,reg123:em} )
        id,em = reg_pattern(pattern = 'vessel',kernel = 10)
        reg_train_step.run({x1: id[0],x2: id[1],x3:batch_x3,reg123:em} )
#        id,em = reg_pattern(pattern = 'blank',kernel = 10)
#        reg_train_step.run({x1: id[0],x2: id[1],x3:batch_x3,reg123:em} )
    if i % 1000 == 0:
        train_loss = loss.eval({x1: batch_x1,x2: batch_x2,x3:batch_x3})
        print('  step, loss = %6d: %6.3f' % (i, train_loss))
normal_list = "list_of_normal"
chd_list = "list_of_chd"
print("normal")
for input in normal_list:
    testdirpath1 = "path_to_test_normal" + str(input) + ".csv"
    testoutput1 = "path_to_output" + str(input)
    print(testdirpath1)
    testoutput(testdirpath1,testoutput1)

print("chd")
for input in chd_list:
    testdirpath2 = "path_to_test_chd" + str(input) + ".csv"
    testoutput2 = "path_to_output" + str(input)
    print(testdirpath2)
    testoutput(testdirpath2,testoutput2)




