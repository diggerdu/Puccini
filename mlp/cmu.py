import os
import mxnet as mx
import numpy as np

posi_path = "../posi"
nega_path = "../nega"

# First, the symbol needs to be defined
data = mx.sym.Variable("data") # input features, mxnet commonly calls this 'data'
mlabel = mx.sym.Variable("softmax_label")

# One can either manually specify all the inputs to ops (data, weight and bias)
w1 = mx.sym.Variable("weight1")
b1 = mx.sym.Variable("bias1")
l1 = mx.sym.FullyConnected(data=data, num_hidden=64, name="layer1", weight=w1, bias=b1)
a1 = mx.sym.Activation(data=l1, act_type="sigmoid", name="act1")

# Or let MXNet automatically create the needed arguments to ops
l2 = mx.sym.FullyConnected(data=a1, num_hidden=64, name="layer2")
a2 = mx.sym.Activation(data=l2, act_type="sigmoid", name="act2")
l3 = mx.sym.FullyConnected(data=a2, num_hidden=1, name="layer3")

# Create some loss symbol
cost_classification = mx.sym.LogisticRegressionOutput(data=l3, label=mlabel)

# Using skdata to get mnist data. This is for portability. Can sub in any data loading you like.


posi_data = np.zeros((1,))
for file in os.listdir(posi_path):
	tmp = np.load(posi_path + '/' + file)
	if posi_data.shape[0] == 1:
		posi_data= tmp
	else:
		posi_data = np.vstack((posi_data, tmp))

nega_data = np.zeros((1,))
for file in os.listdir(nega_path):
	tmp = np.load(nega_path + '/' + file)
	if nega_data.shape[0] == 1:
		nega_data = tmp
	else:
		nega_data = np.vstack((nega_data, tmp))

#print posi_data
#print nega_data

train_data = np.vstack((posi_data, nega_data)) / 10.
label = np.hstack((np.ones(posi_data.shape[0]), np.zeros(nega_data.shape[0])))
batch_size = 1024
idx = [i for i in range(train_data.shape[0])]


# Bind an executor of a given batch size to do forward pass and get gradients
input_shapes = {"data": (batch_size, 53), "softmax_label": (batch_size, )}
executor = cost_classification.simple_bind(ctx=mx.gpu(0),
                                           grad_req='write',
                                           **input_shapes)
# The above executor computes gradients. When evaluating test data we don't need this.
# We want this executor to share weights with the above one, so we will use bind
# (instead of simple_bind) and use the other executor's arguments.
executor_test = cost_classification.bind(ctx=mx.gpu(0),
                                         grad_req='null',
                                         args=executor.arg_arrays)

# initialize the weights
for r in executor.arg_arrays:
    r[:] = np.random.randn(*r.shape)*0.2
#    r[:] = np.zeros(r.shape)
#    print np.zeros(r.shape)


best = 0
for epoch in range(10000):
  print "Starting epoch", epoch
  np.random.shuffle(idx)

  for x in range(0, len(idx), batch_size):
    # extract a batch from mnist
    batchX = train_data[idx[x:x+batch_size]]
    batchY = label[idx[x:x+batch_size]]
    # our executor was bound to 128 size. Throw out non matching batches.
    if batchX.shape[0] != batch_size:
        continue
    # Store batch in executor 'data'
    executor.arg_dict['data'][:] = batchX
    # Store label's in 'softmax_label'
    executor.arg_dict['softmax_label'][:] = batchY
    executor.forward()
    executor.backward()

    # do weight updates in imperative
    for pname, W, G in zip(cost_classification.list_arguments(), executor.arg_arrays, executor.grad_arrays):
        # Don't update inputs
        # MXNet makes no distinction between weights and data.
        if pname in ['data', 'softmax_label']:
            continue
        # what ever fancy update to modify the parameters
        W[:] = W - G * 0.0001
        #print G.asnumpy()

  # Evaluation at each epoch
  num_correct = 0
  num_total = 0
  threshold = 0.5
  pred_all = np.array([])
  real_all = np.array([])
  for x in range(0, len(idx), batch_size):
    batchX = train_data[idx[x:x+batch_size]]
    batchY = label[idx[x:x+batch_size]]
    if batchX.shape[0] != batch_size:
        continue
    #use the test executor as we don't care about gradients
    executor_test.arg_dict['data'][:] = batchX
    executor_test.forward()
    pred = executor_test.outputs[0].asnumpy().reshape(batchY.shape)
    if best > 0.999:
        pred_all = np.hstack((pred_all, pred))
        real_all = np.hstack((real_all, batchY))
    for i in range(pred.shape[0]):
        if pred[i] > threshold:
            pred[i] = 1
        else:
            pred[i] = 0
    num_correct += sum(batchY == pred)
    num_total += len(batchY)
  if best > 0.999:
    print "ok"
    np.save("real", real_all)
    np.save("pred", pred_all)
    break
	
  print "Accuracy thus far", num_correct * 1.0 / num_total 
  if num_correct * 1.0 / num_total > best:
     best = num_correct * 1.0 / num_total
  print "Best thus far", best

