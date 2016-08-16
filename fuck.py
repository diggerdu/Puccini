import mxnet as mx
import numpy as np

posi_path = "posi/data0.npy"
nega_path = "nega/data0.npy"

# First, the symbol needs to be defined
data = mx.sym.Variable("data") # input features, mxnet commonly calls this 'data'
label = mx.sym.Variable("softmax_label")

# One can either manually specify all the inputs to ops (data, weight and bias)
w1 = mx.sym.Variable("weight1")
b1 = mx.sym.Variable("bias1")
l1 = mx.sym.FullyConnected(data=data, num_hidden=64, name="layer1", weight=w1, bias=b1)
a1 = mx.sym.Activation(data=l1, act_type="sigmoid", name="act1")

# Or let MXNet automatically create the needed arguments to ops
l2 = mx.sym.FullyConnected(data=a1, num_hidden=64, name="layer2")

# Create some loss symbol
cost_classification = mx.sym.SoftmaxOutput(data=l2, label=label)

# Using skdata to get mnist data. This is for portability. Can sub in any data loading you like.

posi_data = np.load(posi_path)
nega_data = np.load(nega_path)
print posi_data[0]
#print posi_data
n
#print nega_data

train_data = np.vstack((posi_data, nega_data)) / 10.
label = np.hstack((np.ones(posi_data.shape[0])*2., np.ones(nega_data.shape[0])))
batch_size = 1024
idx = [i for i in range(train_data.shape[0])]


# Bind an executor of a given batch size to do forward pass and get gradients
input_shapes = {"data": (batch_size, 53), "softmax_label": (batch_size, )}
executor = cost_classification.simple_bind(ctx=mx.cpu(),
                                           grad_req='write',
                                           **input_shapes)
# The above executor computes gradients. When evaluating test data we don't need this.
# We want this executor to share weights with the above one, so we will use bind
# (instead of simple_bind) and use the other executor's arguments.
executor_test = cost_classification.bind(ctx=mx.cpu(),
                                         grad_req='null',
                                         args=executor.arg_arrays)

# initialize the weights
for r in executor.arg_arrays:
    r[:] = np.random.randn(*r.shape)*0.2
#    r[:] = np.zeros(r.shape)
#    print np.zeros(r.shape)


learning_rate = 0.0002
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
        W[:] = W - G * learning_rate

  # Evaluation at each epoch
  num_correct = 0
  num_total = 0
  for x in range(0, len(idx), batch_size):
    batchX = train_data[idx[x:x+batch_size]]
    batchY = label[idx[x:x+batch_size]]
    if batchX.shape[0] != batch_size:
        continue
    # use the test executor as we don't care about gradients
    executor_test.arg_dict['data'][:] = batchX
    executor_test.forward()
    #print executor_test.outputs[0].asnumpy()
    num_correct += sum(batchY == np.argmax(executor_test.outputs[0].asnumpy(), axis=1))
    num_total += len(batchY)
  print "Accuracy thus far", num_correct / float(num_total)
  if num_correct / float(num_total) > 87:
	learning_rate = 0.00001
