import deepchem as dc
import tensorflow as tf 

_, (train, valid, test), _ = dc.molnet.load_tox21()
train_X, train_y, train_w = train.X, train.y, train.w
valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
test_X, test_y, test_w = test.X, test.y, test.w

# Remove extra tasks
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

"""     When dealing with minibatched data, it is often convenient to be able to feed
batches of variable size. Suppose that a dataset has 947 elements. Then with a mini‐
batch size of 50, the last batch will have 47 elements. This would cause the code to crash.
 using None
as a dimensional argument to a placeholder allows the placeholder to accept tensors
with arbitrary size in that dimension """

d = 1024
with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (None, d))
    y = tf.placeholder(tf.float32, (None,))


with tf.name_scope("hidden-layer"):
    W = tf.Variable(tf.random_normal((d, n_hidden)))
    b = tf.Variable(tf.random_normal((n_hidden,)))
    x_hidden = tf.nn.relu(tf.matmul(x, W) + b)

with tf.name_scope("placeholders"):
    x = tf.placeholder(tf.float32, (None, d))
    y = tf.placeholder(tf.float32, (None,))
with tf.name_scope("hidden-layer"):
    W = tf.Variable(tf.random_normal((d, n_hidden)))
    b = tf.Variable(tf.random_normal((n_hidden,)))
    x_hidden = tf.nn.relu(tf.matmul(x, W) + b)
    # Apply dropout
    x_hidden = tf.nn.dropout(x_hidden, keep_prob)

"""     keep_prob is the probability that any given node is kept.
    we want to turn on dropout when training and turn off dropout when making predictions
    a new placeholder for keep_prob is introduced to handle this. 
    During training, we pass in the desired value, often 0.5, but at test time we set
    keep_prob to 1.0 since we want predictions made with all learned nodes"""

with tf.name_scope("output"):
    W = tf.Variable(tf.random_normal((n_hidden, 1)))
    b = tf.Variable(tf.random_normal((1,)))
    y_logit = tf.matmul(x_hidden, W) + b
    # the sigmoid gives the class probability of 1
    y_one_prob = tf.sigmoid(y_logit)
    # Rounding P(y=1) will give the correct prediction.
    y_pred = tf.round(y_one_prob)
with tf.name_scope("loss"):
 # Compute the cross-entropy term for each datapoint
    y_expand = tf.expand_dims(y, 1)
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
 # Sum all contributions
    l = tf.reduce_sum(entropy)
with tf.name_scope("optim"):
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(l)
with tf.name_scope("summaries"):
    tf.summary.scalar("loss", l)
    merged = tf.summary.merge_all()

keep_prob = tf.placeholder(tf.float32)

# Implementing Mini Bactching 

step = 0
for epoch in range(n_epochs):
    pos = 0
    while pos < N:
    batch_X = train_X[pos:pos+batch_size]
    batch_y = train_y[pos:pos+batch_size]
    feed_dict = {x: batch_X, y: batch_y, keep_prob: dropout_prob}
    _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
    print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
    train_writer.add_summary(summary, step)
    step += 1
    pos += batch_size

# Evaluating Model Accuracy
"""     the fact that the data is imbalanced makes this tricky. The classification accuracy metric we
    used in the previous chapter simply measures the fraction of datapoints that were labeled correctly. 
    However, 95% of data in our dataset is labeled 0 and only 5% are labeled 1. As a result the all-0
     model (which labels everything negative) would achieve 95% accuracy! 
     
     One solution is to increase the weights of positive examples so that they count for more
     
     we use the function accuracy_score(true,
        pred, sample_weight=given_sample_weight) from sklearn.metrics. This func‐
        tion has a keyword argument sample_weight, which lets us specify the desired weight
        for each datapoint. We use this function to compute the weighted metric on both the
        training and validation sets"""

train_weighted_score = accuracy_score(train_y, train_y_pred, sample_weight=train_w)
print("Train Weighted Classification Accuracy: %f" % train_weighted_score)
valid_weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
print("Valid Weighted Classification Accuracy: %f" % valid_weighted_score)
