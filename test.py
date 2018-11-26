import tensorflow as tf
import numpy as np
import textcnn

voca_size = 10
embedding_mode='rand'
embedding_size = 4
window_size = [3,4,5]
filters=2
num_classes = 2
pad_idx = 0
lr = 0.01
idx = np.array([[1,2,3,4,5], [1,2,3,0,0], [2,3,0,0,0]])
target = np.array([[1,0], [1,0], [0,1]])

sess = tf.Session()
model = textcnn.TextCNN(
		sess=sess, 
		voca_size=voca_size, 
		embedding_mode=embedding_mode, 
		embedding_size=embedding_size, 
		window_size=window_size, 
		filters=filters,
		num_classes=num_classes, 
		pad_idx=pad_idx, 
		lr=lr
	)

print(sess.run(model.embedding, {model.idx_input:idx}))

zz = model.convolved_features
#print(sess.run(zz, {model.idx_input:idx}))
print(zz[0])
print(zz[1])
print(zz[2])
print(sess.run(zz[0], {model.idx_input:idx}))

aa0 = sess.run(zz[0], {model.idx_input:idx})
aa1 = sess.run(zz[1], {model.idx_input:idx})
aa2 = sess.run(zz[2], {model.idx_input:idx})

print(aa0.shape, aa1.shape, aa2.shape)

qq0 = sess.run(model.pooled_features[0], {model.idx_input:idx})
qq1 = sess.run(model.pooled_features[1], {model.idx_input:idx})
qq2 = sess.run(model.pooled_features[2], {model.idx_input:idx})

print(qq0.shape, qq1.shape, qq2.shape)
#print(aa0,'\n', qq0, '\n\n')
#print(aa1,'\n', qq1, '\n\n')
#print(aa2,'\n', qq2, '\n\n')

print(qq0,'\n')
print(qq1,'\n')
print(qq2,'\n')

xx = sess.run(model.concat_and_flatten_features, {model.idx_input:idx})
print(xx, xx.shape)

pred = sess.run(model.pred, {model.idx_input:idx, model.keep_prob:1})
print(pred, pred.shape)

cost = sess.run(model.train_cost, {model.idx_input:idx, model.target:target, model.keep_prob:1})
print(cost)


for i in range(1,100):
	cost , _, pred= sess.run([model.train_cost, model.minimize, tf.nn.softmax(model.pred, dim=1)], {model.idx_input:idx, model.target:target, model.keep_prob:1})
	print(i, cost, pred)
#pred = sess.run(tf.nn.softmax(model.pred, dim=1), {model.idx_input:idx, model.keep_prob:1})
#print(pred)