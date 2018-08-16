import tensorflow as tf
import os
import logging
import json
"""
Designed to work with either Dataset api or feed-dict "standalone".

tf.reset_default_graph()

epochs = 1
batch_size = 32
in_placeholder = tf.placeholder(tf.float32, in_data.shape)
labels_placeholder = tf.placeholder(tf.float32, oh_labels.shape)
dataset = (tf.data.Dataset
           .from_tensor_slices((in_placeholder, labels_placeholder))
           .repeat()
           .shuffle(100)
           .batch(batch_size))

iterator = dataset.make_initializable_iterator()
inputs, targets = iterator.get_next()
iterator_init_op = iterator.initializer

test.build(inputs, targets)


with tf.Session() as sess:
    sess.run(iterator.initializer, 
             feed_dict = {
            in_placeholder: in_data,
            labels_placeholder: oh_labels
        })    
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        _, loss_val, out = sess.run([test.optimizer, test.loss, test.out])
"""


# fixed architecture net
class fMRIConvNet(object):
    def __init__(self, params, n_classes, input_size, l2=True, name="cnn-model", save_path="./model", learning_rate=0.001):
        self.name = name
        self.save_path = save_path
        self.n_classes = n_classes
        # pad up to make sure everything will be contained
        self.input_size = input_size
        self.l2 = l2
        self.learning_rate = learning_rate
        self.params = params.copy()
        self.is_built = False
        for p in self.params:
            # if p["kernel_size"] is None:
            #     p["kernel_size"] = calculate_fc_kernel(self.input_size, params)
            if "logit" in p["name"] and p["filters"] is None:
                p["filters"] = self.n_classes

    def _parse_param(self, p, inputs):
        # someday this will be awesome
        op = p["name"].split("_")[0]
        if op == "conv":
            return self._add_conv_relu_layer(p, inputs)
        elif op == "maxpool":
            return self._add_maxpool(p, inputs)
        elif op == "logit":
            return self._add_conv_layer(p, inputs)
        elif op == "drop":
            return self._add_dropout(p, inputs)

    def _add_dropout(self, layer_params, inputs):
        return tf.layers.dropout(
            inputs=inputs,
            rate=layer_params["rate"],
            name=layer_params["name"]
        )

    def _standalone_placeholders(self, input_size, n_classes):
        input_shape = tuple([None] + list(input_size))
        self.inputs = tf.placeholder(tf.float32, input_shape, name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, n_classes], name="targets")

    def _add_conv_relu_layer(self, layer_params, inputs):
        if layer_params["kernel_size"] is None:
            layer_params["kernel_size"] = inputs.get_shape().as_list()[1:4]
        return tf.layers.conv3d(
            inputs=inputs,
            filters=layer_params["filters"],
            kernel_size=layer_params["kernel_size"],
            padding=layer_params["padding"],
            name=layer_params["name"],
            activation=tf.nn.relu
        )

    def _add_conv_layer(self, layer_params, inputs):
        return tf.layers.conv3d(
            inputs=inputs,
            filters=layer_params["filters"],
            kernel_size=layer_params["kernel_size"],
            padding=layer_params["padding"],
            name=layer_params["name"],
            activation=None
        )

    def _add_maxpool(self, layer_params, inputs):
        return tf.layers.max_pooling3d(
            inputs=inputs,
            pool_size=layer_params["pool_size"],
            name=layer_params["name"],
            strides=layer_params["strides"],
            padding=layer_params["padding"]
        )

    def _build_loss(self, logits, targets):
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits,
                labels=targets
            )
        )
        return loss

    def _compute_accuracy(self, pred, label):
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(label, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _build_optimizer(self, loss, learning_rate):
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.minimize(loss)
        return optimizer

    def _build(self, inputs, targets):
        if self.is_built is False:
            if inputs is None and targets is None:
                self._standalone_placeholders(self.input_size, self.n_classes)
            else:
                self.inputs, self.targets = inputs, targets

            # if train_mode
            # for p in self.params:
            #     self._parse_param(p)
            a = [tf.expand_dims(self.inputs, -1)]
            for i, layer_params in enumerate(self.params):
                a.append(self._parse_param(layer_params, a[i]))

            logits = tf.squeeze(a[-1])
            self.out = tf.nn.softmax(logits, name="predictions")
            self.loss = self._build_loss(logits, self.targets)
            self.accuracy = self._compute_accuracy(self.out, self.targets)
            if self.l2:
                self.loss += tf.losses.get_regularization_loss()
            self.optimizer = self._build_optimizer(self.loss, self.learning_rate)
            self.is_built = True
        else:
            pass

    def build(self, inputs=None, targets=None, rebuild=False):
        if rebuild:
            self.is_built = False
        self._build(inputs, targets)

    def save(self, sess, saver, **params):
        saver.save(sess, os.path.join(self.save_path, self.name), **params)
        with open(os.path.join(self.save_path, "{}.json".format(self.name)), "w") as out:
            json.dump(self.params, out)

    def load(self, sess, saver):
        restore_from = tf.train.latest_checkpoint(self.save_path)
        if restore_from is not None:
            saver.restore(sess, restore_from)
        else:
            pass

    def train(self, sess, x, y):
        pass

    def eval(self, sess, x, y):
        return sess.run(self.accuracy)

    def predict(self, sess, x, y):
        return sess.run(self.out)


