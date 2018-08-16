import tensorflow as tf

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
    def __init__(self, params, n_classes, input_size, learning_rate=0.001, keep_prob=0.5):
        self.n_classes = n_classes
        # pad up to make sure everything will be contained
        self.input_size = input_size

        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        for p in params:
            # if p["kernel_size"] is None:
            #     p["kernel_size"] = calculate_fc_kernel(self.input_size, params)
            if "logit" in p["name"] and p["filters"] is None:
                p["filters"] = self.n_classes
        self.params = params

    def _parse_param(self, p, inputs):
        # someday this will be awesome
        op = p["name"].split("_")[0]
        if op == "conv":
            return self._add_conv_relu_layer(p, inputs)
        elif op == "maxpool":
            return self._add_maxpool(p, inputs)
        elif op == "logit":
            return self._add_conv_layer(p, inputs)

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
        return tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits,
                labels=targets
            )
        )

    def _build_optimizer(self, loss, learning_rate):
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.minimize(loss)
        return optimizer

    def _build(self, inputs, targets):
        if inputs is None and targets is None:
            self._standalone_placeholders(self.input_size, self.n_classes)
        else:
            self.inputs, self.targets = inputs, targets
        # for p in self.params:
        #     self._parse_param(p)
        a = [tf.expand_dims(self.inputs, -1)]
        for i, layer_params in enumerate(self.params):
            a.append(self._parse_param(layer_params, a[i]))

        logits = tf.squeeze(a[-1])
        self.out = tf.nn.softmax(logits, name="predictions")

        self.loss = self._build_loss(logits, self.targets)

        self.optimizer = self._build_optimizer(self.loss, self.learning_rate)

    def build(self, inputs=None, targets=None):
        self._build(inputs, targets)

    def train(self):
        pass

    def eval(self, sess, x, y):
        pass

    def predict(self):
        pass



