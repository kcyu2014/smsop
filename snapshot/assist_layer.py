from keras.regularizers import Regularizer
import keras.backend as K

import numpy as np
import tensorflow as tf

from keras import backend as K, initializers as initializations
from keras.engine import Layer, InputSpec
from keras.layers import BatchNormalization, RepeatVector
from keras.layers.merge import _Merge


def block_diagonal(matrices, dtype=tf.float32):
    """Constructs block-diagonal matrices from a list of batched 2D tensors.

    Args:
        matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
          matrices with the same batch dimension).
        dtype: Data type to use. The Tensors in `matrices` must match this dtype.
      Returns:
        A matrix with the input matrices stacked along its main diagonal, having
        shape [..., \sum_i N_i, \sum_i M_i].

      """
    matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
    blocked_rows = tf.Dimension(0)
    blocked_cols = tf.Dimension(0)
    batch_shape = tf.TensorShape(None)
    for matrix in matrices:
        full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
        batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
        blocked_rows += full_matrix_shape[-2]
        blocked_cols += full_matrix_shape[-1]
    ret_columns_list = []
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        ret_columns_list.append(matrix_shape[-1])
    ret_columns = tf.add_n(ret_columns_list)
    row_blocks = []
    current_column = 0
    for matrix in matrices:
        matrix_shape = tf.shape(matrix)
        row_before_length = current_column
        current_column += matrix_shape[-1]
        row_after_length = ret_columns - current_column
        row_blocks.append(tf.pad(
            tensor=matrix,
            paddings=tf.concat(
              [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
               [(row_before_length, row_after_length)]],
              axis=0)))
    blocked = tf.concat(row_blocks, -2)
    blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
    return blocked


######### MATH LAYER ################


class SignedSqrt(Layer):

    def __init__(self, scale=1, **kwargs):
        super(SignedSqrt, self).__init__(**kwargs)
        self.input_spec = [InputSpec(min_ndim=2)]
        self.scale = scale

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        from src.tensorflow.ops import safe_sign_sqrt
        return safe_sign_sqrt(self.scale * inputs)


class LogLayer(Layer):

    def __init__(self, scale=1, **kwargs):
        super(LogLayer, self).__init__(**kwargs)
        self.input_spec = [InputSpec(min_ndim=2)]
        self.scale = scale

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        return K.log(K.abs(inputs))


class PowLayer(Layer):

    def __init__(self, power=1.0/3.0, scale=1.0, center=0.0, **kwargs):
        super(PowLayer, self).__init__(**kwargs)
        self.input_spec = [InputSpec(min_ndim=2)]
        self.power = power
        self.center = center
        self.scale = scale
        self.name += "pow_{}-scale_{}-center_{}".format(self.power, self.scale, self.center)

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        return K.pow(self.scale * inputs, self.power) - self.center


class L2Norm(Layer):

    def __init__(self, axis=1, **kwargs):
        super(L2Norm, self).__init__(**kwargs)
        self.axis = axis
        self.input_spec = [InputSpec(min_ndim=2)]

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs, **kwargs):
        return K.l2_normalize(inputs, self.axis)

    def get_config(self):
        """
        To serialize the model given and generate all related parameters
        Returns
        -------

        """
        config = {'axis': self.axis}
        base_config = super(L2Norm, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FlattenSymmetric(Layer):
    """
    Flatten Symmetric is a layer to flatten the previous layer with symmetric matrix.

        # Input shape
            3D tensor with (samples, input_dim, input_dim)
        # Output shape
            2D tensor with (samples, input_dim * (input_dim +1) / 2 )
            Drop the duplicated terms
        # Arguments
            name            name of the model
    """

    def __init__(self, **kwargs):
        self.input_spec = [InputSpec(ndim='3+')]
        super(FlattenSymmetric, self).__init__(**kwargs)

    def build(self, input_shape):
        # create and store the mask
        assert input_shape[1] == input_shape[2]
        self.upper_triangular_mask = tf.constant(
            np.triu(
                np.ones((input_shape[1], input_shape[2]), dtype=np.bool_),
            0),
            dtype=tf.bool
            )
        self.built = True

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise Exception('The shape of the input to "Flatten" '
                            'is not fully defined '
                            '(got ' + str(input_shape[1:]) + '. '
                            'Make sure to pass a complete "input_shape" '
                            'or "batch_input_shape" argument to the first '
                            'layer in your model.')
        assert input_shape[1] == input_shape[2]
        return input_shape[0], input_shape[1]*(input_shape[1]+1)/2

    def call(self, x, mask=None):
        fn = lambda x : tf.boolean_mask(x, self.upper_triangular_mask)
        return tf.map_fn(fn, x)


class TransposeFlattenSymmetric(Layer):
    """
    Implement the Transposed operation for FlattenSymmetric
    """
    def __init__(self, **kwargs):
        self.input_spec = [InputSpec(ndim='2+')]
        self.input_dim = None
        self.batch_size = None
        super(TransposeFlattenSymmetric, self).__init__(**kwargs)

    def build(self, input_shape):
        """ Input a batch-vector form """
        self.input_dim = input_shape[1]
        self.batch_size = input_shape[0]
        self.built = True

    def call(self, x, mask=None):
        """ Call the to symmetric matrices. """


class SeparateConvolutionFeatures(Layer):
    """
    SeparateConvolutionFeatures is a layer to separate previous convolution feature maps
    into groups equally.

        # Input shape
            ND tensor with (nb_sample, x, y, z)
        # Output shape, directly transpose
            n xD tensor with (nb_sample, x, y, z/n) for tensorflow.
        # Arguments
            n   should make z/n an integer

    """
    def __init__(self, n=-1, split_axis=3, **kwargs):
        if K.backend() == 'theano' or K.image_data_format() == 'channels_first':
            raise RuntimeError("Only support tensorflow backend or image ordering")
        self.n = n
        self.input_spec = [InputSpec(ndim='4+')]
        self.split_axis = split_axis
        self.output_dim = None
        self.out_shape = None
        self.split_loc = None
        super(SeparateConvolutionFeatures, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.n == -1:
            self.n = input_shape[self.split_axis]
        self.output_dim = input_shape[self.split_axis] / self.n
        self.out_shape = input_shape[:self.split_axis] + (self.output_dim,) + input_shape[self.split_axis+1:]
        self.split_loc = [self.output_dim * i for i in range(self.n + 1)]
        # self.split_loc.append(self.output_dim * self.n)
        self.built = True

    def compute_output_shape(self, input_shape):
        """ Return a list """
        output_shape = []
        for i in range(self.n):
            if self.output_dim > 1:
                output_shape.append(input_shape[:self.split_axis] + input_shape[self.split_axis+1:] + (self.output_dim,))
            else:
                output_shape.append(
                    input_shape[:self.split_axis] + input_shape[self.split_axis + 1:])
        return output_shape

    def call(self, inputs, mask=None):
        if self.n == -1:
            raise ValueError("Should build {} layer before calling".format(self.__class__))
        # Transpose it
        perm_axis = range(0, K.ndim(inputs))
        perm_axis = perm_axis[:self.split_axis] + perm_axis[self.split_axis + 1:] + [self.split_axis,]
        x = K.permute_dimensions(inputs, perm_axis)
        outputs = []
        for i in range(self.n):
            outputs.append(K.squeeze(x[..., self.split_loc[i]:self.split_loc[i + 1]], axis=-1))
        return outputs

    def compute_mask(self, input, input_mask=None):
        """ Override the compute mask to produce two masks """
        if input_mask is None:
            return [None for i in range(self.n)]
        else:
            raise ValueError("Not supporting mask for this layer {}".format(self.name))

    def get_config(self):
        return {'n': self.n,
                'split_axis': self.split_axis}


class MultiplySlice(Layer):
    """
    Get this multiply with input given!

    """
    def __init__(self, mask_axis=-1, feature_axis=-1, **kwargs):
        super(MultiplySlice, self).__init__(**kwargs)
        self.mask_axis = mask_axis
        self.feature_axis = feature_axis

    def build(self, input_shape):
        mask_shape, feature_shape = input_shape
        if not len(mask_shape) == len(feature_shape):
            raise ValueError("Must be same dimension")

        if not all([mask_shape[i] == feature_shape[i] for i in range(0, len(mask_shape) - 1)]):
            raise ValueError("Dim should be the same {}".format(K.int_shape(feature_shape)))
        Warning("Only support axis = -1")
        self.permute_axis = [0, len(mask_shape)] + range(1, len(mask_shape))
        self.built = True

    def compute_output_shape(self, input_shape):
        mask_shape, feature_shape = input_shape
        out_shape = feature_shape + (mask_shape[-1],)
        out_shape = [out_shape[i] for i in self.permute_axis]
        return tuple(out_shape)

    def call(self, inputs, **kwargs):
        """

        :param inputs: mask, feature
        :param kwargs:
        :return:
        """
        mask, feature = inputs
        mask = K.expand_dims(mask, axis=-2)
        feature = K.expand_dims(feature, axis=-1)
        output = mask * feature
        output = K.permute_dimensions(output, self.permute_axis)
        return output

    def get_config(self):
        return {'mask_axis': self.mask_axis,
                'feature_axis': self.feature_axis}


class Regrouping(Layer):
    """
    Regrouping layer is a layer to provide different combination of given layers.

    References : keras.layer.Merge

    into groups equally.

        # Input shape
            n 4D tensor with (nb_sample, x, y, z)
        # Output shape
            C(n,2) = n*(n-1)/2 4D tensor with (nb_sample, x, y, z/n) for tensorflow.
        # Arguments
               should make z/n an integer

    """
    def __init__(self, inputs, mode='group', concat_axis=-1,
                 output_shape=None, output_mask=None,
                 arguments=None, node_indices=None, tensor_indices=None,
                 name=None, version=1,
                 ):
        if K.backend() == 'theano' or K.image_dim_ordering() == 'th':
            raise RuntimeError("Only support tensorflow backend or image ordering")

        self.inputs = inputs
        self.mode = mode
        self.concat_axis = concat_axis
        self._output_shape = output_shape
        self.node_indices = node_indices
        self._output_mask = output_mask
        self.arguments = arguments if arguments else {}

        # Layer parameters
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self._trainable_weights = []
        self._non_trainable_weights = []
        self.supports_masking = True
        self.uses_learning_phase = False
        self.input_spec = None

        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))

        self.name = name

        if inputs:
            # The inputs is a bunch of nodes shares the same input.
            if not node_indices:
                node_indices = [0 for _ in range(len(inputs))]
            self.built = True
            # self.add_inbound_node(inputs, node_indices, tensor_indices)
        else:
            self.built = False

    # def build(self, input_shape):
    #     self.output_dim = input_shape[3] / self.n
    #     self.out_shape = input_shape[:2] + (self.output_dim,)
    #     self.split_loc = [self.output_dim * i for i in range(self.n)]
    #     self.split_loc.append(self.output_dim * self.n)
    #     self.built = True

    def call(self, inputs, mask=None):
        import tensorflow as tf
        if not isinstance(inputs, list) or len(inputs) <= 1:
            raise TypeError("Regrouping must be taking more than one "
                            "tensor, Got: "+ str(inputs))
        # Case: 'mode' is a lambda function or function
        if callable(self.mode):
            arguments = self.arguments
            import inspect
            arg_spec = inspect.getargspec(self.mode)
            if 'mask' in arg_spec.args:
                arguments['mask'] = mask
            return self.mode(inputs, **arguments)

        if self.mode == 'group':

            outputs = []
            n_inputs = len(inputs)
            for i in range(n_inputs - 1):
                for j in range(i + 1, n_inputs):
                    with tf.device('/gpu:0'):
                        outputs.append(K.concatenate([tf.identity(inputs[i]),tf.identity(inputs[j])], self.concat_axis))
            # for i in range(0, n_inputs - 1, 2):
            #     with tf.device('/gpu:0'):
            #         conc = K.concatenate([tf.identity(inputs[i]), tf.identity(inputs[i+1])])
            #     outputs.append(conc)
            return outputs
        else:
            raise RuntimeError("Mode not recognized {}".format(self.mode))

    def compute_mask(self, input, input_mask=None):
        """ Override the compute mask to produce two masks """
        n_inputs = len(input)
        if input_mask is None or all([m is None for m in input_mask]):
            # return [None for _ in range(0, n_inputs - 1, 2)]
            return [None for _ in range(n_inputs * (n_inputs - 1) / 2)]
        else:
            raise ValueError("Not supporting mask for this layer {}".format(self.name))

    def compute_output_shape(self, input_shape):
        """ Return a list """
        assert isinstance(input_shape, list)

        output_shape = []
        n_inputs = len(input_shape)
        for i in range(0, n_inputs - 1):
            for j in range(i, n_inputs - 1):
                tmp_shape = list(input_shape[i])
                tmp_shape[self.concat_axis] += input_shape[j][self.concat_axis]
                output_shape.append(tmp_shape)
            # tmp_shape = list(input_shape[i])
            # tmp_shape[self.concat_axis] += input_shape[i+1][self.concat_axis]
            # output_shape.append(tmp_shape)
        return output_shape


class MatrixConcat(Layer):
    """
        Regrouping layer is a layer to provide different combination of given layers.

        References : keras.layer.Merge

        into groups equally.

            # Input shape
                n 4D tensor with (nb_sample, x, y, z)
            # Output shape
                C(n,2) = n*(n-1)/2 4D tensor with (nb_sample, x, y, z/n) for tensorflow.
            # Arguments
                   should make z/n an integer

        """

    def __init__(self, inputs, name=None):
        if K.backend() == 'theano' or K.image_dim_ordering() == 'th':
            raise RuntimeError("Only support tensorflow backend or image ordering")

        self.inputs = inputs

        # Layer parameters
        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self._trainable_weights = []
        self._non_trainable_weights = []
        self.supports_masking = True
        self.uses_learning_phase = False
        self.input_spec = None
        self.trainable = False

        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))

        self.name = name

        if inputs:
            # The inputs is a bunch of nodes shares the same input.
            self.built = True
            # self.add_inbound_node(inputs, node_indices, tensor_indices)
        else:
            self.built = False

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert (len(input_shape[0]) == 3)

        self.output_dim = input_shape[3] / self.n
        self.out_shape = input_shape[:2] + (self.output_dim,)
        self.built = True

    def call(self, inputs, mask=None):
        if not isinstance(inputs, list) or len(inputs) <= 1:
            raise TypeError("Regrouping must be taking more than one "
                            "tensor, Got: " + str(inputs))
        out = block_diagonal(inputs, K.floatx())
        return out

    def compute_mask(self, input, input_mask=None):
        """ Override the compute mask to produce two masks """
        if input_mask is None or all([m is None for m in input_mask]):
            # return [None for _ in range(0, n_inputs - 1, 2)]
            return None
        else:
            raise ValueError("Not supporting mask for this layer {}".format(self.name))

    def compute_output_shape(self, input_shape):
        """ Return a list """
        assert isinstance(input_shape, list)
        assert len(input_shape[0]) == 3
        output_shape = list(input_shape[0])
        for i in range(1, len(input_shape)):
            output_shape[1] += input_shape[i][1]
            output_shape[2] += input_shape[i][2]
        return [tuple(output_shape), ]

    def get_config(self):
        """
        To serialize the model given and generate all related parameters
        Returns
        -------

        """
        config = {
                  }
        base_config = super(MatrixConcat, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MatrixReLU(Layer):
    """
    MatrixReLU layer supports the input of a 3D tensor, output a corresponding 3D tensor in
        Matrix diagnal ReLU case

    It implement the Matrix ReLU with a small shift (epsilon)

        # Input shape
            3D tensor with (samples, input_dim, input_dim)
        # Output shape
            3D tensor with (samples, input_dim, input_dim)
        # Arguments
            epsilon

    """

    def __init__(self, epsilon=0, **kwargs):
        self.input_spec = [InputSpec(ndim='3+')]
        self.eps = epsilon
        self.out_dim = None
        # self.b = None
        super(MatrixReLU, self).__init__(**kwargs)

    def build(self, input_shape):
        """
                Build the model based on input shape
                Should not set the weight vector here.
                :param input_shape: (nb-sample, input_dim, input_dim)
                :return:
                """
        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        self.out_dim = input_shape[2]
        # self.b = K.eye(self.out_dim, name='strange?')
        self.built = True

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise Exception('The shape of the input to "LogTransform'
                            'is not fully defined '
                            '(got ' + str( input_shape[1:]) + '. ')
        assert input_shape[1] == input_shape[2]
        return input_shape

    def get_config(self):
        """ Get config for model save and reload """
        config = {'epsilon':self.eps}
        base_config = super(MatrixReLU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        """
        2016.12.15 Implement with the theano.scan

        Returns
        -------
        3D tensor with same shape as input
        """
        if K.backend() == 'theano':
            # from theano import scan
            # components, update = scan(fn=lambda tx: self.logm(tx),
            #                           outputs_info=None,
            #                           sequences=[x],
            #                           non_sequences=None)
            #
            # return components
            raise ValueError("Matrix relu not supported for Theano")
        else:
            if self.built:
                import tensorflow as tf
                s, u = tf.self_adjoint_eig(x)
                comp = tf.zeros_like(s) + self.eps
                inner = tf.where(tf.less(s, comp), comp, s)
                # inner = tf.log(inner)
                # inner = tf.Print(inner, [inner], message='MatrixReLU_inner :', summarize=10)
                # inner = tf.where(tf.is_nan(inner), tf.zeros_like(inner), inner)
                inner = tf.matrix_diag(inner)
                tf_relu = tf.matmul(u, tf.matmul(inner, tf.transpose(u, [0, 2, 1])))
                return tf_relu

            else:
                raise RuntimeError("Log transform layer should be built before using")


class ExpandDims(Layer):
    """ define expand dimension layer """
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(ExpandDims, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.axis >=0:
            return input_shape[0:self.axis] + (1,) + input_shape[self.axis:]
        else:
            return input_shape + (1,)

    def get_config(self):
        config = {'axis': self.axis
                  }
        base_config = super(ExpandDims, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        return tf.expand_dims(x, axis=self.axis)


class Squeeze(Layer):
    """ define expand dimension layer """
    def __init__(self, axis=-1):
        self.axis = axis
        super(Squeeze, self).__init__()

    def compute_output_shape(self, input_shape):
        assert input_shape[self.axis] == 1
        return input_shape[0:self.axis] + input_shape[self.axis: -1]

    def get_config(self):
        config = {'axis': self.axis
                  }
        base_config = super(Squeeze, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        return tf.squeeze(x, axis=self.axis)


class BatchNormalization_v2(BatchNormalization):
    """
    Support expand dimension batch-normalization
    """
    def __init__(self, expand_dim=True, **kwargs):
        self.expand_dim = expand_dim
        super(BatchNormalization_v2, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.expand_dim and x is not None:
            x = tf.expand_dims(x, axis=-1)

        if self.mode == 0 or self.mode == 2:
            assert self.built, 'Layer must be built before being called'
            input_shape = K.int_shape(x)

            reduction_axes = list(range(len(input_shape)))
            del reduction_axes[self.axis]
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[self.axis] = input_shape[self.axis]

            x_normed, mean, std = K.normalize_batch_in_training(
                x, self.gamma, self.beta, reduction_axes,
                epsilon=self.epsilon)

            if self.mode == 0:
                self.add_update([K.moving_average_update(self.running_mean, mean, self.momentum),
                                 K.moving_average_update(self.running_std, std, self.momentum)], x)

                if sorted(reduction_axes) == range(K.ndim(x))[:-1]:
                    x_normed_running = K.batch_normalization(
                        x, self.running_mean, self.running_std,
                        self.beta, self.gamma,
                        epsilon=self.epsilon)
                else:
                    # need broadcasting
                    broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
                    broadcast_running_std = K.reshape(self.running_std, broadcast_shape)
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                    broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
                    x_normed_running = K.batch_normalization(
                        x, broadcast_running_mean, broadcast_running_std,
                        broadcast_beta, broadcast_gamma,
                        epsilon=self.epsilon)

                # pick the normalized form of x corresponding to the training phase
                x_normed = K.in_train_phase(x_normed, x_normed_running)

        elif self.mode == 1:
            # sample-wise normalization
            m = K.mean(x, axis=-1, keepdims=True)
            std = K.sqrt(K.var(x, axis=-1, keepdims=True) + self.epsilon)
            x_normed = (x - m) / (std + self.epsilon)
            x_normed = self.gamma * x_normed + self.beta

        if self.expand_dim and x is not None:
            x_normed = tf.squeeze(x_normed, squeeze_dims=-1)

        return x_normed


class PowTransform(Layer):
    """
    PowTranform layer supports the input of a 3D tensor, output a corresponding 3D tensor in
        Power Euclidean space

    References:
        Is second-order information really helpful in large-scale visual recognition?

    It implement the Matrix Logarithm with a small shift (epsilon)

        # Input shape
            3D tensor with (samples, input_dim, input_dim)
        # Output shape
            3D tensor with (samples, input_dim, input_dim)
        # Arguments
            epsilon

    """

    def __init__(self, alpha=0.5, epsilon=1e-7, normalization=None, **kwargs):
        self.input_spec = [InputSpec(ndim='3+')]
        self.eps = epsilon
        self.out_dim = None
        self.alpha = alpha
        self.norm = normalization
        # self.b = None
        super(PowTransform, self).__init__(**kwargs)

    def build(self, input_shape):
        """
                Build the model based on input shape
                Should not set the weight vector here.
                :param input_shape: (nb-sample, input_dim, input_dim)
                :return:
                """
        assert len(input_shape) == 3
        assert input_shape[1] == input_shape[2]
        self.out_dim = input_shape[2]
        # self.b = K.eye(self.out_dim, name='strange?')
        self.built = True

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise Exception('The shape of the input to "LogTransform'
                            'is not fully defined '
                            '(got ' + str(input_shape[1:]) + '. ')
        assert input_shape[1] == input_shape[2]
        return input_shape

    def get_config(self):
        """ Get config for model save and reload """
        config = {'epsilon': self.eps}
        base_config = super(PowTransform, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        """

        Returns
        -------
        3D tensor with same shape as input
        """
        if K.backend() == 'theano' or K.backend() == 'CNTK':
            raise NotImplementedError("This is not implemented for theano anymore.")
        else:
            if self.built:
                # import tensorflow as tf
                # from kyu.tensorflow.ops import safe_truncated_sqrt, safe_sign_sqrt
                # with tf.device('/cpu:0'):
                #     s, u = safe_matrix_eig_op(x)
                #     # s, u = tf.self_adjoint_eig(x)
                # inner = safe_sign_sqrt(s)
                # if self.norm == 'l2':
                #     inner /= tf.reduce_max(inner)
                # elif self.norm == 'frob' or self.norm == 'Frob':
                #     inner /= tf.sqrt(tf.reduce_sum(s))
                # elif self.norm is None:
                #     pass
                # else:
                #     raise ValueError("PowTransform: Normalization not supported {}".format(self.norm))
                # # inner = tf.Print(inner, [inner], message='power inner', summarize=65)
                # inner = tf.matrix_diag(inner)
                # tf_pow = tf.matmul(u, tf.matmul(inner, tf.transpose(u, [0, 2, 1])))
                # return tf_pow

                import tensorflow as tf

                s, u = tf.self_adjoint_eig(x)
                comp = tf.zeros_like(s) + self.eps
                inner = tf.where(tf.less(s, comp), comp, s)
                inner = inner + self.eps
                inner = tf.sqrt(inner)
                if self.norm == 'l2':
                    pass
                elif self.norm == 'frob' or self.norm == 'Frob':
                    inner /= tf.norm(s)
                # inner = tf.Print(inner, [inner], message='power inner', summarize=65)
                inner = tf.matrix_diag(inner)
                tf_pow = tf.matmul(u, tf.matmul(inner, tf.transpose(u, [0, 2, 1])))
                return tf_pow

            else:
                raise RuntimeError("PowTransform layer should be built before using")




class Scale(Layer):
    '''Custom Layer for DenseNet used for BatchNormalization.

    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''

    def __init__(self, weights=None, axis=-1, momentum=0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        # Tensorflow >= 1.0.0 compatibility
        self.gamma = K.variable(self.gamma_init(shape), name='{}_gamma'.format(self.name))
        self.beta = K.variable(self.beta_init(shape), name='{}_beta'.format(self.name))
        # self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        # self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def robust_estimate_eigenvalues(s, alpha):
    """
    Robust estimatation in RAID-G paper


    Reference
    ----------
        Wang, Q., Li, P., Zuo, W., & Zhang, L. (2016).
        RAID-G - Robust Estimation of Approximate Infinite Dimensional
            Gaussian with Application to Material Recognition.

    Parameters
    ----------
    s : tf.tensor   Tensorflow input

    Returns
    -------

    """
    return K.sqrt(K.pow((1 - alpha) / 2 / alpha, 2) + s / alpha) - (1-alpha) / (2*alpha)


class L2InnerNorm(Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l2=0.):
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        regularization = 0.
        regularization += K.sum(self.l2 * K.square(K.dot(x, K.transpose(x))))
        return regularization

    def get_config(self):
        return {
                'l2': float(self.l2)}


class FrobNormRegularizer(Regularizer):
    def __init__(self, dim, alpha=0.01):
        self.alpha = alpha
        self.dim = dim

    def __call__(self, x):
        regularization = self.alpha * K.sqrt(K.sum((x - K.eye(self.dim))**2)) / self.dim
        return regularization

    def get_config(self):
        return {'name': self.__class__.__name__,
                'dim': float(self.dim),
                'alpha': float(self.alpha)}


class VonNeumannDistanceRegularizer(Regularizer):
    def __init__(self, dim, alpha=0.01, epsilon=1e-5):
        self.alpha = alpha
        self.dim = dim
        self.eps = epsilon
        if K.backend() == 'theano':
            raise RuntimeError("v-N divergence not support theano now")

    def __call__(self, x):
        """
        Define the regularization of von Neumann matrix divergence

        Parameters
        ----------
        x

        Returns
        -------

        """
        import tensorflow as tf
        s, u = tf.self_adjoint_eig(x)
        ## TO ensure the stability
        comp = tf.zeros_like(s) + self.eps
        # comp = tf.Print(comp, [comp], message='comp')
        inner = tf.where(tf.less(s, comp), comp, s)
        # inner = tf.Print(inner, [inner], message='inner', summarize=self.dim)
        inner = tf.log(inner)
        von = tf.matmul(tf.matrix_diag(s), tf.matrix_diag(inner)) - tf.matrix_diag(s - 1)
        von = tf.matmul(u, tf.matmul(von, tf.transpose(u, [0,2,1])))
        # von = tf.Print(von, [von], message='von')
        reg = tf.reduce_sum(self.alpha * tf.trace(von, 'vN_reg')) / self.dim
        # reg = tf.Print(reg, [reg], message='vN_reg')
        return reg

    def get_config(self):
        return {'name': self.__class__.__name__,
                'dim': float(self.dim),
                'alpha': float(self.alpha),
                }


def fob(dim, alpha=0.01):
    return FrobNormRegularizer(dim, alpha)


def vN(dim, alpha=0.01, epsilon=1e-5):
    return VonNeumannDistanceRegularizer(dim, alpha, epsilon)