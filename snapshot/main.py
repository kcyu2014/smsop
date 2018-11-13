import numpy as np
import tensorflow as tf

from keras.layers import Flatten, Dense, Conv2D, Conv2DTranspose, Reshape, BatchNormalization, Activation
from keras.layers import concatenate, add, average

from smsop import SecondaryStatistic, WeightedVectorization, GlobalSquarePooling
from assist_layer import FlattenSymmetric, MatrixConcat, SignedSqrt, LogLayer, PowLayer, PowTransform


def get_tensorboard_layer_name_keys():
    return ['cov', 'o2t', 'pv', '1x1', 'bn-', 'last_bn', 'pow', ]


def get_cov_name_base(stage, block, epsilon):
    if epsilon > 0:
        cov_name_base = 'cov-{}-br_{}-eps_{}'.format(str(stage), block, epsilon)
    else:
        cov_name_base = 'cov-{}-br_{}'.format(str(stage), block)
    return cov_name_base


def get_o2t_name_base(stage, block):
    return 'o2t-{}-br_{}'.format(str(stage), block)


def get_pv_name_base(stage, block):
    return 'pv-{}-br_{}'.format(str(stage), block)


def get_name_base(name, stage, block):
    return '{}-{}-br_{}'.format(name, str(stage), block)


def fn_regroup(tensors):
    """
    Python function which takes a list of tensors, returning their combinations.

    Parameters
    ----------
    tensors

    Returns
    -------
    Combinations of them
    """
    outputs = []
    n_inputs = len(tensors)
    for i in range(n_inputs - 1):
        for j in range(i + 1, n_inputs):
            outputs.append(tf.concat([tf.identity(tensors[i]), tf.identity(tensors[j])]))
    return outputs

def covariance_block_newn_wv(input_tensor, nb_class, stage, block,
                             epsilon=0,
                             parametric=[],
                             vectorization='wv',
                             batch_norm=True,
                             batch_norm_end=False,
                             batch_norm_kwargs={},
                             pow_norm=False,
                             weight_decay=0.0,
                             cov_kwargs={},
                             o2t_kwargs={},
                             pv_kwargs={},
                             **kwargs):

    cov_name_base = get_cov_name_base(stage, block, epsilon)
    # o2t_name_base = 'o2t' + str(stage) + block + '_branch'
    wp_name_base = 'pv' + str(stage) + block + '_branch'

    # TODO Test batch-norm first and last
    if isinstance(batch_norm_kwargs, list) and len(batch_norm_kwargs) == 2:
        batch_norm_first_kwargs = batch_norm_kwargs[0]
        batch_norm_end_kwargs = batch_norm_kwargs[1]
    else:
        batch_norm_first_kwargs = batch_norm_kwargs
        batch_norm_end_kwargs = batch_norm_kwargs

    # Add a normalization before goinging into secondary statistics
    x = input_tensor
    if batch_norm:
        print(batch_norm_kwargs)
        batch_norm_layer = BatchNormalization(axis=3, name='last_BN_{}_{}'.format(stage, block),
                                              **batch_norm_first_kwargs)
        x = batch_norm_layer(x)

    with tf.name_scope(cov_name_base):
        x = SecondaryStatistic(name=cov_name_base,
                               **cov_kwargs)(x)
    if pow_norm:
        x = PowTransform()(x)
    if vectorization == 'pv' or vectorization == 'wv':
        pv_kwargs['batch_norm_moving_variance'] = batch_norm_layer.moving_variance if batch_norm else None
        x = WeightedVectorization(nb_class,
                                  name=wp_name_base,
                                  **pv_kwargs)(x)
        # legacy
        use_sqrt = pv_kwargs['output_sqrt']
        if use_sqrt:
            x = SignedSqrt(2)(x)

        output_normalization = kwargs['output_normalization']
        if output_normalization == 'log':
            x = LogLayer(scale=1)(x)
        elif output_normalization == 'sqrt':
            x = SignedSqrt(2)(x)
        elif output_normalization == 'cubicroot':
            x = PowLayer(scale=0.5, center=np.power(49./2 - 1.0/3, 1.0/3))(x)

        if batch_norm_end:
            x = Reshape((1,1, nb_class))(x)
            x = BatchNormalization(axis=3, name='BN-end{}_{}'.format(stage, block),
                                   **batch_norm_end_kwargs)(x)
            x = Flatten()(x)
            # Use RELU after BN
            x = Activation("relu")(x)
    return x


def covariance_block_pv_equivelent(input_tensor, nb_class, stage, block,
                                   batch_norm=True,
                                   batch_norm_kwargs={},
                                   batch_norm_end=False,
                                   conv_kwargs={},
                                   gsp_kwargs={},
                                   **kwargs):
    conv_name_base = get_name_base('pv', stage, block)
    gsp_name_base = get_name_base('gsp', stage, block)

    # Add a normalization before goinging into secondary statistics
    x = input_tensor
    if batch_norm:
        print(batch_norm_kwargs)
        x = BatchNormalization(axis=3, name='BN_{}_{}'.format(stage, block),
                               **batch_norm_kwargs)(x)
    # As the PV Layer
    x = Conv2D(filters=nb_class, kernel_size=(1,1), name=conv_name_base, **conv_kwargs)(x)
    x = GlobalSquarePooling(nb_class, name=gsp_name_base, **gsp_kwargs)(x)

    # Handle output normalization
    output_normalization = kwargs['output_normalization']
    use_sqrt = gsp_kwargs['output_sqrt']
    if use_sqrt:
        output_normalization = 'sqrt'
    if output_normalization == 'log':
        x = LogLayer(scale=1)(x)
    elif output_normalization == 'sqrt':
        x = SignedSqrt(2)(x)
    elif output_normalization == 'cubicroot':
        x = PowLayer(scale=0.5, center=np.power(49. / 2 - 1.0 / 3, 1.0 / 3))(x)

    if batch_norm_end:
        print(batch_norm_kwargs)
        x = Reshape((1, 1, nb_class))(x)
        x = BatchNormalization(axis=3, name='BN_{}_{}-end'.format(stage, block),
                               **batch_norm_kwargs)(x)
        x = Flatten()(x)
    return x


def upsample_wrapper_v1(x, last_conv_feature_maps=[],method='conv',kernel=(1,1), stage='', **kwargs):
    """
    Wrapper to decrease the dimension feed into SecondStat layers.

    Parameters
    ----------
    last_conv_feature_maps
    method
    kernel

    Returns
    -------

    """
    if method == 'conv':
        for ind, feature_dim in enumerate(last_conv_feature_maps):
            x = Conv2D(feature_dim, kernel, name='1x1_conv_' + str(ind) + stage, **kwargs)(x)
    elif method == 'deconv':
        for feature_dim in last_conv_feature_maps:
            x = Conv2DTranspose(feature_dim, kernel, name='dconv_' + stage, **kwargs)(x)
    else:
        raise ValueError("Upsample wrapper v1 : Error in method {}".format(method))
    return x


def merge_branches_with_method(concat, cov_outputs,
                               cov_output_vectorization=None,
                               cov_output_dim=1024,
                               pv_constraints=None, pv_regularizer=None, pv_activation='relu',
                               pv_normalization=False,
                               pv_output_sqrt=True, pv_use_bias=False,
                               **kwargs
                               ):
    """
    Helper to merge all branches with given methods.
    Parameters
    ----------
    concat
    cov_outputs
    cov_output_vectorization
    cov_output_dim
    **kwargs (Passed into vectorization layer)
    Returns
    -------

    """
    if len(cov_outputs) < 2:
        return cov_outputs[0]
    else:
        if concat == 'concat':
            if all(len(cov_output.shape) == 3 for cov_output in cov_outputs):
                # use matrix concat
                x = MatrixConcat(cov_outputs)(cov_outputs)
            else:
                x = concatenate(cov_outputs)
        elif concat == 'sum':
            x = add(cov_outputs)
        elif concat == 'ave' or concat == 'avg':
            x = average(cov_outputs)
        else:
            raise ValueError("Concat mode not supported {}".format(concat))
    if cov_output_vectorization == 'wv' or cov_output_vectorization == 'pv':
        x = WeightedVectorization(output_dim=cov_output_dim,
                                  output_sqrt=pv_output_sqrt,
                                  use_bias=pv_use_bias,
                                  normalization=pv_normalization,
                                  kernel_regularizer=pv_regularizer,
                                  kernel_constraint=pv_constraints,
                                  activation=pv_activation,
                                  )(x)
    else:
        raise ValueError("Merge branches with method only support pv layer now")

    return x


def get_cov_block(cov_branch):
    if cov_branch == 'smsop':
        covariance_block = covariance_block_newn_wv
    elif cov_branch == "smsop-equ":
        covariance_block = covariance_block_pv_equivelent
    else:
        raise ValueError('covariance cov_mode not supported')

    return covariance_block

if __name__ == '__main__':
    # The initial version of SMSOP
    get_cov_block('smsop')
    # Equivalent implementation
    get_cov_block('smsop-equ')
