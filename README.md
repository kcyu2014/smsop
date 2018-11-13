# Statistically-motivated Second-order Pooling (ECCV2018)

Code for our work on ECCV2018.
Paper URL: https://arxiv.org/abs/1801.07492

It is implemented in Keras with Tensorflow as backend.

Requirements:
* Tensorflow: 1.4.0
* Keras 2.1.2

## Notes

The first version of the code is put in `snapshot` folder, which containing only the implementation of SMSOP structure. You can obtain these by calling ```get_cov_block(option)``` function in the `main.py`. 

```
def get_cov_block(cov_branch):
    if cov_branch == 'smsop':
        covariance_block = covariance_block_newn_wv
    elif cov_branch == "smsop-equ":
        covariance_block = covariance_block_pv_equivelent
    else:
        raise ValueError('covariance cov_mode not supported')

    return covariance_block
```



