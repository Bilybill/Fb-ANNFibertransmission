from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C

# 0. basic config
__C.version_name = 'fft_lineartrans'
__C.image_dim = 120 if __C.version_name != 'inverse_genspc' else 92
__C.orig_dim = 92 if __C.version_name != 'inverse_genspc' else 120
__C.lamb = 0.1
__C.datafile_location = '../data/Data_1m.h5'
__C.mgpus = False
__C.root_result_dir = '../result_dir/trainlog/%s_train' % __C.version_name

__C.MODEL = edict()
__C.MODEL.mid_dim = []

# general training and testing config
__C.TRAIN = edict()
__C.TRAIN.usefftdata = True
__C.TRAIN.usegenerator = True
__C.TRAIN.x_toload = 'Training/Speckle_images/ImageNet' if __C.version_name != 'inverse_genspc' else 'Training/Original_images/ImageNet'
__C.TRAIN.y_toload = 'Training/Original_images/ImageNet' if __C.version_name != 'inverse_genspc' else 'Training/Speckle_images/ImageNet'
__C.TRAIN.one_minimize = True if __C.version_name == 'inverse_genspc' else False
__C.TRAIN.loadweights = True
# __C.TRAIN.load_path = '../result_dir/%s_resdir/checkpoint' % __C.version_name
__C.TRAIN.load_path = '../result_dir/fft_resdir/bestcheckpoint'

__C.TRAIN.epochs = 500
__C.TRAIN.lr = 2e-8
__C.TRAIN.batch_size = 45
__C.TRAIN.checkpoint_dir = '../result_dir/%s_resdir' % __C.version_name
__C.TRAIN.checkpoint_path = '../result_dir/%s_resdir/checkpoint' % __C.version_name
__C.TRAIN.loss = 'mse'
__C.TRAIN.fftdata_load = '../data/trainfft_data.npy'
__C.TRAIN.use_train_to_test = True

__C.TEST = edict()
__C.TESTALL = True
__C.TEST.usefftdata = True
__C.TEST.compare = False
__C.TEST.testclass = 'punch'
__C.TEST.x_toload = 'Testing/Speckle_images/%s' % __C.TEST.testclass if __C.version_name != 'inverse_genspc' else 'Testing/Original_images/%s' % __C.TEST.testclass
__C.TEST.y_toload = 'Testing/Original_images/%s' % __C.TEST.testclass if __C.version_name != 'inverse_genspc' else 'Testing/Speckle_images/%s' % __C.TEST.testclass
__C.TEST.show_spc = True
__C.TEST.show_rgb = True
__C.TEST.fftdata_load = '../data/testfft_data.npy' if not __C.TESTALL else '../data/testallfft_data.npy'
__C.TEST.testlist = ['Earth_B', 'Earth_G', 'Earth_R', 'Jupyter_B', 'Jupyter_G', 'Jupyter_R', 'cat', 'horse', 'parrot', 'punch']


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))
        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]), type(v), k))
        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(type(value), type(d[subkey]))
        d[subkey] = value


def save_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], edict):
            if logger is not None:
                logger.info('\n%s.%s = edict()' % (pre, key))
            else:
                print('\n%s.%s = edict()' % (pre, key))
            save_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue

        if logger is not None:
            logger.info('%s.%s: %s' % (pre, key, val))
        else:
            print('%s.%s: %s' % (pre, key, val))
