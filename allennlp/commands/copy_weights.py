import functools
from os import path

import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.python import pywrap_tensorflow
import torch


# From https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification: https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def get_torch_param_names(model):
    res = []
    for k, _ in model.named_parameters():
        res.append(k)
    return sorted(res)


def get_tf_weights(tf_archive):
    file_name = path.join(tf_archive, "model.max.ckpt")
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    tf_names = sorted(var_to_shape_map.keys())

    tf_names = [name for name in tf_names if "Adam" not in name and "beta" not in name
                and "global_step" not in name]

    tf_params = {}
    for key in tf_names:
        tf_params[key] = reader.get_tensor(key)

    return tf_params


def set_lstm_weights(model, tf_weights):
    def get_tf(key):
        fullkey = "layer_0/bidirectional_rnn/" + key
        return tf_weights[fullkey]

    def swap_inner_blocks(X):
        Y = X.copy()
        Y[hidden_dim: 2 * hidden_dim] = X[2 * hidden_dim: 3 * hidden_dim]
        Y[2 * hidden_dim: 3 * hidden_dim] = X[hidden_dim: 2 * hidden_dim]
        return Y

    torch_lstm = model._context_layer._module
    total_hidden_dim, input_dim = tuple(torch_lstm.weight_ih_l0.shape)
    hidden_dim = int(total_hidden_dim / 4)

    kernel_fw = get_tf("fw/basic_lstm_cell/kernel")
    bias_fw = get_tf("fw/basic_lstm_cell/bias")
    bias_fw[2 * hidden_dim: 3 * hidden_dim] += 1
    kernel_ih_fw = kernel_fw[:input_dim, :].T
    kernel_hh_fw = kernel_fw[input_dim:, :].T
    torch_lstm.weight_ih_l0 = torch.nn.Parameter(torch.tensor(swap_inner_blocks(kernel_ih_fw)))
    torch_lstm.weight_hh_l0 = torch.nn.Parameter(torch.tensor(swap_inner_blocks(kernel_hh_fw)))
    torch_lstm.bias_ih_l0 = torch.nn.Parameter(torch.tensor(swap_inner_blocks(bias_fw)))
    torch_lstm.bias_hh_l0 = torch.nn.Parameter(torch.zeros_like(torch_lstm.bias_ih_l0))

    kernel_bw = get_tf("bw/basic_lstm_cell/kernel")
    bias_bw = get_tf("bw/basic_lstm_cell/bias")
    bias_bw[2 * hidden_dim: 3 * hidden_dim] += 1
    kernel_ih_bw = kernel_bw[:input_dim, :].T
    kernel_hh_bw = kernel_bw[input_dim:, :].T
    torch_lstm.weight_ih_l0_reverse = torch.nn.Parameter(torch.tensor(swap_inner_blocks(kernel_ih_bw)))
    torch_lstm.weight_hh_l0_reverse = torch.nn.Parameter(torch.tensor(swap_inner_blocks(kernel_hh_bw)))
    torch_lstm.bias_ih_l0_reverse = torch.nn.Parameter(torch.tensor(swap_inner_blocks(bias_bw)))
    torch_lstm.bias_hh_l0_reverse = torch.nn.Parameter(torch.zeros_like(torch_lstm.bias_ih_l0_reverse))


def set_pairs(model, tf_weights, name_pairs):
    """
    The `name_pairs` are pairs whose first element is the name of the pytorch array to replce, and
    the second is the name of the corresponding tf tensor.
    """
    for torch_name, tf_name in name_pairs:
        new_weights = tf_weights[tf_name]
        # Torch weights are transposed from tensorflow.
        if "weight" in torch_name:
            new_weights = new_weights.T
        new_params = torch.nn.Parameter(torch.tensor(new_weights))
        rsetattr(model, torch_name, new_params)


def set_entity_weights(model, tf_weights):
    tf_relevant = {k: v for k, v in tf_weights.items() if "entity_scores" in k}
    pruner = model._relation._mention_pruner._scorer
    name_pairs = [('0._module._linear_layers.0.bias', 'entity_scores/hidden_bias_0'),
                  ('0._module._linear_layers.0.weight', 'entity_scores/hidden_weights_0'),
                  ('0._module._linear_layers.1.bias', 'entity_scores/hidden_bias_1'),
                  ('0._module._linear_layers.1.weight', 'entity_scores/hidden_weights_1'),
                  ('1._module.bias', 'entity_scores/output_bias'),
                  ('1._module.weight', 'entity_scores/output_weights')]
    set_pairs(pruner, tf_relevant, name_pairs)


def set_relation_weights(model, tf_weights):
    tf_relevant = {k: v for k, v in tf_weights.items() if "relation_scores" in k}
    relation_feedforward = model._relation._relation_feedforward
    feedforward_pairs = [('_linear_layers.0.bias', 'relation_scores/hidden_bias_0'),
                         ('_linear_layers.0.weight', 'relation_scores/hidden_weights_0'),
                         ('_linear_layers.1.bias', 'relation_scores/hidden_bias_1'),
                         ('_linear_layers.1.weight', 'relation_scores/hidden_weights_1')]
    set_pairs(relation_feedforward, tf_relevant, feedforward_pairs)

    relation_scorer = model._relation._relation_scorer
    scorer_pairs = [('bias', 'relation_scores/output_bias'),
                    ('weight', 'relation_scores/output_weights')]
    set_pairs(relation_scorer, tf_relevant, scorer_pairs)


def copy_weights(model, tf_archive):
    tf_weights = get_tf_weights(tf_archive)
    set_lstm_weights(model, tf_weights)
    set_entity_weights(model, tf_weights)
    set_relation_weights(model, tf_weights)
