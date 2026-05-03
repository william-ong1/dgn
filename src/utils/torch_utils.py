import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class RNNBase(nn.Module):
    """
    Base class for RNNs, where its type ``rnn_type`` and ``nonlinearity`` can be specified. The rest follows nn.RNNCell notations.
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        """
        Args:
            input_size: Size of the input to the RNN.
            hidden_size: Size of the hidden state of the RNN.
            kwargs: Additional keyword arguments.
                - rnn_type (str): type of RNN, can be ``rnncell``, ``grucell``, ``rnn`` or ``gru``, default: "grucell"
                - nonlinearity (str): can be tanh or relu (note that it cannot be relu if it's gru), default: "tanh"
                - ev_scale (float): scale factor for the eigenvalues of the weight matrix, default: 1
        """
        super(RNNBase, self).__init__()
        
        self.params = {
            "rnn_type": "grucell",
            "nonlinearity": "tanh",
            "ev_scale": 1,
        }
        self.params.update(kwargs)
        
        # Define attributes and self.model
        self.input_size, self.hidden_size = input_size, hidden_size
        self.model = get_rnn_type(self.input_size, self.hidden_size, self.params["rnn_type"], self.params["nonlinearity"])
        
    def forward(self, *inp):
        return self.model(*inp)
    
    def _init_spectral_weight(self):
        if self.params["rnn_type"] != "rnncell":
            print("WARNING: rnn_type is not RNNcell.")
            return
        
        with torch.no_grad():
            self.model.weight_hh.copy_(self.normalize_eigenvalues(self.model.weight_hh.data, self.params["ev_scale"]))
    
    @staticmethod
    def normalize_eigenvalues(W, target_radius):
        W_np = W.detach().cpu().numpy()
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(W_np)
        spectral_radius = max(np.abs(eigenvalues))

        # Scale the eigenvalues to achieve the target spectral radius
        scaling_factor = target_radius / (spectral_radius + 1e-8)
        scaled_eigenvalues = eigenvalues * scaling_factor

        # Reconstruct the weight matrix
        scaled_W = eigenvectors @ np.diag(scaled_eigenvalues) @ np.linalg.inv(eigenvectors)
        return torch.from_numpy(scaled_W).to(W.device).type_as(W)


class MLPBase(nn.Module):
    """
    Base class for MLPs.
    """
    def __init__(self, features_list):
        """
        Args:
            features_list (list): list containing specifications of each layer, (input_dim, output_dim, nonlinearity).
        """
        super(MLPBase, self).__init__()
        
        # Define attributes
        self.features_list = features_list
        
        # Define self.model
        layers = OrderedDict({})
        for i, (in_features, out_features, activation_type) in enumerate(self.features_list):
            layers[f"linear{i}"] = nn.Linear(in_features=in_features, out_features=out_features)
            if activation_type:
                layers[f"{activation_type}{i}"] = get_activation_type(activation_type)

        self.model = nn.Sequential(layers)
        
    def forward(self, *inp):
        return self.model(*inp)


class RNNWrapper(nn.Module):
    """
    Wrapper class for RNN (GRU, LSTM) classes so that their output matches those of their cell counterparts.
    """
    def __init__(self, model):
        """
        Args:
            model (nn.Module subclass): a RNNCell class
        """
        super(RNNWrapper, self).__init__()
        self.model = model
        
    def forward(self, *inp):
        return self.model(*inp)[0]
    

class RNNChannel(nn.Module):
    """
    RNNChannel class for DGN models, which is a combination of an RNN and an MLP.
    """
    def __init__(self, input_size, hidden_size, output_sizes, output_nonlinearity, rnn_nonlinearity="tanh", override_single=False, **kwargs):
        """
        Args:
            input_size (int): Size of the input to the RNN.
            hidden_size (int): Size of the hidden state of the RNN.
            output_sizes (list): Sizes of the outputs of the MLP.
            output_nonlinearity (str): Nonlinearity of the outputs of the MLP.
            rnn_nonlinearity (str): Nonlinearity of the RNN.
            override_single (bool): If True, the output of the MLP will be split into the different outputs if there are multiple outputs.
            kwargs: Additional keyword arguments for the RNNBase class.
        """
        super(RNNChannel, self).__init__()
        self.input_size = input_size
        self.output_sizes = output_sizes
        self.override_single = override_single
        self.rnn = RNNBase(input_size, hidden_size, nonlinearity=rnn_nonlinearity, **kwargs)
        self.output = MLPBase([[hidden_size, sum(output_sizes), output_nonlinearity]])

    def forward(self, *inp):
        # Forward pass through the RNN
        h = self.rnn(*inp)

        # Forward pass through the MLP
        m = self.output(h)

        # Split the output into the different outputs if there are multiple outputs
        if (len(self.output_sizes) > 1) or self.override_single:
            return h, torch.split(m, self.output_sizes, dim=-1)

        else: return h, m


def get_rnn_type(input_size, hidden_size, rnn_type, nonlinearity):
    """
    Get the RNN type.

    Args:
        input_size: Size of the input to the RNN.
        hidden_size: Size of the hidden state of the RNN.
        rnn_type: Type of RNN.
        nonlinearity: Nonlinearity of the RNN.

    Returns:
        model: The RNN model.
    """
    if rnn_type == "grucell":
        return nn.GRUCell(input_size=input_size, hidden_size=hidden_size)
    elif rnn_type == "rnncell":
        return nn.RNNCell(input_size=input_size, hidden_size=hidden_size, nonlinearity=nonlinearity)
    elif rnn_type == "gru":
        model = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        return RNNWrapper(model)
    elif rnn_type == "rnn":
        model = nn.RNN(input_size=input_size, hidden_size=hidden_size, nonlinearity=nonlinearity, batch_first=True)
        return RNNWrapper(model)
    else:
        raise TypeError(f"``rnn_type`` cannot be {rnn_type}.")
    

# Helper function to get the activation type.
def get_activation_type(activation_type):
    if activation_type == "tanh":
         return nn.Tanh()
    elif activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "sigmoid":
        return nn.Sigmoid()
    else:
        raise TypeError(f"``activation_type`` cannot be {activation_type}.")
