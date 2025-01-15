import torch
import torch.nn as nn
from backbone import TransformerSentenceEncoderLayer
from typing import Tuple
from copy import deepcopy


def get_act_fn(name: str) -> nn.Module:
    """
    Get PyTorch activation function by name.

    Args:
        name: Name of the activation function

    Returns:
        PyTorch activation function module

    Raises:
        ValueError: If activation function name is not supported
    """
    activation_fns = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.3),
        'elu': nn.ELU(),
        'sigmoid': nn.Sigmoid(),
        'softplus': nn.Softplus()
    }

    if name not in activation_fns:
        raise ValueError(
            f'Invalid activation function. Supported functions: {", ".join(activation_fns.keys())}'
        )

    return activation_fns[name]


class HAAQINet(nn.Module):
    """
    HAAQI (Hearing-Aid Audio Quality Index) Neural Network model.
    Base implementation with LSTM and attention mechanisms.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        linear_output: int,
        act_fn: str,
        beats_model
    ):
        """
        Initialize HAAQI network.

        Args:
            input_size: Size of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            linear_output: Size of linear layer outputs
            act_fn: Activation function name
        """
        super().__init__()
        self.blstm = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size,
                             num_layers=num_layers,
                             dropout=dropout,
                             bidirectional=True,
                             batch_first=True)
        # self.transform = get_feature()
        self.dim = 768
        self.beats_model = beats_model
        self.linear0 = nn.Linear(self.dim, linear_output, bias=True)
        self.linear1 = nn.Linear(hidden_size*2, linear_output, bias=True)
        self.act_fn = get_act_fn(act_fn)
        self.dropout = nn.Dropout(p=0.3)
        self.haaqiAtt_layer = nn.MultiheadAttention(linear_output, num_heads=16)
        
        self.haaqiframe_score = nn.Linear(linear_output, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.haaqiaverage_score = nn.AdaptiveAvgPool1d(1)
        
        # weighted sum
        weight_dim = 12
        self.weights = nn.Parameter(torch.ones(weight_dim))
        self.softmax = nn.Softmax(-1)
        layer_norm = []
        for _ in range(weight_dim):
            layer_norm.append(nn.LayerNorm(self.dim))
        self.layer_norm = nn.Sequential(*layer_norm)

    def forward(
        self,
        x: torch.Tensor,
        hl: torch.Tensor,
        layer_norm: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.

        Args:
            x: Input tensor of shape (batch_size, freq, time)
            hl: Hearing level tensor

        Returns:
            Tuple containing:
                - Frame-level HAAQI scores
                - Average HAAQI score
        """
        x, xs = self.beats_model.extract_features(x)[:2]
        xs = torch.cat(xs, 2)
        B, Freq, embed_dim = xs.size()
        lms = torch.split(xs, self.dim, dim=2)
        for i, (lm, layer, weight) in enumerate(zip(lms, self.layer_norm, self.softmax(self.weights))):
            if layer_norm:
                lm = layer(lm)
            if i == 0:
                out = lm*weight
            else:
                out = out+lm*weight
        weighted_sum_x = out 
        
        x_reduced = self.linear0(weighted_sum_x)
        hl = hl.unsqueeze(1)
        hl_repeat = hl.repeat(1, Freq, 1)
        x_concate = torch.cat((x_reduced, hl_repeat), 2)
        
        out, _ = self.blstm(x_concate)
        out = self.dropout(self.act_fn(self.linear1(out))).transpose(0,1) 
        haaqi, _ = self.haaqiAtt_layer(out, out, out)

        haaqi = haaqi.transpose(0, 1)
        haaqi = self.haaqiframe_score(haaqi) 
        haaqi = self.sigmoid(haaqi)
        haaqi_fram = haaqi.permute(0, 2, 1)
        haaqi_avg = self.haaqiaverage_score(haaqi_fram)
        
        return haaqi_fram, haaqi_avg.squeeze(1)


class HAAQINetStudent(nn.Module):
    """
    HAAQI (Hearing-Aid Audio Quality Index) Neural Network Student model.
    Combines BEATs architecture with LSTM for audio quality assessment.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout, linear_output, act_fn):
        super().__init__()
        self._init_beats_config()
        self._init_layers(input_size, hidden_size, num_layers, dropout, linear_output, act_fn)
        self._init_beats_layers()
        self._init_weighted_sum()

    def _init_beats_config(self):
        """Initialize BEATs configuration parameters"""
        # Patch embedding settings
        self.input_patch_size = 16
        self.embed_dim = 512
        self.conv_bias = False
        self.dim = 768

        # Encoder settings
        self.encoder_layers = 12
        self.encoder_embed_dim = 768
        self.encoder_ffn_embed_dim = 3072
        self.encoder_attention_heads = 12
        self.activation_fn = "gelu"

        # Normalization settings
        self.layer_wise_gradient_decay_ratio = 1.0
        self.layer_norm_first = False
        self.deep_norm = True

        # Dropout settings
        self.encoder_dropout = 0.1
        self.attention_dropout = 0.1
        self.activation_dropout = 0.0
        self.encoder_layerdrop = 0.0
        self.dropout_input = 0.0

        # Positional embedding settings
        self.conv_pos = 128
        self.conv_pos_groups = 16
        
        # Relative position embedding settings
        self.relative_position_embedding = True
        self.num_buckets = 320
        self.max_distance = 1280
        self.gru_rel_pos = True

        # Predictor settings
        self.finetuned_model = False
        self.predictor_dropout = 0.1
        self.predictor_class = 527

    def _init_layers(self, input_size, hidden_size, num_layers, dropout, linear_output, act_fn):
        """Initialize neural network layers"""
        # LSTM and linear layers
        self.blstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        
        self.linear0 = nn.Linear(self.dim, linear_output)
        self.linear1 = nn.Linear(hidden_size * 2, linear_output)
        self.linear2 = nn.Linear(self.embed_dim, self.dim)
        
        # Attention and scoring layers - keeping original names
        self.haaqiAtt_layer = nn.MultiheadAttention(linear_output, num_heads=16)
        self.haaqiframe_score = nn.Linear(linear_output, 1)
        self.haaqiaverage_score = nn.AdaptiveAvgPool1d(1)
        
        # Additional layers
        self.act_fn = self._get_act_fn(act_fn)
        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()

    def _init_beats_layers(self):
        """Initialize BEATs-specific layers"""
        # Patch embedding and normalization
        self.patch_embedding = nn.Conv2d(
            1, self.embed_dim,
            kernel_size=self.input_patch_size,
            stride=self.input_patch_size,
            bias=self.conv_bias
        )
        self.beats_layer_norm = nn.LayerNorm(self.embed_dim)
        
        # Transformer encoder layers
        encoder_params = {
            'embedding_dim': self.encoder_embed_dim,
            'ffn_embedding_dim': self.encoder_ffn_embed_dim,
            'num_attention_heads': self.encoder_attention_heads,
            'dropout': self.encoder_dropout,
            'attention_dropout': self.attention_dropout,
            'activation_dropout': self.activation_dropout,
            'activation_fn': self.activation_fn,
            'layer_norm_first': self.layer_norm_first,
            'deep_norm': self.deep_norm,
            'has_relative_attention_bias': self.relative_position_embedding,
            'num_buckets': self.num_buckets,
            'max_distance': self.max_distance,
            'gru_rel_pos': self.gru_rel_pos,
            'encoder_layers': self.encoder_layers,
        }
        
        self.encoder0 = TransformerSentenceEncoderLayer(**encoder_params)
        self.encoder1 = TransformerSentenceEncoderLayer(**encoder_params)
        self.encoder2 = TransformerSentenceEncoderLayer(**encoder_params)
        
        # Linear encoders
        self.encoder5 = nn.Linear(self.dim, self.dim)
        self.encoder8 = nn.Linear(self.dim, self.dim)
        self.encoder11 = nn.Linear(self.dim, self.dim)
        
        # Set requires_grad for BEATs layers
        beats_layers = [self.patch_embedding, self.beats_layer_norm,
                       self.encoder0, self.encoder1, self.encoder2]
        for layer in beats_layers:
            for param in layer.parameters():
                param.requires_grad = True

    def _init_weighted_sum(self):
        """Initialize weighted sum components"""
        weight_dim = 3
        self.weights = nn.Parameter(torch.ones(weight_dim))
        self.softmax = nn.Softmax(dim=-1)
        
        # Keep original layer_norm name and structure
        layer_norm = []
        for _ in range(weight_dim):
            layer_norm.append(nn.LayerNorm(self.dim))
        self.layer_norm = nn.Sequential(*layer_norm)

    @staticmethod
    def _get_act_fn(name):
        """Get activation function by name"""
        return getattr(nn, name)() if hasattr(nn, name) else nn.ReLU()

    def forward(self, fbank, hl, layer_norm=True):
        """
        Forward pass of the HAAQI network.
        
        Args:
            fbank: Input filterbank features
            hl: Hearing level features
            layer_norm: Whether to apply layer normalization
            
        Returns:
            tuple: (frame_scores, average_score, encoder5_output, encoder8_output, encoder11_output)
        """
        # Patch embedding and initial processing
        x = self.patch_embedding(fbank)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
        x = self.beats_layer_norm(x)
        x = self.linear2(x)
        
        # Transformer encoders
        x_te0 = self.encoder0(x)
        x_te1 = self.encoder1(x_te0[0])
        x_te2 = self.encoder2(x_te1[0])
        
        # Weighted sum computation
        xs = torch.cat([x_te0[0], x_te1[0], x_te2[0]], 2)
        lms = torch.split(xs, self.dim, dim=2)
        
        # Keep original weighted sum implementation
        for i, (lm, layer, weight) in enumerate(zip(lms, self.layer_norm, self.softmax(self.weights))):
            if layer_norm:
                lm = layer(lm)
            if i == 0:
                out = lm * weight
            else:
                out = out + lm * weight
        weighted_sum = out
        
        # Additional encoders
        x_te5 = self.encoder5(weighted_sum)
        x_te8 = self.encoder8(weighted_sum)
        x_te11 = self.encoder11(weighted_sum)
        
        # HAAQI score computation
        batch_size, seq_len, _ = weighted_sum.size()
        x_reduced = self.linear0(weighted_sum)
        
        hl_expanded = hl.unsqueeze(1).repeat(1, seq_len, 1)
        combined = torch.cat((x_reduced, hl_expanded), 2)
        
        lstm_out, _ = self.blstm(combined)
        processed = self.dropout(self.act_fn(self.linear1(lstm_out))).transpose(0, 1)
        attention_out, _ = self.haaqiAtt_layer(processed, processed, processed)
        
        haaqi = attention_out.transpose(0, 1)
        haaqi = self.haaqiframe_score(haaqi)
        haaqi = self.sigmoid(haaqi)
        haaqi_fram = haaqi.permute(0, 2, 1)
        haaqi_avg = self.haaqiaverage_score(haaqi_fram)
        
        return haaqi_fram, haaqi_avg.squeeze(1), x_te5, x_te8, x_te11