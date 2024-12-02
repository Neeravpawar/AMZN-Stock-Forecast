import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings using sinusoidal functions.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model (int): Embedding dimension.
            max_len (int): Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()

        # Create a matrix of [max_len, d_model] for positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices in the embedding dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the embedding dimensions
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension for broadcasting
        pe = pe.unsqueeze(0)

        # Register as a buffer so it won't be updated during training
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        # Add positional encodings to the input tensor
        return x + self.pe[:, :seq_len, :]


class SeriesDecomp(nn.Module):
    """
    Series decomposition block to separate trend and seasonal components.
    """
    def __init__(self, kernel_size):
        """
        Initialize the series decomposition module.

        Args:
            kernel_size (int): Size of the moving average kernel used for decomposition.
                             Should be an odd number to ensure symmetric padding.
        """
        super(SeriesDecomp, self).__init__()
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

    def moving_avg(self, x, kernel_size):
        pad = (kernel_size - 1) // 2
        # Pad the time dimension with reflection padding to handle edge effects
        x_padded = F.pad(x, (pad, pad), mode='reflect')
        x_avg = F.avg_pool1d(x_padded, kernel_size=kernel_size, stride=1)
        return x_avg

    def forward(self, x):
        x = x.transpose(1, 2)
        moving_mean = self.moving_avg(x, self.kernel_size)
        res = x - moving_mean
        res = res.transpose(1, 2)
        moving_mean = moving_mean.transpose(1, 2)
        return res, moving_mean



class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with Period-based Dependencies discovery and Time-based Aggregation.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False, n_heads=8):
        """
        Initialize the AutoCorrelation module.

        Args:
            mask_flag (bool): Whether to use masking in attention. Defaults to True.
            factor (int): Factor used to determine number of top-k correlations. Defaults to 1.
            scale (float, optional): Scaling factor for attention scores. Defaults to None.
            attention_dropout (float): Dropout rate for attention weights. Defaults to 0.1.
            output_attention (bool): Whether to output attention weights. Defaults to False.
            n_heads (int): Number of attention heads. Defaults to 8.
        """
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.n_heads = n_heads

    def time_delay_agg(self, values, corr):
        batch, head, length, channel = values.shape
        top_k = max(1, int(self.factor * math.log(length)))

        # Step 1: Identify top_k weights and delays
        weights, delay = torch.topk(corr, top_k, dim=-1) 
        weights = F.softmax(weights, dim=-1)

        # Step 2: Compute delay indices
        positions = torch.arange(length, device=values.device).reshape(1, 1, length, 1)
        delay_indices = (positions - delay) % length 

        # Flatten batch and head dimensions
        B_H = batch * head
        values = values.reshape(B_H, length, channel)  
        delay_indices = delay_indices.reshape(B_H, length, top_k)
        weights = weights.reshape(B_H, length, top_k)

        # Prepare indices for advanced indexing
        batch_indices = torch.arange(B_H, device=values.device).unsqueeze(1).unsqueeze(2)
        batch_indices = batch_indices.expand(-1, length, top_k)

        # Use advanced indexing to gather values
        gathered_values = values[batch_indices, delay_indices]

        # Multiply by weights
        weighted_values = gathered_values * weights.unsqueeze(-1)

        # Sum over top_k dimension
        result = weighted_values.sum(dim=2)

        result = result.reshape(batch, head, length, channel)

        return result.contiguous()

    def forward(self, queries, keys, values, attn_mask):
        B, L, E = queries.shape
        _, S, D = values.shape
        H = self.n_heads

        # Ensure embedding dimension is divisible by the number of heads
        assert E % H == 0, f"Embedding dimension {E} must be divisible by number of heads {H}"
        head_dim = E // H

        # Step 1: Reshape for multi-head attention
        queries = queries.reshape(B, L, H, head_dim).permute(0, 2, 1, 3).contiguous()
        keys = keys.reshape(B, S, H, head_dim).permute(0, 2, 1, 3).contiguous()
        values = values.reshape(B, S, H, head_dim).permute(0, 2, 1, 3).contiguous()

        # Step 2: Adjust sequence lengths if necessary
        if L > S:
            pad_len = L - S
            zeros = torch.zeros(B, H, pad_len, head_dim, device=values.device)
            values = torch.cat([values, zeros], dim=2)
            keys = torch.cat([keys, zeros], dim=2)
        else:
            values = values[:, :, :L, :]
            keys = keys[:, :, :L, :]

        # Step 3: Compute correlation using FFT
        q_fft = torch.fft.rfft(queries, dim=2)
        k_fft = torch.fft.rfft(keys, dim=2)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=2)

        # Step 4: Time delay aggregation
        V = self.time_delay_agg(values, corr)

        # Step 5: Reshape back to original dimensions
        V = V.permute(0, 2, 1, 3).contiguous().reshape(B, L, -1)

        if self.output_attention:
            return V, corr
        else:
            return V, None

class AutoformerEncoderLayer(nn.Module):
    """
    Autoformer encoder layer with series decomposition.
    """
    def __init__(self, d_model, n_heads, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        """
        Initialize an Autoformer encoder layer.

        Args:
            d_model (int): Dimension of the model/hidden states
            n_heads (int): Number of attention heads
            d_ff (int, optional): Dimension of feed forward network. If None, defaults to 4*d_model
            moving_avg (int, optional): Size of moving average kernel for decomposition. Defaults to 25
            dropout (float, optional): Dropout probability. Defaults to 0.1
            activation (str, optional): Activation function to use ("relu" or "gelu"). Defaults to "relu"
        """
        super(AutoformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.moving_avg = moving_avg
        self.decomp = SeriesDecomp(kernel_size=moving_avg)
        self.attention = AutoCorrelation(mask_flag=False, factor=1, attention_dropout=dropout, n_heads=n_heads)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x, attn_mask=None):
        # Decomposition
        seasonal_part, trend_part = self.decomp(x)

        # AutoCorrelation Mechanism on seasonal component
        new_x, attn = self.attention(
            seasonal_part, seasonal_part, seasonal_part,
            attn_mask=attn_mask
        )

        x = seasonal_part + self.dropout(new_x)
        x = self.norm1(x)

        # Feed-forward Network
        y = self.fc1(x)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        x = self.norm2(x + y)

        # Residual connection with trend component
        residual = trend_part + x
        x, _ = self.decomp(residual)
        x = self.norm3(x)

        return x, attn


class AutoformerDecoderLayer(nn.Module):
    """
    Autoformer decoder layer with series decomposition.
    """
    def __init__(self, d_model, n_heads, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        """
        Initialize an Autoformer decoder layer.

        Args:
            d_model (int): Dimension of the model
            n_heads (int): Number of attention heads
            d_ff (int, optional): Dimension of feed forward network. If None, defaults to 4 * d_model
            moving_avg (int, optional): Size of moving average kernel for decomposition. Defaults to 25
            dropout (float, optional): Dropout probability. Defaults to 0.1
            activation (str, optional): Activation function to use ("relu" or "gelu"). Defaults to "relu"
        """
        super(AutoformerDecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.moving_avg = moving_avg
        self.decomp = SeriesDecomp(kernel_size=moving_avg)
        self.self_attention = AutoCorrelation(mask_flag=True, factor=1, attention_dropout=dropout)
        self.cross_attention = AutoCorrelation(mask_flag=False, factor=1, attention_dropout=dropout)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Decomposition
        seasonal_part, trend_part = self.decomp(x)

        # Self Attention on seasonal component
        new_x, _ = self.self_attention(
            seasonal_part, seasonal_part, seasonal_part,
            attn_mask=x_mask
        )
        x = seasonal_part + self.dropout(new_x)
        x = self.norm1(x)

        # Cross Attention with encoder output
        new_x, _ = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )
        x = x + self.dropout(new_x)
        x = self.norm2(x)

        # Feed-forward Network
        y = self.fc1(x)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        x = self.norm3(x + y)

        # Residual connection with trend component
        residual_trend = trend_part + x
        # Decompose the residual trend
        _, residual_trend = self.decomp(residual_trend)
        x = self.norm4(x + residual_trend)

        return x


class Autoformer(nn.Module):
    """
    Autoformer for time series forecasting with series decomposition.
    """
    def __init__(self, input_dim=4, output_dim=4, seq_len=50, label_len=10, pred_len=10, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=2048, moving_avg=25, dropout=0.1, activation='gelu', output_attention=False):
        """Initialize the Autoformer model.

        Args:
            input_dim (int): Number of input features (e.g. OHLC). Defaults to 4.
            output_dim (int): Dimension of predictions. Defaults to 4.
            seq_len (int): Length of input sequence. Defaults to 50.
            label_len (int): Number of labels to predict. Defaults to 10.
            pred_len (int): Length of prediction sequence. Defaults to 10.
            d_model (int): Dimension of model. Defaults to 512.
            n_heads (int): Number of attention heads. Defaults to 8.
            e_layers (int): Number of encoder layers. Defaults to 3.
            d_layers (int): Number of decoder layers. Defaults to 2.
            d_ff (int): Dimension of feed forward network. Defaults to 2048.
            moving_avg (int): Kernel size for moving average. Defaults to 25.
            dropout (float): Dropout rate. Defaults to 0.1.
            activation (str): Activation function type. Defaults to 'gelu'.
            output_attention (bool): Whether to output attention weights. Defaults to False.
        """
        super(Autoformer, self).__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.d_model = d_model

        # Encoding
        self.enc_embedding = nn.Linear(input_dim, d_model)
        self.dec_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=seq_len + label_len + pred_len)
        
        # Encoder
        self.encoder = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads, d_ff, moving_avg, dropout, activation)
            for _ in range(e_layers)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList([
            AutoformerDecoderLayer(d_model, n_heads, d_ff, moving_avg, dropout, activation)
            for _ in range(d_layers)
        ])
        
        # Output projection
        self.projection = nn.Linear(d_model, output_dim)

    def forward(self, x_enc, x_dec=None):
        if x_dec is None:
            # Prepare decoder input by concatenating the known labels and zeros
            dec_input = x_enc[:, -self.label_len:, :]
            zeros_input = torch.zeros(x_enc.size(0), self.pred_len, x_enc.size(2), device=x_enc.device)
            x_dec = torch.cat([dec_input, zeros_input], dim=1)

        # Embedding
        enc_out = self.positional_encoding(self.enc_embedding(x_enc))
        dec_out = self.positional_encoding(self.dec_embedding(x_dec))

        # Encoder
        attns = []
        for encoder in self.encoder:
            enc_out, attn = encoder(enc_out)
            if self.output_attention:
                attns.append(attn)

        # Decoder
        for decoder in self.decoder:
            dec_out = decoder(dec_out, enc_out)

        # Project to output dimension
        outputs = self.projection(dec_out[:, -self.pred_len:, :])

        if self.output_attention:
            return outputs, attns
        else:
            return outputs
