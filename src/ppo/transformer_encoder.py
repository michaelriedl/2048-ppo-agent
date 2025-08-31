import numpy as np
import torch
import torch.nn as nn

BOARD_SIZE = 4


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding2D(nn.Module):
    def __init__(self, x_size, y_size, channels, dropout=0.1, dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.dtype_override = dtype_override
        self.channels = channels

        # Create the dropout
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encoding matrix
        pos_x = torch.arange(x_size, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y_size, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros(
            (x_size, y_size, self.channels * 2),
            dtype=torch.float,
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y
        emb = emb.unsqueeze(0)
        self.register_buffer("pe", emb)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Tensor of the same size with the positional encoding added
        """
        tensor = tensor + self.pe[:, : tensor.shape[1], : tensor.shape[2]]

        return self.dropout(tensor)

    def forward_flat(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: A 3d tensor of size (batch_size, xy, ch). It is assumed
            that x and y are the same x_size and y_size as the embedding.
        :return: Tensor of the same size with the positional encoding added
        """
        pe = self.pe.reshape(1, -1, self.org_channels)
        tensor = tensor + pe
        return self.dropout(tensor)

    def forward_with_inds(self, x: torch.Tensor, inds: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass of the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to apply the positional encoding to. The shape
            should be (batch_size, seq_len, d_model).
        inds : torch.Tensor
            The indices of the input tensor. The shape should be (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            The resulting tensor after applying the positional encoding to the
            input. The shape is the same as the input.

        """
        # Flatten the positional encoding
        pe = self.pe.reshape(1, -1, self.org_channels)
        x = x + pe[0, inds]

        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        """
        Initializes the TransformerEncoder object.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input.
        nhead : int
            The number of heads in the multi-head attention.
        num_layers : int
            The number of sub-encoder-layers in the encoder.
        dim_feedforward : int
            The dimension of the feedforward network model.
        dropout : float, optional
            The dropout value, by default 0.1.
        """
        super(TransformerEncoder, self).__init__()
        # Store the input parameters
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        # Create the positional encoding
        self.positional_encoding = PositionalEncoding2D(
            BOARD_SIZE, BOARD_SIZE, channels=self.d_model, dropout=self.dropout
        )
        # Create the transformer encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                batch_first=True,
            ),
            num_layers=self.num_layers,
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the transformer encoder.

        Parameters
        ----------
        src : torch.Tensor
            The sequence to the encoder (required) with shape (batch_size, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            The output tensor with shape (batch_size, seq_len, d_model).
        """
        return self.encoder(self.positional_encoding.forward_flat(src))
