import torch
import torch.nn as nn


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
        return self.encoder(src)
