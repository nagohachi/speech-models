import torch
import torch.nn as nn


class PointwiseConv1d(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(input_size, output_size, 1)

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x, xlens


class DepthwiseConv1d(nn.Module):
    def __init__(self, input_size: int, output_size: int, kernel_size: int) -> None:
        super().__init__()
        assert input_size == output_size, (
            "For depthwise convolution, `input_size` must be equal to `output_size`"
        )
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            input_size,
            output_size,
            kernel_size,
            groups=input_size,
            padding=self.padding,
        )

    def _calc_out_seq_len(
        self,
        seq_len: torch.Tensor,
        dilation: int = 1,
        stride: int = 1,
    ) -> torch.Tensor:
        # calculate output length after conv subsampling module.
        # See https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        return (
            seq_len + 2 * self.padding - dilation * (self.kernel_size - 1) - 1
        ) // stride + 1

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        xlens = self._calc_out_seq_len(xlens)
        return x, xlens


class ConformerGLU(nn.Module):
    def __init__(
        self, input_size: int, output_size: int, add_trainable_param: bool
    ) -> None:
        super().__init__()
        if add_trainable_param:
            self.w = nn.Linear(input_size, output_size * 2)
        else:
            assert input_size % 2 == 0 and input_size == output_size * 2
            self.w = nn.Identity()

        self.sigmoid = nn.Sigmoid()
        self.add_trainable_param = add_trainable_param

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """forward pass

        Args:
            x (torch.Tensor): innput tensor of shape (batch_size, seq_len, hidden_size)
            xlens (torch.Tensor): lengths of x of shape (batch_size, )

        Returns:
            tuple[torch.Tensor, torch.Tensor]: output of (x, xlens)
        """
        x = self.w(x)
        hidden_size = x.size(2)
        split_size = hidden_size // 2
        x = x[:, :, :split_size] * self.sigmoid(x[:, :, split_size:])

        return x, xlens


class ConformerConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        add_trainable_param_in_glu: bool,
        kernel_size_in_depthwise_conv: int,
        dropout_prob: float,
    ) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size)

        glu_output_size = hidden_size if add_trainable_param_in_glu else hidden_size * 2

        self.pointwise_conv1 = PointwiseConv1d(hidden_size, glu_output_size)
        self.glu = ConformerGLU(
            glu_output_size, hidden_size, add_trainable_param_in_glu
        )

        self.depthwise_conv = DepthwiseConv1d(
            hidden_size, hidden_size, kernel_size_in_depthwise_conv
        )

        self.batchnorm = nn.BatchNorm1d(hidden_size)

        self.swish = nn.SiLU(inplace=True)

        self.pointwise_conv2 = PointwiseConv1d(hidden_size, hidden_size)

        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_residual = x

        # forward pass
        x = self.layernorm(x)
        x, xlens = self.pointwise_conv1(x, xlens)
        x, xlens = self.glu(x, xlens)
        x, xlens = self.depthwise_conv(x, xlens)
        x = x.transpose(1, 2)
        x = self.batchnorm(x)
        x = x.transpose(1, 2)
        x = self.swish(x)
        x, xlens = self.pointwise_conv2(x, xlens)
        x = self.dropout(x)

        return x + x_residual, xlens
