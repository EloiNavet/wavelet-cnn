# DWT_IDWT_layer.py
import torch
import torch.nn as nn
import pywt
from torch.autograd import Function


class DWTFunction_2D_Low(Function):
    """
    2D Discrete Wavelet Transform (DWT) function for the low-pass filter.
    """

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, matrix_h: torch.Tensor, matrix_h_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the 2D DWT function.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            matrix_h (torch.Tensor): Low-pass filter matrix for the rows.
            matrix_h_t (torch.Tensor): Transpose of the low-pass filter matrix for the columns.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height // 2, width // 2).
        """
        # Save matrices for backward pass
        ctx.save_for_backward(matrix_h, matrix_h_t)
        # Perform the low-pass filtering using matrix multiplication
        input_ll = matrix_h @ input @ matrix_h_t
        return input_ll

    @staticmethod
    def backward(ctx, grad_ll: torch.Tensor) -> torch.Tensor:
        """
        Backward pass of the 2D DWT function.

        Args:
            grad_ll (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            torch.Tensor: Gradient of the loss with respect to the input.
        """
        # Retrieve saved matrices
        matrix_h, matrix_h_t = ctx.saved_variables
        # Backpropagate the gradients through the low-pass filter
        grad_output = matrix_h.t() @ grad_ll @ matrix_h_t.t()
        # Only return gradient for input, None for matrices (since they are not learnable)
        return grad_output, None, None


class DWTFunction_2D(Function):
    """
    2D Discrete Wavelet Transform (DWT) function.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        matrix_h_0: torch.Tensor,
        matrix_h_1: torch.Tensor,
        matrix_g_0: torch.Tensor,
        matrix_g_1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the 2D DWT function.

        Args:
            input (torch.Tensor): Input tensor.
            matrix_h_0 (torch.Tensor): Low-pass filter matrix for the rows.
            matrix_h_1 (torch.Tensor): Low-pass filter matrix for the columns.
            matrix_g_0 (torch.Tensor): High-pass filter matrix for the rows.
            matrix_g_1 (torch.Tensor): High-pass filter matrix for the columns.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Output tensors for the low-low, low-high, high-low, and high-high bands of shape (batch_size, in_channels, height // 2, width // 2).
        """
        # Save filter matrices for backward pass
        ctx.save_for_backward(matrix_h_0, matrix_h_1, matrix_g_0, matrix_g_1)

        # Perform wavelet decomposition with low-pass (H) and high-pass (G) filters
        input_l = matrix_h_0 @ input  # Low-pass horizontal
        input_h = matrix_g_0 @ input  # High-pass horizontal
        input_ll = input_l @ matrix_h_1  # Low-pass vertical
        input_lh = input_l @ matrix_g_1  # High-pass vertical
        input_hl = input_h @ matrix_h_1  # High-low-pass vertical
        input_hh = input_h @ matrix_g_1  # High-high-pass vertical

        return input_ll, input_lh, input_hl, input_hh

    @staticmethod
    def backward(
        ctx,
        grad_ll: torch.Tensor,
        grad_lh: torch.Tensor,
        grad_hl: torch.Tensor,
        grad_hh: torch.Tensor,
    ) -> torch.Tensor:
        """
        Backward pass of the 2D DWT function.

        Args:
            grad_ll (torch.Tensor): Gradient of the loss with respect to the output.
            grad_lh (torch.Tensor): Gradient of the loss with respect to the output.
            grad_hl (torch.Tensor): Gradient of the loss with respect to the output.
            grad_hh (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            torch.Tensor: Gradient of the loss with respect to the input.
        """
        # Retrieve saved filter matrices
        matrix_h_0, matrix_h_1, matrix_g_0, matrix_g_1 = ctx.saved_variables

        # Backpropagate through wavelet reconstruction
        grad_L = grad_ll @ matrix_h_1.t() + grad_lh @ matrix_g_1.t()
        grad_H = grad_hl @ matrix_h_1.t() + grad_hh @ matrix_g_1.t()
        grad_input = matrix_h_0.t() @ grad_L + matrix_g_0.t() @ grad_H

        return grad_input, None, None, None, None


class IDWTFunction_2D(Function):
    """
    2D Inverse Discrete Wavelet Transform (IDWT) function.
    """

    @staticmethod
    def forward(
        ctx,
        input_LL: torch.Tensor,
        input_LH: torch.Tensor,
        input_HL: torch.Tensor,
        input_HH: torch.Tensor,
        matrix_low_0: torch.Tensor,
        matrix_low_1: torch.Tensor,
        matrix_high_0: torch.Tensor,
        matrix_high_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the 2D IDWT function.

        Args:
            input_LL (torch.Tensor): Input tensor for the low-low band of shape (batch_size, in_channels, height // 2, width // 2).
            input_LH (torch.Tensor): Input tensor for the low-high band of shape (batch_size, in_channels, height // 2, width // 2).
            input_HL (torch.Tensor): Input tensor for the high-low band of shape (batch_size, in_channels, height // 2, width // 2).
            input_HH (torch.Tensor): Input tensor for the high-high band of shape (batch_size, in_channels, height // 2, width // 2).
            matrix_low_0 (torch.Tensor): Low-pass filter matrix for the rows of shape (height, height + band_length - 2).
            matrix_low_1 (torch.Tensor): Low-pass filter matrix for the columns of shape (width, width + band_length - 2).
            matrix_high_0 (torch.Tensor): High-pass filter matrix for the rows of shape (height, height + band_length - 2).
            matrix_high_1 (torch.Tensor): High-pass filter matrix for the columns of shape (width, width + band_length - 2).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height, width).
        """
        # Save filter matrices for backward pass
        ctx.save_for_backward(matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1)

        # Reconstruct the image using the inverse wavelet transform
        L = (
            input_LL @ matrix_low_1.t() + input_LH @ matrix_high_1.t()
        )  # Low-frequency component
        H = (
            input_HL @ matrix_low_1.t() + input_HH @ matrix_high_1.t()
        )  # High-frequency component
        return (
            matrix_low_0.t() @ L + matrix_high_0.t() @ H
        )  # Combine to form the original signal

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward pass of the 2D IDWT function.

        Args:
            grad_output (torch.Tensor): Gradient of the loss with respect to the output.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Gradient of the loss with respect to the input.
        """
        # Retrieve saved filter matrices
        matrix_low_0, matrix_low_1, matrix_high_0, matrix_high_1 = ctx.saved_variables

        # Backpropagate through inverse wavelet transform
        grad_L = matrix_low_0 @ grad_output
        grad_H = matrix_high_0 @ grad_output

        grad_LL = grad_L @ matrix_low_1
        grad_LH = grad_L @ matrix_high_1
        grad_HL = grad_H @ matrix_low_1
        grad_HH = grad_H @ matrix_high_1

        return grad_LL, grad_LH, grad_HL, grad_HH


class DWT_2D_Low(nn.Module):
    """
    2D Discrete Wavelet Transform (DWT) module for the low-pass filter.
    """

    def __init__(self, wavename: str):
        """
        Initialize the 2D DWT module.

        Args:
            wavename (str): Name of the wavelet. See https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html for more information.
        """
        super(DWT_2D_Low, self).__init__()

        # Device selection (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get the low-pass wavelet filter coefficients
        wavelet = pywt.Wavelet(wavename)
        self.band_low = torch.tensor(wavelet.rec_lo, device=self.device)

        # Validate filter size (should be even)
        self.band_length = len(self.band_low)
        assert (
            self.band_length % 2 == 0
        ), f"Filter length must be even, got {self.band_length}"

    def build_low_filter(
        self, input_height: int, input_width: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build the low-pass filter matrix.

        Args:
            input_height (int): Height of the input tensor.
            input_width (int): Width of the input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Low-pass filter matrices (rows and columns).
        """
        max_size = max(input_height, input_width)
        input_half = max_size // 2
        end = None if self.band_length // 2 == 1 else (-self.band_length // 2 + 1)

        # Build low-pass filter matrix (H)
        matrix_hb = torch.zeros(
            (input_half, max_size + self.band_length - 2), device=self.device
        )
        for i in range(input_half):
            for j in range(self.band_length):
                matrix_hb[i, i * 2 + j] = self.band_low[j]

        # Slice matrices to fit input size
        matrix_h = matrix_hb[
            0 : input_height // 2, 0 : input_height + self.band_length - 2
        ][:, self.band_length // 2 - 1 : end]
        matrix_h_t = matrix_hb[
            0 : input_width // 2, 0 : input_width + self.band_length - 2
        ][:, self.band_length // 2 - 1 : end].t()

        return matrix_h, matrix_h_t

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the 2D DWT module.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height // 2, width // 2).
        """
        # Ensure input is in batch format [batch_size, channels, height, width]
        assert (
            len(input.size()) == 4
        ), f"Expected input of shape [batch_size, channels, height, width], got {len(input.size())}"
        matrix_h, matrix_h_t = self.build_low_filter(input.size()[-2], input.size()[-1])
        return DWTFunction_2D_Low.apply(input, matrix_h, matrix_h_t)


class DWT_2D(nn.Module):
    """
    2D Discrete Wavelet Transform (DWT) module.
    """

    def __init__(self, wavename: str):
        """
        Initialize the 2D DWT module.

        Args:
            wavename (str): Name of the wavelet.
        """
        super(DWT_2D, self).__init__()

        # Device selection (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get wavelet filter coefficients (low-pass and high-pass)
        wavelet = pywt.Wavelet(wavename)
        self.band_low = torch.tensor(wavelet.rec_lo, device=self.device)
        self.band_high = torch.tensor(wavelet.rec_hi, device=self.device)

        # Validate filter sizes (must match)
        assert len(self.band_low) == len(
            self.band_high
        ), f"Low-pass and high-pass filters must have same length"
        self.band_length = len(self.band_low)
        assert (
            self.band_length % 2 == 0
        ), f"Filter length must be even, got {self.band_length}"
        self.band_length_half = self.band_length // 2

    def build_filters(
        self, input_height: int, input_width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the filter matrices.

        Args:
            input_height (int): Height of the input tensor.
            input_width (int): Width of the input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Filter matrices for the low-low, low-high, high-low, and high-high bands of the DWT. There are of shape (height, height + band_length - 2) and (width, width + band_length - 2).
        """
        max_size = max(input_height, input_width)
        input_half = max_size // 2
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        # Build low-pass filter matrix (H)
        matrix_hb = torch.zeros(
            (input_half, max_size + self.band_length - 2), device=self.device
        )
        for i in range(input_half):
            for j in range(self.band_length):
                matrix_hb[i, i * 2 + j] = self.band_low[j]

        matrix_h = matrix_hb[
            0 : input_height // 2, 0 : input_height + self.band_length - 2
        ][:, self.band_length_half - 1 : end]
        matrix_h_t = matrix_hb[
            0 : input_width // 2, 0 : input_width + self.band_length - 2
        ][:, self.band_length_half - 1 : end].t()

        # Build high-pass filter matrix (G)
        matrix_gb = torch.zeros(
            (max_size - input_half, max_size + self.band_length - 2), device=self.device
        )
        for i in range(max_size - input_half):
            for j in range(self.band_length):
                matrix_gb[i, i * 2 + j] = self.band_high[j]

        matrix_g = matrix_gb[
            0 : input_height - input_height // 2,
            0 : input_height + self.band_length - 2,
        ][:, self.band_length_half - 1 : end]
        matrix_g_t = matrix_gb[
            0 : input_width - input_width // 2, 0 : input_width + self.band_length - 2
        ][:, self.band_length_half - 1 : end].t()

        return matrix_h, matrix_h_t, matrix_g, matrix_g_t

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the 2D DWT module.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor for the low-low, low-high, high-low, and high-high bands of shape (batch_size, in_channels, height // 2, width // 2).
        """
        # Ensure input is in batch format [batch_size, channels, height, width]
        assert (
            len(input.size()) == 4
        ), f"Expected input of shape [batch_size, channels, height, width], got {len(input.size())}"
        matrix_h_0, matrix_h_1, matrix_g_0, matrix_g_1 = self.build_filters(
            input.size()[-2], input.size()[-1]
        )
        return DWTFunction_2D.apply(
            input, matrix_h_0, matrix_h_1, matrix_g_0, matrix_g_1
        )


class IDWT_2D(nn.Module):
    """
    2D Inverse Discrete Wavelet Transform (IDWT) module.
    """

    def __init__(self, wavename: str):
        """
        Initialize the 2D IDWT module.

        Args:
            wavename (str): Name of the wavelet.
        """
        super(IDWT_2D, self).__init__()

        # Device selection (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get wavelet filter coefficients (low-pass and high-pass)
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()

        self.band_low = torch.tensor(self.band_low, device=self.device)
        self.band_high = torch.tensor(self.band_high, device=self.device)
        self.band_length = len(self.band_low)

        assert (
            self.band_length % 2 == 0
        ), f"self.band_length: {self.band_length} % 2 != 0"
        self.band_length_half = self.band_length // 2

    def build_h_g(
        self, input_height: int, input_width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the filter matrices.

        Args:
            input_height (int): Height of the input tensor.
            input_width (int): Width of the input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Filter matrices for the low-low, low-high, high-low, and high-high bands of the IDWT. There are of shape (height, height + band_length - 2) and (width, width + band_length - 2).
        """
        max_size = max(input_height, input_width)
        input_half = max_size // 2
        end = None if self.band_length_half == 1 else (-self.band_length_half + 1)

        matrix_hb = torch.zeros(
            (input_half, max_size + self.band_length - 2), device=self.device
        )

        for i in range(input_half):
            for j in range(self.band_length):
                matrix_hb[i, i * 2 + j] = self.band_low[j]

        matrix_h = matrix_hb[
            0 : input_height // 2, 0 : input_height + self.band_length - 2
        ][:, self.band_length_half - 1 : end]
        matrix_h_t = matrix_hb[
            0 : input_width // 2, 0 : input_width + self.band_length - 2
        ][:, self.band_length_half - 1 : end].t()
        matrix_gb = torch.zeros(
            (max_size - input_half, max_size + self.band_length - 2), device=self.device
        )

        for i in range(max_size - input_half):
            for j in range(self.band_length):
                matrix_gb[i, i * 2 + j] = self.band_high[j]
        matrix_g = matrix_gb[
            0 : input_height - input_height // 2,
            0 : input_height + self.band_length - 2,
        ][:, self.band_length_half - 1 : end]
        matrix_g_t = matrix_gb[
            0 : input_width - input_width // 2, 0 : input_width + self.band_length - 2
        ][:, self.band_length_half - 1 : end].t()

        return matrix_h, matrix_h_t, matrix_g, matrix_g_t

    def forward(
        self, LL: torch.Tensor, LH: torch.Tensor, HL: torch.Tensor, HH: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the 2D IDWT module.

        Args:
            LL (torch.Tensor): Input tensor for the low-low band of shape (batch_size, in_channels, height // 2, width // 2).
            LH (torch.Tensor): Input tensor for the low-high band of shape (batch_size, in_channels, height // 2, width // 2).
            HL (torch.Tensor): Input tensor for the high-low band of shape (batch_size, in_channels, height // 2, width // 2).
            HH (torch.Tensor): Input tensor for the high-high band of shape (batch_size, in_channels, height // 2, width // 2).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height, width).
        """
        assert (
            len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4
        ), f"len(LL.size()): {len(LL.size())} != len(LH.size()): {len(LH.size())} != len(HL.size()): {len(HL.size())} != len(HH.size()): {len(HH.size())} != 4"
        input_height = LL.size()[-2] + HH.size()[-2]
        input_width = LL.size()[-1] + HH.size()[-1]
        matrix_h, matrix_h_t, matrix_g, matrix_g_t = self.build_h_g(
            input_height, input_width
        )
        return IDWTFunction_2D.apply(
            LL, LH, HL, HH, matrix_h, matrix_h_t, matrix_g, matrix_g_t
        )
