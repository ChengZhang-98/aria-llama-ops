import torch
from ..utils import check_shape


def rms_norm(x: torch.Tensor, w: torch.Tensor, s: int, h: int, var_eps: float):
    """
    Args:
        x: (b, s, h), where b = batch size, s = sequence length, h = hidden size
        w: (h,)
        s: sequence length
        h: hidden size
        var_eps: epsilon for variance

    Returns:
        out: (b, s, h)
    """
    b = x.size(0)
    check_shape(x, (b, s, h))
    check_shape(w, (h,))

    square_bsh = x.pow(2)  # pow = x * x
    check_shape(square_bsh, (b, s, h))
    sum_bs = square_bsh.sum(dim=2)  # sum over hidden_dim
    check_shape(sum_bs, (b, s))
    var_bs = sum_bs / h  # average over hidden_dim
    check_shape(var_bs, (b, s))
    sqrt_bs = torch.sqrt(var_bs + var_eps)  # add epsilon to avoid division by zero
    check_shape(sqrt_bs, (b, s))
    rsqrt_bs = 1.0 / sqrt_bs  # reciprocal of sqrt
    check_shape(rsqrt_bs, (b, s))
    rsqrt_bs1 = rsqrt_bs.unsqueeze(2)  # broadcast to hidden_dim
    check_shape(rsqrt_bs1, (b, s, 1))
    x_norm_bsh = x * rsqrt_bs1  # broadcast multiplication
    check_shape(x_norm_bsh, (b, s, h))
    out = w * x_norm_bsh  # elementwise scale by weight
    check_shape(out, (b, s, h))
    return out
