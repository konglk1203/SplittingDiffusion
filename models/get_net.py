"""Score / vector-field network construction."""
import torch


def build(opt, input_dim, out_dim, manifold=None):
    if opt.net_name == 'dit':
        from .dit import build_dit
        net = build_dit(opt, input_dim, out_dim, manifold)
    else:
        from .resnet import build_resnet
        net = build_resnet(opt, input_dim, out_dim)
    return NetworkWrapper(opt, net).to(opt.device)


class NetworkWrapper(torch.nn.Module):
    """Common calling convention net(x, v, t) -> flat tangent vector.

    * x: (B, dim_x) flat points; v: (B, dim_v) flat velocities or None (RFM);
      t: scalar or (B,) times.
    * For splitting diffusion the output is the sigma^2-scaled vertical score
      (see ``dynamics/splitting_sde.py``).  With ``score_parametrization=
      'residual_v'`` the output is f(x, v, t) - v: the stationary score -v is
      then the zero function of the inner network f, which only has to learn
      the deviation from stationarity.
    """

    def __init__(self, opt, net):
        super().__init__()
        self.opt = opt
        self.net = net

    def forward(self, x, v, t):
        bs = x.shape[0]
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        if t.dim() == 0:
            t = t.repeat(bs)
        assert t.dim() == 1 and t.shape[0] == bs
        if v is not None:
            v = v.reshape(bs, -1)
        out = self.net(x, v, t)
        if self.opt.algo == 'splitting' and v is not None and self.opt.score_parametrization == 'residual_v':
            out = out - v
        return out
