"""Regression loss measured in the manifold's tangent metric."""


def compute_metric_mse(manifold, x, pred, target, reweight=None):
    """mean_b || reweight_b (pred_b - target_b) ||_{g, x_b}^2.

    For splitting diffusion `pred` is the network output and `target` the DSM
    label of paper Eq. (13); with reweight = sigma_xi(delta) this is the loss of
    paper Algo. 3 (up to the constant sigma^2 of the score convention).
    """
    diff = pred - target
    if reweight is not None:
        diff = reweight * diff
    return manifold.tangent_inner_product(x, diff, diff).mean()
