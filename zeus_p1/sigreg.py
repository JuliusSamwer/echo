import torch


def sigreg(Z, M=256, knots=17, tmax=3.0):
    """Sketched Epps-Pulley normality test (LeJEPA). Z: [n, D] flattened over batch and time."""
    n, D = Z.shape
    U = torch.randn(D, M, device=Z.device)
    U = U / U.norm(dim=0, keepdim=True)  # resampled every call
    H = Z @ U  # [n, M]
    t = torch.linspace(0, tmax, knots, device=Z.device)  # half-domain, integrand is even
    th = H.unsqueeze(-1) * t  # [n, M, K]
    ecf_re = torch.cos(th).mean(0)
    ecf_im = torch.sin(th).mean(0)  # [M, K]
    target = torch.exp(-0.5 * t ** 2)
    err = (ecf_re - target) ** 2 + ecf_im ** 2
    w = torch.exp(-0.5 * t ** 2)
    stat = torch.trapezoid(err * w, t, dim=-1) * 2  # x2 for the even integrand
    return (n * stat).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    for n in (448, 896):
        v = torch.stack([sigreg(torch.randn(n, 64)) for _ in range(50)])
        print(f"n={n}: sigreg(randn) = {v.mean():.3f} +- {v.std():.3f}")
    Z = torch.randn(896, 64)
    print("scaled x0.5:", sigreg(Z * .5).item(), " x2:", sigreg(Z * 2).item(), " collapsed (const):", sigreg(torch.ones(896, 64)).item(),
          " rank-1:", sigreg(torch.randn(896, 1) @ torch.randn(1, 64)).item(), " uniform:", sigreg((torch.rand(896, 64) - .5) * 12 ** .5).item())
