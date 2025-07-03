using DistributionsHEP
using Plots
theme(:boxed)

params = [
    (0.0, 1.0, 1.5, 2.0, 2.0, 3.0),
    (0.0, 1.0, 2.0, 3.0, 2.0, 3.0),
    (0.0, 1.0, 1.0, 2.0, 1.0, 2.0), # symmetric
    (0.0, 1.0, 0.5, 1.5, 3.0, 5.0),
    (0.0, 0.5, 0.5, 1.5, 3.0, 5.0),
]

x_range = -7:0.01:7

pv = map(enumerate(params)) do (i, (μ, σ, αL, nL, αR, nR))
    d = DoubleCrystalBall(μ, σ, αL, nL, αR, nR)
    p1 = plot(
        x_range,
        x -> pdf(d, x),
        label = "PDF",
        xlabel = "x",
        ylabel = "PDF",
        linewidth = 2,
    )
    # 
    scatter!(
        [-αL * σ + μ, αR * σ + μ],
        [pdf(d, -αL * σ + μ), pdf(d, αR * σ + μ)],
        label = "Transitions",
        markersize = 6,
        color = :red,
    )

    p2 = plot(
        x_range,
        x -> cdf(d, x),
        label = "CDF",
        xlabel = "x",
        ylabel = "CDF",
        linewidth = 2,
        ylims = (0, 1),
    )

    p3 = plot(
        range(0, 1, 100)[2:end-1],
        x -> quantile(d, x),
        label = "Quantile",
        xlabel = "cdf",
        ylabel = "x",
        linewidth = 2,
        ylims = (-7, 7),
    )

    plot(p1, p2, p3, layout = (1, 3))
end

plot(pv..., layout = (5, 1), size = (900, 1500))
