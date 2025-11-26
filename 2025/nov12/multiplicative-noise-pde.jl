using LinearAlgebra
using CairoMakie

Δx = 0.01
Δt = 0.2 * Δx^2
T = 1.0
b = 0.1

nx = round(Int, π/Δx) + 1
nt = round(Int, T/Δt) + 1
xs = range(0,π,nx)
ts = range(0,T,nt)

results = Matrix{Float64}(undef, nx, nt)

@simd for x in axes(results, 1)
    results[x,1] = 0.0
end
results[round(Int,nx/2)+1,1] = 1.0 # delta initial condition

r = Δt / Δx
s = Δt / (Δx)^2

@inbounds for t in 2:nt
    @simd for x in 2:(nx-1)
        xprev = Δx * (x - 1);  uprev = results[x,t-1]
        results[x,t] = uprev - r * 10 * sin(xprev)^9 * cos(xprev) * (results[x+1,t-1] - uprev) + s * (1 - sin(xprev)^10 + b) * (results[x+1,t-1] - 2*uprev + results[x-1,t-1])
    end

    # periodic BC
    uprev = results[1,t-1]
    results[1,t] = uprev + s * (1 + b) * (results[2,t-1] - 2*uprev + results[end,t-1])

    xprev = Δx * (nx - 1)
    uprev = results[end,t-1]
    results[end,t] = uprev - r * 10 * sin(xprev)^9 * cos(xprev) * (results[1,t-1] - uprev) + s * (1 - sin(xprev)^10 + b) * (results[1,t-1] - 2*uprev + results[end-1,t-1])
end

f = Figure()
ax = Axis(f[1,1])
heatmap!(ax, xs, ts[1000:end], results[:,1000:end])
f