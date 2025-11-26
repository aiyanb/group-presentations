using Random
using CairoMakie

# rng = Xoshiro(125);

Δt = 0.0001
T = 1000.0
nt = round(Int, T/Δt) + 1
b = 0.001
γ = 15.0
n = 2*50

ts = range(0.0,T,nt)

pos = Vector{Float64}(undef, nt)
pos[1] = 0.0
@inbounds @simd for i in 2:nt
    x = pos[i-1]
    θ = (sin(x))^(n-1)
    D = 1+b-θ*sin(x)
    pos[i] = x + (0.5*γ*D-1)*n*θ*cos(x)*Δt + sqrt(2Δt*D)*randn() #rng
end


function msd_traj(θ::AbstractVector{<:Real}; max_length::Integer = 5000)
    n = length(θ)
    n < 2 && return Float64[]

    len = min(n - 1, max_length)
    msd = Vector{Float64}(undef, len)

    @inbounds for τ in 1:len
        s = 0.0
        count = n - τ
        @simd for i in 1:count
            δ = θ[i + τ] - θ[i]
            s += δ * δ
        end
        msd[τ] = s / count
    end

    return msd
end




f = Figure()
ax = Axis(f[1,1])
lines!(ax, ts[1:100:end], pos[1:100:end])
f
