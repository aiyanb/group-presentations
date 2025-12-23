using Random
using CairoMakie

"""
Calculate the MSD of a trajectory (with homogeneous time steps), 
up to `max_length` increments.
"""
function msd_traj(x::AbstractVector; max_length=10000)
    n = length(x)
    n < 2 && return Float64[]
    len = min(n-1, max_length)
    msd = zeros(Float64, len)
    @inbounds @simd for τ in eachindex(msd)
        diffs = @. abs(x[τ+1:end] - x[1:end-τ])
        r2 = diffs .* diffs
        msd[τ] = sum(r2) / length(r2)
    end
    return msd
end


"""
Periodic 1D lattice diffusion with a slow domain and two semipermeable membranes.

- Lattice sites are 1:N (periodic).
- A barrier at b+0.5 means a membrane between sites b and b+1.
- When a step attempts to cross a membrane, it succeeds with probability λ,
  otherwise it reflects (direction flips).
- Time step depends on current site (before the move), matching your original logic.

Returns: (times, positions)
"""
function simulate_membrane_walk(; rng=Xoshiro(1234),
    N::Int=1000,
    barriers::Tuple{Float64,Float64}=(450.5, 549.5),
    λ::Float64=0.3,
    Δt_fast::Float64=5e-7,
    Δt_slow::Float64=5e-6,
    T::Float64=1.0,
    store_every::Int=1
)
    b1 = floor(Int, barriers[1])
    b2 = floor(Int, barriers[2])

    dt_site = fill(Δt_fast, N)
    dt_site[b1+1:b2] .= Δt_slow

    cross_right = falses(N); cross_left = falses(N)
    cross_right[b1] = true;  cross_left[b1 + 1] = true
    cross_right[b2] = true;  cross_left[b2 + 1] = true

    xwrap = rand(rng, 1:N) # periodic cell coordinate
    xZ = xwrap # unwrapped integer coordinate
    t = 0.0

    posZ = Int[];  times = Float64[]
    push!(posZ, xZ);  push!(times, t)

    stepcount = 0
    while t < T
        t += dt_site[xwrap]

        dir = rand(rng, Bool) ? 1 : -1

        # membrane accept/reflect based on attempted crossing in the *cell*
        if (dir == 1 && cross_right[xwrap]) || (dir == -1 && cross_left[xwrap])
            if rand(rng) ≥ λ
                dir = -dir
            end
        end

        # update wrapped and unwrapped coordinates
        xZ += dir
        xwrap += dir
        if xwrap == 0
            xwrap = N
        elseif xwrap == N + 1
            xwrap = 1
        end

        stepcount += 1
        if stepcount % store_every == 0
            push!(posZ, xZ)
            push!(times, t)
        end
    end

    return times, posZ
end


"""Sample a trajectory at equal time intervals."""
function temporally_sample(times::AbstractVector{T}, positions::AbstractVector{P};
                           Δ::Real = 1e-4) where {T<:Real,P}
    n = length(times)
    n == 0 && return T[], P[]
    @assert length(positions) == n
    ΔT = T(Δ)
    @assert ΔT > 0

    t0 = times[1]
    period = times[end]
    ns = Int(floor((period - t0)/ΔT)) + 1  # number of samples

    out_t = Vector{T}(undef, ns)
    out_x = Vector{P}(undef, ns)

    idx = 1
    x = positions[1]

    target = t0
    @inbounds for k in 1:ns
        while idx < n && times[idx + 1] <= target
            idx += 1
            x = positions[idx]
        end
        out_t[k] = target 
        out_x[k] = x
        target += ΔT
    end

    return out_t, out_x
end

function occupation_fraction(times::AbstractVector{T}, positions::AbstractVector{P};
                             barriers::Tuple{Float64,Float64}=(450.5, 549.5)) where {T<:Real,P<:Integer}
    n = length(times)
    n < 2 && return (T(0), T(0))
    @assert length(positions) == n

    b1 = floor(Int, barriers[1])
    b2 = floor(Int, barriers[2])
    slow_lo, slow_hi = b1 + 1, b2

    slow_time = zero(T)
    fast_time = zero(T)

    @inbounds for i in 1:(n-1)
        dt = times[i+1] - times[i]
        x = positions[i]
        if slow_lo <= x && x <= slow_hi
            slow_time += dt
        else
            fast_time += dt
        end
    end

    total = slow_time + fast_time
    total == 0 && return T(0)
    return slow_time/total
end



rng = Xoshiro(1234)
N = 1000

num_probs = 99
num_trajs = 25
λs = range(0.01, 1e-2*num_probs, num_probs)
fractions = Vector{Float64}(undef, num_probs)
for i in eachindex(fractions)
    _fracs = Vector{Float64}(undef, num_trajs)
    for j in eachindex(_fracs)
        times, positions = simulate_membrane_walk(; rng, N, λ=λs[i], T=1.0, store_every=10)
        _fracs[j] = occupation_fraction(times, positions)
    end
    fractions[i] = sum(_fracs) / num_trajs
end

# sampled_times, sampled_positions = temporally_sample(times, positions)
# msd = msd_traj(sampled_positions)

# plotting on the surface of a cylinder
# angles = (2π / N) .* positions
# radius = 1
# X = @. radius * cos(angles)
# Y = @. radius * sin(angles)
# lines(X,Y, times ./ times[end])