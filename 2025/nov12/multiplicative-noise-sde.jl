using Random
using CairoMakie

rng = Xoshiro(124);

Δt = 0.0001
T = 200.0
nt = round(Int, T/Δt) + 1
b = 0.1
n = 10

ts = range(0.0,T,nt)

pos = Vector{Float64}(undef, nt)
pos[1] = 0.0
@simd for i in 2:nt
    x = pos[i-1]
    θ = (sin(x))^(n-1)
    pos[i] = x - 0.5*n*θ*cos(x)*Δt + sqrt(2Δt*(1-θ*sin(x) + b))*randn(rng)
end

mpos = mod.(pos, π)
mask = asin((1/2)^(1/n)) .< mpos .< π-asin((1/2)^(1/n))



function msd_traj(θ::AbstractVector; max_length=5000) #allow for max len for faster computation
    n = length(θ)
    n < 2 && return Float64[]
    len = min(n-1, max_length)
    msd = zeros(Float64, len)
    @inbounds for τ in eachindex(msd)
        diffs = mod.(abs.(θ[τ+1:end] .- θ[1:end-τ]), π)
        r2 = diffs .* diffs
        msd[τ] = sum(r2) / length(r2)
    end
    return msd
end

function split_by_mask(θ::AbstractVector, mask::AbstractVector{Bool})
    segments_high = Vector{Vector{Float64}}()
    segments_low  = Vector{Vector{Float64}}()

    current = Float64[]
    current_mask = mask[1]
    for (ang, m) in zip(θ, mask)
        if m == current_mask
            push!(current, ang)
        else
            # close previous segment
            if current_mask
                push!(segments_high, current)
            else
                push!(segments_low, current)
            end
            # start new segment
            current = [ang]
            current_mask = m
        end
    end
    # push last segment
    if current_mask
        push!(segments_high, current)
    else
        push!(segments_low, current)
    end

    return segments_high, segments_low
end

function aggregate_msds(segments::Vector{Vector{Float64}})
    # find longest segment length
    maxlen = maximum(length.(segments))
    maxlen < 2 && return Float64[]

    sum_msd = zeros(Float64, maxlen-1)
    counts  = zeros(Int, maxlen-1)

    for seg in segments
        msd_seg = msd_traj(seg)
        for τ in 1:length(msd_seg)
            sum_msd[τ] += msd_seg[τ]
            counts[τ]  += 1
        end
    end

    msd = similar(sum_msd)
    for τ in 1:length(msd)
        msd[τ] = counts[τ] > 0 ? sum_msd[τ] / counts[τ] : NaN
    end
    return msd
end



low_mpos = mpos[mask]
high_mpos = mpos[.!mask]

low_segs, high_segs = split_by_mask(mpos, mask)
msd_high = aggregate_msds(high_segs)
msd_low = aggregate_msds(low_segs)

skip = 10

f_high = Figure()
ax_high = Axis(f_high[1,1])
num_msd_high = length(msd_high[1:skip:end])
ts_high = range(0, Δt*skip*(num_msd_high-1), num_msd_high)
scatter!(ax_high, ts_high, msd_high[1:skip:end])
f_high

f_low = Figure()
ax_low = Axis(f_low[1,1])
num_msd_low = length(msd_low[1:skip:end])
ts_low = range(0, Δt*skip*(num_msd_low-1), num_msd_low)
scatter!(ax_low, ts_low, msd_low[1:skip:end])
f_low



# f = Figure()
# ax = Axis(f[1,1])
# scatter!(ax, ts[1:100:end], mpos[1:100:end])
# f

# x = range(0,π,1000)
# f = Figure()
# ax = Axis(f[1,1])

# lines!(ax, x, @. 1 - (sin(x))^10 + b)
# save("diffusivity3.pdf", f)


# x = range(0,π,1000)

# f = Figure()
# ax = Axis(f[1,1])

# lines!(ax, x, @. sin(x)*sin(x)+b)
# save("diffusivity.pdf", f)