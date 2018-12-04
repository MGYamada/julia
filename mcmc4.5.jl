using MPI
using SparseArrays
using LinearAlgebra
using Statistics
const Jx = 1 / 3
const Jy = 1 / 3
const Jz = 1 / 3
function Metropolis(βF::Float64, βFnew::Float64)::Bool
    βF - βFnew > log(rand())
end
function simpleMetropolis(Δlogp::Float64)::Bool
    Δlogp > log(rand())
end
function openhoneycomb(Lx::Int64, Ly::Int64)::Tuple
    N = 2Lx * Ly
    nnx = zip(1 : 2 : (N - 1), 2 : 2 : N)
    nny = Iterators.flatten((zip((1 + 2i) : 2Lx : (2Lx * (Ly - 1)  + 1 + 2i), 2i : 2Lx : (2Lx * (Ly - 1)  + 2i)) for i in 1 : (Lx - 1)))
    nnz = zip(1 : 2 : (N - 1), Iterators.flatten(((2Lx + 2) : 2 : N, 2 : 2 : 2Lx)))
    plaquette = Iterators.flatten(zip((Lx * (i - 1) + 1) : (Lx * (i - 1) + Lx - 1), (Lx * (i - 1) + 2) : (Lx * (i - 1) + Lx)) for i in 1 : Ly)
    N, nnx, nny, nnz, plaquette
end
function freeenergy(β₂::Float64)::Function
    ϵ -> log(exp(β₂ * ϵ) + exp(-β₂ * ϵ))
end
function energy(β₂::Float64)::Function
    ϵ -> ϵ * tanh(β₂ * ϵ)
end
function denergy(β₂::Float64)::Function
    ϵ -> (ϵ * sech(β₂ * ϵ)) ^ 2
end
function leaveoneout(before::Function, after::Function, v::AbstractVector)
    ind = eachindex(v)
    map(i -> after(mean(map(before, view(v, filter(!isequal(i), ind))))), ind)
end
meanJ(b::Function, a::Function, v::AbstractVector) = mean(leaveoneout(b, a, v))
stdmJ(b::Function, a::Function, v::AbstractVector, m) = stdm(leaveoneout(b, a, v), m, corrected = false) * sqrt(length(v) - 1)
TTCv(v::Vector{Float64}) = [v[1] ^ 2 - v[2], v[1]]
function Cv(β::Float64)::Function
    β² = β ^ 2
    meanTTCv::Vector{Float64} -> β² * (meanTTCv[1] - meanTTCv[2] ^ 2)
end
function measurement(comm::MPI.Comm, rank::Int64, method::Function, lattice::Function, β::Float64, Lx::Int64, Ly::Int64)::Channel
    Channel(ctype = Vector{Float64}) do channel::Channel{Vector{Float64}}
        N, nnx, nny, nnz, plaquette = lattice(Lx, Ly)
        iter = Iterators.flatten((Iterators.product(J, nn) for (J, nn) in [(Jx, nnx), (Jy, nny), (Jz, nnz)]))
        h = spzeros(Complex{Float64}, N, N)
        for (J, nn) in iter
            h[nn[1], nn[2]] = 0.5im * J
            h[nn[2], nn[1]] = -0.5im * J
        end
        NNz = collect(nnz)
        Nz = length(NNz)
        η = ones(Int64, Nz) # BitArray does not work in MPI
        ηnew = similar(η)
        βF = 0.0
        β₂ = β * 0.5
        hdense = Array(h)
        posev = zeros(Float64, N >> 1)
        step = 0
        while true
            MPI.Barrier(comm) # for your safety
            if rank == 0
                MPI.Send(η, 1, 4step, comm)
                MPI.Recv!(ηnew, 1, 4step + 1, comm)
            elseif rank == 1
                MPI.Recv!(ηnew, 0, 4step, comm)
                MPI.Send(η, 0, 4step + 1, comm)
            end
            for (j, σ) in enumerate(ηnew)
                h[NNz[j][1], NNz[j][2]] = 0.5im * Jz * σ
                h[NNz[j][2], NNz[j][1]] = -0.5im * Jz * σ
            end
            evnew = eigvals(Hermitian(Array(h)))
            iter = Iterators.drop(evnew, N >> 1)
            βFnew = -mapreduce(freeenergy(β₂), +, iter)
            MPI.Barrier(comm) # for your safety
            if rank == 0
                MPI.Send([βF - βFnew], 1, 4step + 2, comm)
                isaccepted = [0]
                MPI.Recv!(isaccepted, 1, 4step + 3, comm)
                if isaccepted[1] == 1
                    η .= ηnew
                    βF = βFnew
                    posev = collect(iter)
                    hdense = Array(h)
                end
            elseif rank == 1
                diff = [0.0]
                MPI.Recv!(diff, 0, 4step + 2, comm)
                if simpleMetropolis(βF - βFnew + diff[1])
                    MPI.Send([1], 0, 4step + 3, comm)
                    η .= ηnew
                    βF = βFnew
                    posev = collect(iter)
                    hdense = Array(h)
                else
                    MPI.Send([0], 0, 4step + 3, comm)
                end
            end
            MPI.Barrier(comm) # for your safety

            for i in 1 : Nz
                j = rand(1 : Nz)
                hdense[NNz[j][1], NNz[j][2]] = -hdense[NNz[j][1], NNz[j][2]]
                hdense[NNz[j][2], NNz[j][1]] = -hdense[NNz[j][2], NNz[j][1]]
                evnew = eigvals(Hermitian(hdense))
                iter = Iterators.drop(evnew, N >> 1)
                βFnew = -mapreduce(freeenergy(β₂), +, iter)
                if method(βF, βFnew)
                    η[j] = -η[j]
                    βF = βFnew
                    posev = collect(iter)
                else
                    hdense[NNz[j][1], NNz[j][2]] = -hdense[NNz[j][1], NNz[j][2]]
                    hdense[NNz[j][2], NNz[j][1]] = -hdense[NNz[j][2], NNz[j][1]]
                end
            end
            Ef = -0.5 * mapreduce(energy(β₂), +, posev)
            ∂Ef∂β = -0.25 * mapreduce(denergy(β₂), +, posev)
            put!(channel, [Ef, ∂Ef∂β])
        step += 1
        end
    end
end
function replica(comm::MPI.Comm, rank::Int64, method::Function, lattice::Function, β::Float64, Lx::Int64, Ly::Int64)::Tuple
    mcstep = Iterators.drop(measurement(comm, rank, method, lattice, β, Lx, Ly), 2000)
    iter = Iterators.take(mcstep, 10000)
    data = collect(iter)
    m = meanJ(TTCv, Cv(β), data)
    m, stdmJ(TTCv, Cv(β), data, m)
end

MPI.Init()
comm = MPI.COMM_WORLD
size = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)
# sleep(0.001 * rank)
# println("size = $size, rank = $rank")
if size == 2
    β = [150.0, 300.0][rank + 1]
    m, s = replica(comm, rank, Metropolis, openhoneycomb, β, 4, 4)
    println("rank = $rank, β = $β, Cv(β) = $m ± $s")
else
    println("Execute by mpiexec -np 2 julia mcmc4.5.jl")
end
MPI.Finalize()
