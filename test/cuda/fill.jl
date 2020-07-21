module test_cuda_fill

using Test
using CUDA

N = 5
x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0

@test x_d isa CuArray{Float32,1,Nothing}
@test Array(x_d) isa Vector{Float32}

function gpu_add(y, n)
    @inbounds for i in 1:length(y)
        y[i] += n
    end
    return nothing
end

@cuda gpu_add(y_d, 1)
@test Array(y_d) == fill(3, N)

end # module test_cuda_fill
