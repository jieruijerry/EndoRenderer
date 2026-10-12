#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>


#define CONCAT   0
#define MEAN     1
#define PRODUCT  2

#define PRIME_U 1
#define PRIME_V 19349663

#ifndef LEVELS
#define LEVELS 4
#endif

#ifndef DIMENSIONS
#define DIMENSIONS 8
#endif

#ifndef LOG_HASHMAP_SIZE
#define LOG_HASHMAP_SIZE 19
#endif

#ifndef BASE_RESOLUTION
#define BASE_RESOLUTION 32
#endif

#ifndef PER_LEVEL_SCALE
#define PER_LEVEL_SCALE 2.0
#endif

#ifndef N_DIMS
#define N_DIMS 4
#endif

#ifndef K_REDUCE
#define K_REDUCE PRODUCT
#endif

#ifndef LEVEL_REDUCE
#define LEVEL_REDUCE CONCAT
#endif

#define K_PLANES (N_DIMS * (N_DIMS - 1) / 2)
#define HASHMAP_SIZE (1 << LOG_HASHMAP_SIZE)


__constant__ int resolutions[LEVELS];
__constant__ int grid_sizes[LEVELS];
__constant__ int plane_axis0[K_PLANES];
__constant__ int plane_axis1[K_PLANES];


__device__ __forceinline__ int _plane_index(int u, int v, int level) {
    int res1 = resolutions[level] + 1;
    int grid_size = grid_sizes[level];
    int64_t raw;
    if (static_cast<int64_t>(res1) * res1 > HASHMAP_SIZE) {
        raw = static_cast<int64_t>(u) * PRIME_U + static_cast<int64_t>(v) * PRIME_V;
        raw %= HASHMAP_SIZE;
    } else {
        raw = static_cast<int64_t>(res1) * u + v;
    }
    raw %= grid_size;
    if (raw < 0) {
        raw += grid_size;
    }
    return static_cast<int>(raw);
}


template <typename scalar_t>
__global__ void forward_kernel(
    const scalar_t* __restrict__ pos,
    const scalar_t** __restrict__ grids,
    scalar_t* __restrict__ result,
    int N
) {
    int sample = blockIdx.x;
    int tid = threadIdx.x;
    if (sample >= N) {
        return;
    }

    extern __shared__ char dyn_shared[];
    scalar_t* acc = reinterpret_cast<scalar_t*>(dyn_shared);
    int acc_size = LEVELS * K_PLANES * DIMENSIONS;
    for (int i = tid; i < acc_size; i += blockDim.x) {
        acc[i] = static_cast<scalar_t>(0);
    }
    __syncthreads();

    const scalar_t* pos_ptr = pos + sample * N_DIMS;
    int work = LEVELS * K_PLANES * 4;
    for (int job = tid; job < work; job += blockDim.x) {
        int corner = job & 3;
        int plane = (job >> 2) % K_PLANES;
        int level = job / (K_PLANES * 4);

        int res = resolutions[level];
        scalar_t pu = pos_ptr[plane_axis0[plane]] * static_cast<scalar_t>(res);
        scalar_t pv = pos_ptr[plane_axis1[plane]] * static_cast<scalar_t>(res);
        int bu = static_cast<int>(floorf(static_cast<float>(pu)));
        int bv = static_cast<int>(floorf(static_cast<float>(pv)));
        scalar_t ou = pu - static_cast<scalar_t>(bu);
        scalar_t ov = pv - static_cast<scalar_t>(bv);

        int iu = bu + (corner & 1);
        int iv = bv + (corner >> 1);
        scalar_t weight = ((corner & 1) ? ou : (static_cast<scalar_t>(1) - ou)) *
                          ((corner >> 1) ? ov : (static_cast<scalar_t>(1) - ov));

        int index = _plane_index(iu, iv, level);
        const scalar_t* grid = grids[level] + (static_cast<int64_t>(plane) * grid_sizes[level] + index) * DIMENSIONS;
        scalar_t* slot = acc + (level * K_PLANES + plane) * DIMENSIONS;
        for (int d = 0; d < DIMENSIONS; ++d) {
            atomicAdd(slot + d, grid[d] * weight);
        }
    }
    __syncthreads();

    // k_reduce within each level, then level_reduce into the output row.
    int level_width = (K_REDUCE == CONCAT) ? K_PLANES * DIMENSIONS : DIMENSIONS;
    int out_width = (LEVEL_REDUCE == CONCAT) ? LEVELS * level_width : level_width;
    scalar_t* out = result + sample * out_width;

    for (int i = tid; i < out_width; i += blockDim.x) {
#if (LEVEL_REDUCE == CONCAT)
        int level = i / level_width;
        int inner = i % level_width;
        scalar_t value = static_cast<scalar_t>(0);
#if (K_REDUCE == CONCAT)
        int plane = inner / DIMENSIONS;
        int feat = inner % DIMENSIONS;
        value = acc[(level * K_PLANES + plane) * DIMENSIONS + feat];
#elif (K_REDUCE == MEAN)
        int feat = inner;
        value = static_cast<scalar_t>(0);
        for (int plane = 0; plane < K_PLANES; ++plane) {
            value += acc[(level * K_PLANES + plane) * DIMENSIONS + feat];
        }
        value /= static_cast<scalar_t>(K_PLANES);
#elif (K_REDUCE == PRODUCT)
        int feat = inner;
        value = static_cast<scalar_t>(1);
        for (int plane = 0; plane < K_PLANES; ++plane) {
            value *= acc[(level * K_PLANES + plane) * DIMENSIONS + feat];
        }
#endif
        out[i] = value;
#elif (LEVEL_REDUCE == MEAN)
        int inner = i;
        scalar_t value = static_cast<scalar_t>(0);
        for (int level = 0; level < LEVELS; ++level) {
            scalar_t level_value;
#if (K_REDUCE == CONCAT)
            int plane = inner / DIMENSIONS;
            int feat = inner % DIMENSIONS;
            level_value = acc[(level * K_PLANES + plane) * DIMENSIONS + feat];
#elif (K_REDUCE == MEAN)
            int feat = inner;
            level_value = static_cast<scalar_t>(0);
            for (int plane = 0; plane < K_PLANES; ++plane) {
                level_value += acc[(level * K_PLANES + plane) * DIMENSIONS + feat];
            }
            level_value /= static_cast<scalar_t>(K_PLANES);
#elif (K_REDUCE == PRODUCT)
            int feat = inner;
            level_value = static_cast<scalar_t>(1);
            for (int plane = 0; plane < K_PLANES; ++plane) {
                level_value *= acc[(level * K_PLANES + plane) * DIMENSIONS + feat];
            }
#endif
            value += level_value;
        }
        out[i] = value / static_cast<scalar_t>(LEVELS);
#endif
    }
}


template <typename scalar_t>
const scalar_t** tensor_list_to_device_ptrs(const std::vector<torch::Tensor>& tensors) {
    size_t n = tensors.size();
    std::vector<const scalar_t*> host_ptrs(n);
    for (size_t i = 0; i < n; ++i) {
        host_ptrs[i] = tensors[i].data_ptr<scalar_t>();
    }
    const scalar_t** device_ptrs;
    cudaMalloc(&device_ptrs, sizeof(const scalar_t*) * n);
    cudaMemcpy(device_ptrs, host_ptrs.data(), sizeof(const scalar_t*) * n, cudaMemcpyHostToDevice);
    return device_ptrs;
}


void launch_forward(
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& grids,
    torch::Tensor result
) {
    int axis0[K_PLANES];
    int axis1[K_PLANES];
    int pair = 0;
    for (int i = 0; i < N_DIMS; ++i) {
        for (int j = i + 1; j < N_DIMS; ++j) {
            axis0[pair] = i;
            axis1[pair] = j;
            ++pair;
        }
    }
    cudaMemcpyToSymbol(plane_axis0, axis0, sizeof(int) * K_PLANES);
    cudaMemcpyToSymbol(plane_axis1, axis1, sizeof(int) * K_PLANES);

    int res_table[LEVELS];
    int grid_table[LEVELS];
    for (int i = 0; i < LEVELS; ++i) {
        int res = static_cast<int>(BASE_RESOLUTION * std::pow(PER_LEVEL_SCALE, i));
        int64_t dense = static_cast<int64_t>(res + 1) * (res + 1);
        res_table[i] = res;
        grid_table[i] = static_cast<int>(dense > HASHMAP_SIZE ? HASHMAP_SIZE : dense);
    }
    cudaMemcpyToSymbol(resolutions, res_table, sizeof(int) * LEVELS);
    cudaMemcpyToSymbol(grid_sizes, grid_table, sizeof(int) * LEVELS);

    int N = pos.size(0);
    int work = LEVELS * K_PLANES * 4;
    int n_threads = work < 256 ? work : 256;
    int shared_bytes = LEVELS * K_PLANES * DIMENSIONS * pos.element_size();

    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "k_planes_forward", ([&] {
        const scalar_t** device_ptrs = tensor_list_to_device_ptrs<scalar_t>(grids);
        forward_kernel<scalar_t><<<N, n_threads, shared_bytes>>>(
            pos.data_ptr<scalar_t>(),
            device_ptrs,
            result.data_ptr<scalar_t>(),
            N
        );
        cudaFree(device_ptrs);
    }));
}


template <typename scalar_t>
scalar_t** tensor_list_to_device_mut_ptrs(const std::vector<torch::Tensor>& tensors) {
    size_t n = tensors.size();
    std::vector<scalar_t*> host_ptrs(n);
    for (size_t i = 0; i < n; ++i) {
        host_ptrs[i] = tensors[i].data_ptr<scalar_t>();
    }
    scalar_t** device_ptrs;
    cudaMalloc(&device_ptrs, sizeof(scalar_t*) * n);
    cudaMemcpy(device_ptrs, host_ptrs.data(), sizeof(scalar_t*) * n, cudaMemcpyHostToDevice);
    return device_ptrs;
}


template <typename scalar_t>
__device__ void reduced_feature_grad(
    const scalar_t* acc,
    const scalar_t* grad_out,
    int level,
    int plane,
    scalar_t grad_feat[DIMENSIONS]
) {
    constexpr scalar_t level_scale = (LEVEL_REDUCE == MEAN)
        ? static_cast<scalar_t>(1) / static_cast<scalar_t>(LEVELS)
        : static_cast<scalar_t>(1);
    constexpr int level_width = (K_REDUCE == CONCAT) ? K_PLANES * DIMENSIONS : DIMENSIONS;

    for (int feat = 0; feat < DIMENSIONS; ++feat) {
        scalar_t upstream;
#if (LEVEL_REDUCE == CONCAT)
        int out_index = level * level_width;
#else
        int out_index = 0;
#endif
#if (K_REDUCE == CONCAT)
        upstream = grad_out[out_index + plane * DIMENSIONS + feat];
#elif (K_REDUCE == MEAN)
        upstream = grad_out[out_index + feat] / static_cast<scalar_t>(K_PLANES);
#elif (K_REDUCE == PRODUCT)
        scalar_t others = static_cast<scalar_t>(1);
        for (int other = 0; other < K_PLANES; ++other) {
            if (other == plane) {
                continue;
            }
            others *= acc[(level * K_PLANES + other) * DIMENSIONS + feat];
        }
        upstream = grad_out[out_index + feat] * others;
#endif
        grad_feat[feat] = upstream * level_scale;
    }
}


template <typename scalar_t>
__global__ void backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ pos,
    const scalar_t** __restrict__ grids,
    scalar_t** __restrict__ grad_grids,
    scalar_t* __restrict__ grad_pos,
    int N
) {
    int sample = blockIdx.x;
    int tid = threadIdx.x;
    if (sample >= N) {
        return;
    }

    extern __shared__ char dyn_shared[];
    scalar_t* acc = reinterpret_cast<scalar_t*>(dyn_shared);
    scalar_t* grad_pos_s = acc + LEVELS * K_PLANES * DIMENSIONS;
    int acc_size = LEVELS * K_PLANES * DIMENSIONS;
    for (int i = tid; i < acc_size; i += blockDim.x) {
        acc[i] = static_cast<scalar_t>(0);
    }
    for (int i = tid; i < N_DIMS; i += blockDim.x) {
        grad_pos_s[i] = static_cast<scalar_t>(0);
    }
    __syncthreads();

    const scalar_t* pos_ptr = pos + sample * N_DIMS;
    const scalar_t* grad_ptr = grad_out + sample * (
        ((LEVEL_REDUCE == CONCAT) ? LEVELS : 1) *
        ((K_REDUCE == CONCAT) ? K_PLANES * DIMENSIONS : DIMENSIONS)
    );
    int work = LEVELS * K_PLANES * 4;

    for (int job = tid; job < work; job += blockDim.x) {
        int corner = job & 3;
        int plane = (job >> 2) % K_PLANES;
        int level = job / (K_PLANES * 4);
        int res = resolutions[level];
        scalar_t pu = pos_ptr[plane_axis0[plane]] * static_cast<scalar_t>(res);
        scalar_t pv = pos_ptr[plane_axis1[plane]] * static_cast<scalar_t>(res);
        int bu = static_cast<int>(floorf(static_cast<float>(pu)));
        int bv = static_cast<int>(floorf(static_cast<float>(pv)));
        scalar_t ou = pu - static_cast<scalar_t>(bu);
        scalar_t ov = pv - static_cast<scalar_t>(bv);
        int iu = bu + (corner & 1);
        int iv = bv + (corner >> 1);
        scalar_t weight = ((corner & 1) ? ou : (static_cast<scalar_t>(1) - ou)) *
                          ((corner >> 1) ? ov : (static_cast<scalar_t>(1) - ov));
        int index = _plane_index(iu, iv, level);
        const scalar_t* grid = grids[level] + (static_cast<int64_t>(plane) * grid_sizes[level] + index) * DIMENSIONS;
        scalar_t* slot = acc + (level * K_PLANES + plane) * DIMENSIONS;
        for (int d = 0; d < DIMENSIONS; ++d) {
            atomicAdd(slot + d, grid[d] * weight);
        }
    }
    __syncthreads();

    for (int job = tid; job < work; job += blockDim.x) {
        int corner = job & 3;
        int plane = (job >> 2) % K_PLANES;
        int level = job / (K_PLANES * 4);
        int res = resolutions[level];
        int axis_u = plane_axis0[plane];
        int axis_v = plane_axis1[plane];
        scalar_t pu = pos_ptr[axis_u] * static_cast<scalar_t>(res);
        scalar_t pv = pos_ptr[axis_v] * static_cast<scalar_t>(res);
        int bu = static_cast<int>(floorf(static_cast<float>(pu)));
        int bv = static_cast<int>(floorf(static_cast<float>(pv)));
        scalar_t ou = pu - static_cast<scalar_t>(bu);
        scalar_t ov = pv - static_cast<scalar_t>(bv);
        scalar_t wu = (corner & 1) ? ou : (static_cast<scalar_t>(1) - ou);
        scalar_t wv = (corner >> 1) ? ov : (static_cast<scalar_t>(1) - ov);
        scalar_t weight = wu * wv;
        scalar_t dw_du = ((corner & 1) ? static_cast<scalar_t>(1) : static_cast<scalar_t>(-1)) * wv;
        scalar_t dw_dv = ((corner >> 1) ? static_cast<scalar_t>(1) : static_cast<scalar_t>(-1)) * wu;

        int index = _plane_index(bu + (corner & 1), bv + (corner >> 1), level);
        const scalar_t* grid = grids[level] + (static_cast<int64_t>(plane) * grid_sizes[level] + index) * DIMENSIONS;
        scalar_t* grad_grid = grad_grids[level] + (static_cast<int64_t>(plane) * grid_sizes[level] + index) * DIMENSIONS;

        scalar_t grad_feat[DIMENSIONS];
        reduced_feature_grad(acc, grad_ptr, level, plane, grad_feat);

        scalar_t pos_u = static_cast<scalar_t>(0);
        scalar_t pos_v = static_cast<scalar_t>(0);
        for (int d = 0; d < DIMENSIONS; ++d) {
            atomicAdd(grad_grid + d, grad_feat[d] * weight);
            pos_u += grad_feat[d] * dw_du * grid[d];
            pos_v += grad_feat[d] * dw_dv * grid[d];
        }
        atomicAdd(grad_pos_s + axis_u, pos_u * static_cast<scalar_t>(res));
        atomicAdd(grad_pos_s + axis_v, pos_v * static_cast<scalar_t>(res));
    }
    __syncthreads();

    scalar_t* grad_pos_row = grad_pos + sample * N_DIMS;
    for (int i = tid; i < N_DIMS; i += blockDim.x) {
        grad_pos_row[i] = grad_pos_s[i];
    }
}


void launch_backward(
    const torch::Tensor grad_out,
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& grids,
    std::vector<torch::Tensor>& grad_grids,
    torch::Tensor grad_pos
) {
    int axis0[K_PLANES];
    int axis1[K_PLANES];
    int pair = 0;
    for (int i = 0; i < N_DIMS; ++i) {
        for (int j = i + 1; j < N_DIMS; ++j) {
            axis0[pair] = i;
            axis1[pair] = j;
            ++pair;
        }
    }
    cudaMemcpyToSymbol(plane_axis0, axis0, sizeof(int) * K_PLANES);
    cudaMemcpyToSymbol(plane_axis1, axis1, sizeof(int) * K_PLANES);

    int res_table[LEVELS];
    int grid_table[LEVELS];
    for (int i = 0; i < LEVELS; ++i) {
        int res = static_cast<int>(BASE_RESOLUTION * std::pow(PER_LEVEL_SCALE, i));
        int64_t dense = static_cast<int64_t>(res + 1) * (res + 1);
        res_table[i] = res;
        grid_table[i] = static_cast<int>(dense > HASHMAP_SIZE ? HASHMAP_SIZE : dense);
    }
    cudaMemcpyToSymbol(resolutions, res_table, sizeof(int) * LEVELS);
    cudaMemcpyToSymbol(grid_sizes, grid_table, sizeof(int) * LEVELS);

    int N = pos.size(0);
    int work = LEVELS * K_PLANES * 4;
    int n_threads = work < 256 ? work : 256;
    int shared_floats = LEVELS * K_PLANES * DIMENSIONS + N_DIMS;
    int shared_bytes = shared_floats * pos.element_size();

    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "k_planes_backward", ([&] {
        const scalar_t** grid_ptrs = tensor_list_to_device_ptrs<scalar_t>(grids);
        scalar_t** grad_ptrs = tensor_list_to_device_mut_ptrs<scalar_t>(grad_grids);
        backward_kernel<scalar_t><<<N, n_threads, shared_bytes>>>(
            grad_out.data_ptr<scalar_t>(),
            pos.data_ptr<scalar_t>(),
            grid_ptrs,
            grad_ptrs,
            grad_pos.data_ptr<scalar_t>(),
            N
        );
        cudaFree(grid_ptrs);
        cudaFree(grad_ptrs);
    }));
}
