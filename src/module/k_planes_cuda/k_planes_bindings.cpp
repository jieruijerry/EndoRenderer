#include <torch/extension.h>
#include <cassert>
#include <stdexcept>
#include <string>


#define CONCAT   0
#define MEAN     1
#define PRODUCT  2

#ifndef LEVELS
#define LEVELS 4
#endif

#ifndef DIMENSIONS
#define DIMENSIONS 8
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


void launch_forward(
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& grids,
    torch::Tensor result
);

void launch_backward(
    const torch::Tensor grad_out,
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& grids,
    std::vector<torch::Tensor>& grad_grids,
    torch::Tensor grad_pos
);

torch::Tensor forward(
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& grids
) {
    auto N = pos.size(0);
    auto L = grids.size();
    auto K = grids[0].size(0);
    auto D = grids[0].size(2);
    auto options = grids[0].options();

    if (L != LEVELS || K != K_PLANES || D != DIMENSIONS) {
        throw std::invalid_argument(
            "KPlanes CUDA kernel was built for LEVELS=" + std::to_string(LEVELS) +
            " K=" + std::to_string(K_PLANES) +
            " D=" + std::to_string(DIMENSIONS)
        );
    }
    if (pos.size(1) != N_DIMS) {
        throw std::invalid_argument(
            "KPlanes CUDA kernel expects " + std::to_string(N_DIMS) + "D positions"
        );
    }

    int64_t out_dim = DIMENSIONS;
    if (K_REDUCE == CONCAT) {
        out_dim *= K_PLANES;
    }
    if (LEVEL_REDUCE == CONCAT) {
        out_dim *= LEVELS;
    }
    torch::Tensor result = torch::empty({N, out_dim}, options);

    launch_forward(pos, grids, result);
    return result;
}

std::vector<torch::Tensor> backward(
    const torch::Tensor grad_out,
    const torch::Tensor pos,
    const std::vector<torch::Tensor>& grids
) {
    if (grids.size() != LEVELS || pos.size(1) != N_DIMS) {
        throw std::invalid_argument("KPlanes backward received tensors that do not match the built kernel");
    }
    int64_t out_dim = DIMENSIONS;
    if (K_REDUCE == CONCAT) {
        out_dim *= K_PLANES;
    }
    if (LEVEL_REDUCE == CONCAT) {
        out_dim *= LEVELS;
    }
    if (grad_out.size(0) != pos.size(0) || grad_out.size(1) != out_dim) {
        throw std::invalid_argument("KPlanes backward grad_out has the wrong shape");
    }

    std::vector<torch::Tensor> grad_grids;
    grad_grids.reserve(grids.size());
    for (const auto& grid : grids) {
        grad_grids.push_back(torch::zeros_like(grid));
    }
    torch::Tensor grad_pos = torch::zeros_like(pos);
    launch_backward(grad_out.contiguous(), pos.contiguous(), grids, grad_grids, grad_pos);

    std::vector<torch::Tensor> grads;
    grads.reserve(grad_grids.size() + 1);
    grads.push_back(grad_pos);
    grads.insert(grads.end(), grad_grids.begin(), grad_grids.end());
    return grads;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward,
        py::arg("position"),
        py::arg("grids"),
        "Multi-resolution K-Planes forward (CUDA)"
    );
    m.def("backward", &backward,
        py::arg("grad_output"),
        py::arg("position"),
        py::arg("grids"),
        "Multi-resolution K-Planes backward (CUDA)"
    );
}
