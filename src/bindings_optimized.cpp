// bindings_optimized.cpp - Python bindings for optimized render kernels
// Exposes multiple kernel variants for benchmarking

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

// Launcher declarations for optimized kernels
extern "C" {
    void launch_render_v1_shared_colors(
        const int8_t* voxels, const float* cameras, float* output,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees, int block_size
    );

    void launch_render_v2_ldg(
        const int8_t* voxels, const float* cameras, float* output,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees, int block_size
    );

    void launch_render_v3_branchless(
        const int8_t* voxels, const float* cameras, float* output,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees, int block_size
    );

    void launch_render_v4_unrolled(
        const int8_t* voxels, const float* cameras, float* output,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees, int block_size
    );

    void launch_render_v5_combined(
        const int8_t* voxels, const float* cameras, float* output,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees, int block_size
    );

    void launch_render_v6_fast_math(
        const int8_t* voxels, const float* cameras, float* output,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees, int block_size
    );

    void launch_render_v7_full(
        const int8_t* voxels, const float* cameras, float* output,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees, int block_size
    );

    void launch_render_v8_unrolled_ldg(
        const int8_t* voxels, const float* cameras, float* output,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees, int block_size
    );

    void launch_render_v9_best(
        const int8_t* voxels, const float* cameras, float* output,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees, int block_size
    );
}

// Template for creating Python wrapper functions
#define MAKE_RENDER_WRAPPER(name, launcher) \
void py_##name( \
    torch::Tensor voxels, \
    torch::Tensor cameras, \
    torch::Tensor output, \
    int max_steps, \
    float view_distance, \
    float fov_degrees, \
    int block_size \
) { \
    TORCH_CHECK(voxels.is_cuda(), "voxels must be CUDA tensor"); \
    TORCH_CHECK(cameras.is_cuda(), "cameras must be CUDA tensor"); \
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor"); \
    TORCH_CHECK(voxels.dtype() == torch::kInt8, "voxels must be int8"); \
    TORCH_CHECK(cameras.dtype() == torch::kFloat32, "cameras must be float32"); \
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32"); \
    \
    int batch_size = voxels.size(0); \
    int world_size = voxels.size(1); \
    int height = output.size(1); \
    int width = output.size(2); \
    \
    launcher( \
        voxels.data_ptr<int8_t>(), \
        cameras.data_ptr<float>(), \
        output.data_ptr<float>(), \
        batch_size, world_size, width, height, \
        max_steps, view_distance, fov_degrees, block_size \
    ); \
}

MAKE_RENDER_WRAPPER(render_v1_shared_colors, launch_render_v1_shared_colors)
MAKE_RENDER_WRAPPER(render_v2_ldg, launch_render_v2_ldg)
MAKE_RENDER_WRAPPER(render_v3_branchless, launch_render_v3_branchless)
MAKE_RENDER_WRAPPER(render_v4_unrolled, launch_render_v4_unrolled)
MAKE_RENDER_WRAPPER(render_v5_combined, launch_render_v5_combined)
MAKE_RENDER_WRAPPER(render_v6_fast_math, launch_render_v6_fast_math)
MAKE_RENDER_WRAPPER(render_v7_full, launch_render_v7_full)
MAKE_RENDER_WRAPPER(render_v8_unrolled_ldg, launch_render_v8_unrolled_ldg)
MAKE_RENDER_WRAPPER(render_v9_best, launch_render_v9_best)

#undef MAKE_RENDER_WRAPPER

PYBIND11_MODULE(_C_opt, m) {
    m.doc() = "mine.cu optimized kernel variants for benchmarking";

    #define ADD_RENDER_FUNC(name, desc) \
        m.def(#name, &py_##name, desc, \
              py::arg("voxels"), py::arg("cameras"), py::arg("output"), \
              py::arg("max_steps") = 64, \
              py::arg("view_distance") = 32.0f, \
              py::arg("fov_degrees") = 70.0f, \
              py::arg("block_size") = 256);

    ADD_RENDER_FUNC(render_v1_shared_colors, "V1: Shared memory for block colors")
    ADD_RENDER_FUNC(render_v2_ldg, "V2: __ldg() for read-only voxel data")
    ADD_RENDER_FUNC(render_v3_branchless, "V3: Branchless DDA stepping")
    ADD_RENDER_FUNC(render_v4_unrolled, "V4: Loop unrolling (4x)")
    ADD_RENDER_FUNC(render_v5_combined, "V5: Combined __ldg + shared colors")
    ADD_RENDER_FUNC(render_v6_fast_math, "V6: Fast math intrinsics + early termination")
    ADD_RENDER_FUNC(render_v7_full, "V7: All optimizations combined")
    ADD_RENDER_FUNC(render_v8_unrolled_ldg, "V8: Unrolled + __ldg")
    ADD_RENDER_FUNC(render_v9_best, "V9: Unrolled + __ldg + fast intrinsics")

    #undef ADD_RENDER_FUNC
}
