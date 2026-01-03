// mine.cu Python bindings via pybind11
// Exposes CUDA kernels to Python with PyTorch tensor support

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

// Launcher declarations (defined in kernels.cu)
extern "C" {
    void launch_render(
        const int8_t* voxels, const float* cameras, float* output,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees
    );

    void launch_physics(
        float* positions, float* velocities, float* yaws, float* pitches,
        bool* on_ground, const float* forward_in, const float* strafe_in,
        const float* delta_yaw_in, const float* delta_pitch_in, const bool* jump_in,
        int batch_size, int world_size, float dt, float gravity, float walk_speed, float jump_vel
    );

    void launch_raycast_break(
        int8_t* voxels, const float* positions, const float* yaws, const float* pitches,
        const bool* do_break, float* rewards,
        int batch_size, int world_size, int8_t target_block, float reward_value
    );

    void launch_reset(
        float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
        const bool* do_reset, int batch_size, float spawn_x, float spawn_y, float spawn_z
    );

    void launch_generate_world(
        int8_t* voxels, int batch_size, int world_size, int ground_height, unsigned int seed
    );

    void launch_place_tree(
        int8_t* voxels, int batch_size, int world_size, int tree_x, int tree_z, int ground_height
    );
}

// Python-facing wrapper functions with tensor validation

void py_render(
    torch::Tensor voxels,    // [B, Y, X, Z] int8
    torch::Tensor cameras,   // [B, 5] float32
    torch::Tensor output,    // [B, H, W, 3] float32
    int max_steps,
    float view_distance,
    float fov_degrees
) {
    TORCH_CHECK(voxels.is_cuda(), "voxels must be CUDA tensor");
    TORCH_CHECK(cameras.is_cuda(), "cameras must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(voxels.dtype() == torch::kInt8, "voxels must be int8");
    TORCH_CHECK(cameras.dtype() == torch::kFloat32, "cameras must be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be float32");

    int batch_size = voxels.size(0);
    int world_size = voxels.size(1);
    int height = output.size(1);
    int width = output.size(2);

    launch_render(
        voxels.data_ptr<int8_t>(),
        cameras.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void py_physics(
    torch::Tensor positions,    // [B, 3] float32
    torch::Tensor velocities,   // [B, 3] float32
    torch::Tensor yaws,         // [B] float32
    torch::Tensor pitches,      // [B] float32
    torch::Tensor on_ground,    // [B] bool
    torch::Tensor forward_in,   // [B] float32
    torch::Tensor strafe_in,    // [B] float32
    torch::Tensor delta_yaw_in, // [B] float32
    torch::Tensor delta_pitch_in, // [B] float32
    torch::Tensor jump_in,      // [B] bool
    int world_size,
    float dt,
    float gravity,
    float walk_speed,
    float jump_vel
) {
    int batch_size = positions.size(0);

    launch_physics(
        positions.data_ptr<float>(),
        velocities.data_ptr<float>(),
        yaws.data_ptr<float>(),
        pitches.data_ptr<float>(),
        on_ground.data_ptr<bool>(),
        forward_in.data_ptr<float>(),
        strafe_in.data_ptr<float>(),
        delta_yaw_in.data_ptr<float>(),
        delta_pitch_in.data_ptr<float>(),
        jump_in.data_ptr<bool>(),
        batch_size, world_size, dt, gravity, walk_speed, jump_vel
    );
}

void py_raycast_break(
    torch::Tensor voxels,     // [B, Y, X, Z] int8
    torch::Tensor positions,  // [B, 3] float32
    torch::Tensor yaws,       // [B] float32
    torch::Tensor pitches,    // [B] float32
    torch::Tensor do_break,   // [B] bool
    torch::Tensor rewards,    // [B] float32
    int8_t target_block,
    float reward_value
) {
    int batch_size = voxels.size(0);
    int world_size = voxels.size(1);

    launch_raycast_break(
        voxels.data_ptr<int8_t>(),
        positions.data_ptr<float>(),
        yaws.data_ptr<float>(),
        pitches.data_ptr<float>(),
        do_break.data_ptr<bool>(),
        rewards.data_ptr<float>(),
        batch_size, world_size, target_block, reward_value
    );
}

void py_reset(
    torch::Tensor positions,
    torch::Tensor velocities,
    torch::Tensor yaws,
    torch::Tensor pitches,
    torch::Tensor on_ground,
    torch::Tensor do_reset,
    float spawn_x,
    float spawn_y,
    float spawn_z
) {
    int batch_size = positions.size(0);

    launch_reset(
        positions.data_ptr<float>(),
        velocities.data_ptr<float>(),
        yaws.data_ptr<float>(),
        pitches.data_ptr<float>(),
        on_ground.data_ptr<bool>(),
        do_reset.data_ptr<bool>(),
        batch_size, spawn_x, spawn_y, spawn_z
    );
}

void py_generate_world(
    torch::Tensor voxels,
    int ground_height,
    unsigned int seed
) {
    int batch_size = voxels.size(0);
    int world_size = voxels.size(1);

    launch_generate_world(
        voxels.data_ptr<int8_t>(),
        batch_size, world_size, ground_height, seed
    );
}

void py_place_tree(
    torch::Tensor voxels,
    int tree_x,
    int tree_z,
    int ground_height
) {
    int batch_size = voxels.size(0);
    int world_size = voxels.size(1);

    launch_place_tree(
        voxels.data_ptr<int8_t>(),
        batch_size, world_size, tree_x, tree_z, ground_height
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "mine.cu - High-performance batched voxel RL environment";

    m.def("render", &py_render, "Batched raymarching render",
          py::arg("voxels"), py::arg("cameras"), py::arg("output"),
          py::arg("max_steps") = 64,
          py::arg("view_distance") = 32.0f,
          py::arg("fov_degrees") = 70.0f);

    m.def("physics", &py_physics, "Batched physics step",
          py::arg("positions"), py::arg("velocities"),
          py::arg("yaws"), py::arg("pitches"), py::arg("on_ground"),
          py::arg("forward_in"), py::arg("strafe_in"),
          py::arg("delta_yaw_in"), py::arg("delta_pitch_in"), py::arg("jump_in"),
          py::arg("world_size"),
          py::arg("dt") = 0.05f,
          py::arg("gravity") = -20.0f,
          py::arg("walk_speed") = 4.0f,
          py::arg("jump_vel") = 8.0f);

    m.def("raycast_break", &py_raycast_break, "Batched block breaking with reward",
          py::arg("voxels"), py::arg("positions"),
          py::arg("yaws"), py::arg("pitches"),
          py::arg("do_break"), py::arg("rewards"),
          py::arg("target_block") = -1,
          py::arg("reward_value") = 1.0f);

    m.def("reset", &py_reset, "Batched episode reset",
          py::arg("positions"), py::arg("velocities"),
          py::arg("yaws"), py::arg("pitches"), py::arg("on_ground"),
          py::arg("do_reset"),
          py::arg("spawn_x"), py::arg("spawn_y"), py::arg("spawn_z"));

    m.def("generate_world", &py_generate_world, "Generate flat world with ground",
          py::arg("voxels"), py::arg("ground_height"), py::arg("seed") = 42);

    m.def("place_tree", &py_place_tree, "Place a tree at given position",
          py::arg("voxels"), py::arg("tree_x"), py::arg("tree_z"), py::arg("ground_height"));
}
