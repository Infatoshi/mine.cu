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

    void launch_render_uint8(
        const int8_t* voxels, const float* cameras, uint8_t* output,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees
    );

    void launch_physics(
        float* positions, float* velocities, float* yaws, float* pitches,
        bool* on_ground, const float* forward_in, const float* strafe_in,
        const float* delta_yaw_in, const float* delta_pitch_in, const bool* jump_in,
        const float* speed_mult,
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
        int8_t* voxels, int batch_size, int world_x, int world_y, int world_z, int ground_height, unsigned int seed
    );

    void launch_place_tree(
        int8_t* voxels, int batch_size, int world_x, int world_y, int world_z, int tree_x, int tree_z, int ground_height
    );

    void launch_place_house(
        int8_t* voxels, int batch_size, int world_x, int world_y, int world_z, int house_x, int house_z, int ground_height
    );

    // New step API
    void launch_step(
        int8_t* voxels,
        float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
        const int8_t* buttons, const float* look,
        const bool* do_reset,
        float* forward_in, float* strafe_in, float* delta_yaw_in, float* delta_pitch_in,
        bool* jump_in, bool* do_break, bool* do_place, float* speed_mult,
        float* cameras,
        float* obs_buffer, float* rewards,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees,
        float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
        int8_t target_block, float reward_value,
        float spawn_x, float spawn_y, float spawn_z
    );

    void launch_step_uint8(
        int8_t* voxels,
        float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
        const int8_t* buttons, const float* look,
        const bool* do_reset,
        float* forward_in, float* strafe_in, float* delta_yaw_in, float* delta_pitch_in,
        bool* jump_in, bool* do_break, bool* do_place, float* speed_mult,
        float* cameras,
        uint8_t* obs_buffer, float* rewards,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees,
        float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
        int8_t target_block, float reward_value,
        float spawn_x, float spawn_y, float spawn_z
    );

    void launch_step_uint8_fast(
        int8_t* voxels,
        float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
        const int8_t* buttons, const float* look,
        const bool* do_reset,
        float* forward_in, float* strafe_in, float* delta_yaw_in, float* delta_pitch_in,
        bool* jump_in, bool* do_break, bool* do_place, float* speed_mult,
        float* cameras,
        uint8_t* obs_buffer, float* rewards,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees,
        float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
        int8_t target_block, float reward_value,
        float spawn_x, float spawn_y, float spawn_z,
        int ground_height
    );

    void launch_step_uint8_minimal(
        int8_t* voxels,
        float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
        const int8_t* buttons, const float* look,
        const bool* do_reset,
        float* forward_in, float* strafe_in, float* delta_yaw_in, float* delta_pitch_in,
        bool* jump_in, bool* do_break, bool* do_place, float* speed_mult,
        float* cameras,
        uint8_t* obs_buffer, float* rewards,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees,
        float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
        int8_t target_block, float reward_value,
        float spawn_x, float spawn_y, float spawn_z
    );

    void launch_step_uint8_fp16(
        int8_t* voxels,
        float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
        const int8_t* buttons, const float* look,
        const bool* do_reset,
        float* forward_in, float* strafe_in, float* delta_yaw_in, float* delta_pitch_in,
        bool* jump_in, bool* do_break, bool* do_place, float* speed_mult,
        float* cameras,
        uint8_t* obs_buffer, float* rewards,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees,
        float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
        int8_t target_block, float reward_value,
        float spawn_x, float spawn_y, float spawn_z
    );

    void launch_step_uint8_prebasis(
        int8_t* voxels,
        float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
        const int8_t* buttons, const float* look,
        const bool* do_reset,
        float* forward_in, float* strafe_in, float* delta_yaw_in, float* delta_pitch_in,
        bool* jump_in, bool* do_break, bool* do_place, float* speed_mult,
        float* cameras,
        float* basis,
        uint8_t* obs_buffer, float* rewards,
        int batch_size, int world_x, int world_y, int world_z, int width, int height,
        int max_steps, float view_distance, float fov_degrees,
        float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
        int8_t target_block, float reward_value,
        float spawn_x, float spawn_y, float spawn_z
    );

    // Fast step: episode check/update in C++, world regen handled by Python
    void launch_step_uint8_prebasis_full(
        int8_t* voxels,
        float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
        const int8_t* buttons, const float* look,
        bool* do_reset,  // Writable: set by episode_check_kernel
        int* step_count, // [B] step counter
        float* forward_in, float* strafe_in, float* delta_yaw_in, float* delta_pitch_in,
        bool* jump_in, bool* do_break, bool* do_place, float* speed_mult,
        float* cameras,
        float* basis,
        uint8_t* obs_buffer, float* rewards,
        int batch_size, int world_x, int world_y, int world_z, int width, int height,
        int max_steps, float view_distance, float fov_degrees,
        float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
        int8_t target_block, float reward_value,
        float spawn_x, float spawn_y, float spawn_z,
        int episode_length  // 0 = no limit
    );

    // Standalone render functions for benchmarking
    void launch_precompute_camera_basis(
        const float* cameras, float* basis,
        int batch_size, int width, int height, float fov_degrees
    );

    void launch_render_uint8_prebasis(
        const int8_t* voxels, const float* basis, uint8_t* output,
        int batch_size, int world_size, int width, int height, int max_steps
    );

    void launch_render_uint8_smem(
        const int8_t* voxels, const float* basis, uint8_t* output,
        int batch_size, int world_size, int width, int height, int max_steps
    );

    void launch_render_uint8_coalesced(
        const int8_t* voxels,  // [Y, X, Z, B] layout!
        const float* basis, uint8_t* output,
        int batch_size, int world_size, int width, int height, int max_steps
    );

    void launch_step_uint8_smem(
        int8_t* voxels,
        float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
        const int8_t* buttons, const float* look,
        const bool* do_reset,
        float* forward_in, float* strafe_in, float* delta_yaw_in, float* delta_pitch_in,
        bool* jump_in, bool* do_break, bool* do_place, float* speed_mult,
        float* cameras,
        float* basis,
        uint8_t* obs_buffer, float* rewards,
        int batch_size, int world_size, int width, int height,
        int max_steps, float view_distance, float fov_degrees,
        float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
        int8_t target_block, float reward_value,
        float spawn_x, float spawn_y, float spawn_z
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

void py_render_uint8(
    torch::Tensor voxels,    // [B, Y, X, Z] int8
    torch::Tensor cameras,   // [B, 5] float32
    torch::Tensor output,    // [B, H, W, 3] uint8
    int max_steps,
    float view_distance,
    float fov_degrees
) {
    TORCH_CHECK(voxels.is_cuda(), "voxels must be CUDA tensor");
    TORCH_CHECK(cameras.is_cuda(), "cameras must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(voxels.dtype() == torch::kInt8, "voxels must be int8");
    TORCH_CHECK(cameras.dtype() == torch::kFloat32, "cameras must be float32");
    TORCH_CHECK(output.dtype() == torch::kUInt8, "output must be uint8");

    int batch_size = voxels.size(0);
    int world_size = voxels.size(1);
    int height = output.size(1);
    int width = output.size(2);

    launch_render_uint8(
        voxels.data_ptr<int8_t>(),
        cameras.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
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
    torch::Tensor speed_mult,   // [B] float32
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
        speed_mult.data_ptr<float>(),
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
    int world_y = voxels.size(1);
    int world_x = voxels.size(2);
    int world_z = voxels.size(3);

    launch_generate_world(
        voxels.data_ptr<int8_t>(),
        batch_size, world_x, world_y, world_z, ground_height, seed
    );
}

void py_place_tree(
    torch::Tensor voxels,
    int tree_x,
    int tree_z,
    int ground_height
) {
    int batch_size = voxels.size(0);
    int world_y = voxels.size(1);
    int world_x = voxels.size(2);
    int world_z = voxels.size(3);

    launch_place_tree(
        voxels.data_ptr<int8_t>(),
        batch_size, world_x, world_y, world_z, tree_x, tree_z, ground_height
    );
}

void py_place_house(
    torch::Tensor voxels,
    int house_x,
    int house_z,
    int ground_height
) {
    int batch_size = voxels.size(0);
    int world_y = voxels.size(1);
    int world_x = voxels.size(2);
    int world_z = voxels.size(3);

    launch_place_house(
        voxels.data_ptr<int8_t>(),
        batch_size, world_x, world_y, world_z, house_x, house_z, ground_height
    );
}

// New unified step function
void py_step(
    // World state
    torch::Tensor voxels,         // [B, Y, X, Z] int8
    // Agent state
    torch::Tensor positions,      // [B, 3] float32
    torch::Tensor velocities,     // [B, 3] float32
    torch::Tensor yaws,           // [B] float32
    torch::Tensor pitches,        // [B] float32
    torch::Tensor on_ground,      // [B] bool
    // Actions
    torch::Tensor buttons,        // [B, 8] int8 multi-hot
    torch::Tensor look,           // [B, 2] float32 radians
    // Reset mask
    torch::Tensor do_reset,       // [B] bool
    // Internal buffers
    torch::Tensor forward_in,     // [B] float32
    torch::Tensor strafe_in,      // [B] float32
    torch::Tensor delta_yaw_in,   // [B] float32
    torch::Tensor delta_pitch_in, // [B] float32
    torch::Tensor jump_in,        // [B] bool
    torch::Tensor do_break,       // [B] bool
    torch::Tensor do_place,       // [B] bool
    torch::Tensor speed_mult,     // [B] float32
    // Camera buffer
    torch::Tensor cameras,        // [B, 5] float32
    // Output
    torch::Tensor obs_buffer,     // [B, H, W, 3] float32
    torch::Tensor rewards,        // [B] float32
    // Config
    int width,
    int height,
    int max_steps,
    float view_distance,
    float fov_degrees,
    float dt,
    float gravity,
    float walk_speed,
    float sprint_mult,
    float jump_vel,
    int8_t target_block,
    float reward_value,
    float spawn_x,
    float spawn_y,
    float spawn_z
) {
    TORCH_CHECK(voxels.is_cuda(), "voxels must be CUDA tensor");
    TORCH_CHECK(buttons.is_cuda(), "buttons must be CUDA tensor");
    TORCH_CHECK(look.is_cuda(), "look must be CUDA tensor");
    TORCH_CHECK(buttons.dtype() == torch::kInt8, "buttons must be int8");
    TORCH_CHECK(look.dtype() == torch::kFloat32, "look must be float32");

    int batch_size = voxels.size(0);
    int world_size = voxels.size(1);

    launch_step(
        voxels.data_ptr<int8_t>(),
        positions.data_ptr<float>(),
        velocities.data_ptr<float>(),
        yaws.data_ptr<float>(),
        pitches.data_ptr<float>(),
        on_ground.data_ptr<bool>(),
        buttons.data_ptr<int8_t>(),
        look.data_ptr<float>(),
        do_reset.data_ptr<bool>(),
        forward_in.data_ptr<float>(),
        strafe_in.data_ptr<float>(),
        delta_yaw_in.data_ptr<float>(),
        delta_pitch_in.data_ptr<float>(),
        jump_in.data_ptr<bool>(),
        do_break.data_ptr<bool>(),
        do_place.data_ptr<bool>(),
        speed_mult.data_ptr<float>(),
        cameras.data_ptr<float>(),
        obs_buffer.data_ptr<float>(),
        rewards.data_ptr<float>(),
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees,
        dt, gravity, walk_speed, sprint_mult, jump_vel,
        target_block, reward_value,
        spawn_x, spawn_y, spawn_z
    );
}

// UINT8 version of unified step function
void py_step_uint8(
    // World state
    torch::Tensor voxels,         // [B, Y, X, Z] int8
    // Agent state
    torch::Tensor positions,      // [B, 3] float32
    torch::Tensor velocities,     // [B, 3] float32
    torch::Tensor yaws,           // [B] float32
    torch::Tensor pitches,        // [B] float32
    torch::Tensor on_ground,      // [B] bool
    // Actions
    torch::Tensor buttons,        // [B, 8] int8 multi-hot
    torch::Tensor look,           // [B, 2] float32 radians
    // Reset mask
    torch::Tensor do_reset,       // [B] bool
    // Internal buffers
    torch::Tensor forward_in,     // [B] float32
    torch::Tensor strafe_in,      // [B] float32
    torch::Tensor delta_yaw_in,   // [B] float32
    torch::Tensor delta_pitch_in, // [B] float32
    torch::Tensor jump_in,        // [B] bool
    torch::Tensor do_break,       // [B] bool
    torch::Tensor do_place,       // [B] bool
    torch::Tensor speed_mult,     // [B] float32
    // Camera buffer
    torch::Tensor cameras,        // [B, 5] float32
    // Output
    torch::Tensor obs_buffer,     // [B, H, W, 3] uint8
    torch::Tensor rewards,        // [B] float32
    // Config
    int width,
    int height,
    int max_steps,
    float view_distance,
    float fov_degrees,
    float dt,
    float gravity,
    float walk_speed,
    float sprint_mult,
    float jump_vel,
    int8_t target_block,
    float reward_value,
    float spawn_x,
    float spawn_y,
    float spawn_z
) {
    TORCH_CHECK(voxels.is_cuda(), "voxels must be CUDA tensor");
    TORCH_CHECK(buttons.is_cuda(), "buttons must be CUDA tensor");
    TORCH_CHECK(look.is_cuda(), "look must be CUDA tensor");
    TORCH_CHECK(obs_buffer.is_cuda(), "obs_buffer must be CUDA tensor");
    TORCH_CHECK(buttons.dtype() == torch::kInt8, "buttons must be int8");
    TORCH_CHECK(look.dtype() == torch::kFloat32, "look must be float32");
    TORCH_CHECK(obs_buffer.dtype() == torch::kUInt8, "obs_buffer must be uint8");

    int batch_size = voxels.size(0);
    int world_size = voxels.size(1);

    launch_step_uint8(
        voxels.data_ptr<int8_t>(),
        positions.data_ptr<float>(),
        velocities.data_ptr<float>(),
        yaws.data_ptr<float>(),
        pitches.data_ptr<float>(),
        on_ground.data_ptr<bool>(),
        buttons.data_ptr<int8_t>(),
        look.data_ptr<float>(),
        do_reset.data_ptr<bool>(),
        forward_in.data_ptr<float>(),
        strafe_in.data_ptr<float>(),
        delta_yaw_in.data_ptr<float>(),
        delta_pitch_in.data_ptr<float>(),
        jump_in.data_ptr<bool>(),
        do_break.data_ptr<bool>(),
        do_place.data_ptr<bool>(),
        speed_mult.data_ptr<float>(),
        cameras.data_ptr<float>(),
        obs_buffer.data_ptr<uint8_t>(),
        rewards.data_ptr<float>(),
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees,
        dt, gravity, walk_speed, sprint_mult, jump_vel,
        target_block, reward_value,
        spawn_x, spawn_y, spawn_z
    );
}

// UINT8 Fast version with ground-accelerated rendering
void py_step_uint8_fast(
    torch::Tensor voxels,
    torch::Tensor positions, torch::Tensor velocities,
    torch::Tensor yaws, torch::Tensor pitches, torch::Tensor on_ground,
    torch::Tensor buttons, torch::Tensor look,
    torch::Tensor do_reset,
    torch::Tensor forward_in, torch::Tensor strafe_in,
    torch::Tensor delta_yaw_in, torch::Tensor delta_pitch_in,
    torch::Tensor jump_in, torch::Tensor do_break, torch::Tensor do_place, torch::Tensor speed_mult,
    torch::Tensor cameras,
    torch::Tensor obs_buffer, torch::Tensor rewards,
    int width, int height,
    int max_steps, float view_distance, float fov_degrees,
    float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
    int8_t target_block, float reward_value,
    float spawn_x, float spawn_y, float spawn_z,
    int ground_height
) {
    TORCH_CHECK(voxels.is_cuda(), "voxels must be CUDA tensor");
    TORCH_CHECK(obs_buffer.dtype() == torch::kUInt8, "obs_buffer must be uint8");

    int batch_size = voxels.size(0);
    int world_size = voxels.size(1);

    launch_step_uint8_fast(
        voxels.data_ptr<int8_t>(),
        positions.data_ptr<float>(),
        velocities.data_ptr<float>(),
        yaws.data_ptr<float>(),
        pitches.data_ptr<float>(),
        on_ground.data_ptr<bool>(),
        buttons.data_ptr<int8_t>(),
        look.data_ptr<float>(),
        do_reset.data_ptr<bool>(),
        forward_in.data_ptr<float>(),
        strafe_in.data_ptr<float>(),
        delta_yaw_in.data_ptr<float>(),
        delta_pitch_in.data_ptr<float>(),
        jump_in.data_ptr<bool>(),
        do_break.data_ptr<bool>(),
        do_place.data_ptr<bool>(),
        speed_mult.data_ptr<float>(),
        cameras.data_ptr<float>(),
        obs_buffer.data_ptr<uint8_t>(),
        rewards.data_ptr<float>(),
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees,
        dt, gravity, walk_speed, sprint_mult, jump_vel,
        target_block, reward_value,
        spawn_x, spawn_y, spawn_z,
        ground_height
    );
}

void py_step_uint8_minimal(
    torch::Tensor voxels,
    torch::Tensor positions, torch::Tensor velocities,
    torch::Tensor yaws, torch::Tensor pitches, torch::Tensor on_ground,
    torch::Tensor buttons, torch::Tensor look,
    torch::Tensor do_reset,
    torch::Tensor forward_in, torch::Tensor strafe_in,
    torch::Tensor delta_yaw_in, torch::Tensor delta_pitch_in,
    torch::Tensor jump_in, torch::Tensor do_break, torch::Tensor do_place, torch::Tensor speed_mult,
    torch::Tensor cameras,
    torch::Tensor obs_buffer, torch::Tensor rewards,
    int width, int height,
    int max_steps, float view_distance, float fov_degrees,
    float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
    int8_t target_block, float reward_value,
    float spawn_x, float spawn_y, float spawn_z
) {
    TORCH_CHECK(voxels.is_cuda(), "voxels must be CUDA tensor");
    TORCH_CHECK(obs_buffer.dtype() == torch::kUInt8, "obs_buffer must be uint8");

    int batch_size = voxels.size(0);
    int world_size = voxels.size(1);

    launch_step_uint8_minimal(
        voxels.data_ptr<int8_t>(),
        positions.data_ptr<float>(),
        velocities.data_ptr<float>(),
        yaws.data_ptr<float>(),
        pitches.data_ptr<float>(),
        on_ground.data_ptr<bool>(),
        buttons.data_ptr<int8_t>(),
        look.data_ptr<float>(),
        do_reset.data_ptr<bool>(),
        forward_in.data_ptr<float>(),
        strafe_in.data_ptr<float>(),
        delta_yaw_in.data_ptr<float>(),
        delta_pitch_in.data_ptr<float>(),
        jump_in.data_ptr<bool>(),
        do_break.data_ptr<bool>(),
        do_place.data_ptr<bool>(),
        speed_mult.data_ptr<float>(),
        cameras.data_ptr<float>(),
        obs_buffer.data_ptr<uint8_t>(),
        rewards.data_ptr<float>(),
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees,
        dt, gravity, walk_speed, sprint_mult, jump_vel,
        target_block, reward_value,
        spawn_x, spawn_y, spawn_z
    );
}

void py_step_uint8_fp16(
    torch::Tensor voxels,
    torch::Tensor positions, torch::Tensor velocities,
    torch::Tensor yaws, torch::Tensor pitches, torch::Tensor on_ground,
    torch::Tensor buttons, torch::Tensor look,
    torch::Tensor do_reset,
    torch::Tensor forward_in, torch::Tensor strafe_in,
    torch::Tensor delta_yaw_in, torch::Tensor delta_pitch_in,
    torch::Tensor jump_in, torch::Tensor do_break, torch::Tensor do_place, torch::Tensor speed_mult,
    torch::Tensor cameras,
    torch::Tensor obs_buffer, torch::Tensor rewards,
    int width, int height,
    int max_steps, float view_distance, float fov_degrees,
    float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
    int8_t target_block, float reward_value,
    float spawn_x, float spawn_y, float spawn_z
) {
    TORCH_CHECK(voxels.is_cuda(), "voxels must be CUDA tensor");
    TORCH_CHECK(obs_buffer.dtype() == torch::kUInt8, "obs_buffer must be uint8");

    int batch_size = voxels.size(0);
    int world_size = voxels.size(1);

    launch_step_uint8_fp16(
        voxels.data_ptr<int8_t>(),
        positions.data_ptr<float>(),
        velocities.data_ptr<float>(),
        yaws.data_ptr<float>(),
        pitches.data_ptr<float>(),
        on_ground.data_ptr<bool>(),
        buttons.data_ptr<int8_t>(),
        look.data_ptr<float>(),
        do_reset.data_ptr<bool>(),
        forward_in.data_ptr<float>(),
        strafe_in.data_ptr<float>(),
        delta_yaw_in.data_ptr<float>(),
        delta_pitch_in.data_ptr<float>(),
        jump_in.data_ptr<bool>(),
        do_break.data_ptr<bool>(),
        do_place.data_ptr<bool>(),
        speed_mult.data_ptr<float>(),
        cameras.data_ptr<float>(),
        obs_buffer.data_ptr<uint8_t>(),
        rewards.data_ptr<float>(),
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees,
        dt, gravity, walk_speed, sprint_mult, jump_vel,
        target_block, reward_value,
        spawn_x, spawn_y, spawn_z
    );
}

void py_step_uint8_prebasis(
    torch::Tensor voxels,
    torch::Tensor positions, torch::Tensor velocities,
    torch::Tensor yaws, torch::Tensor pitches, torch::Tensor on_ground,
    torch::Tensor buttons, torch::Tensor look,
    torch::Tensor do_reset,
    torch::Tensor forward_in, torch::Tensor strafe_in,
    torch::Tensor delta_yaw_in, torch::Tensor delta_pitch_in,
    torch::Tensor jump_in, torch::Tensor do_break, torch::Tensor do_place, torch::Tensor speed_mult,
    torch::Tensor cameras,
    torch::Tensor basis,  // [B, 14] precomputed camera basis
    torch::Tensor obs_buffer, torch::Tensor rewards,
    int width, int height,
    int max_steps, float view_distance, float fov_degrees,
    float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
    int8_t target_block, float reward_value,
    float spawn_x, float spawn_y, float spawn_z
) {
    TORCH_CHECK(voxels.is_cuda(), "voxels must be CUDA tensor");
    TORCH_CHECK(obs_buffer.dtype() == torch::kUInt8, "obs_buffer must be uint8");
    TORCH_CHECK(basis.dtype() == torch::kFloat32, "basis must be float32");

    int batch_size = voxels.size(0);
    int world_y = voxels.size(1);
    int world_x = voxels.size(2);
    int world_z = voxels.size(3);

    launch_step_uint8_prebasis(
        voxels.data_ptr<int8_t>(),
        positions.data_ptr<float>(),
        velocities.data_ptr<float>(),
        yaws.data_ptr<float>(),
        pitches.data_ptr<float>(),
        on_ground.data_ptr<bool>(),
        buttons.data_ptr<int8_t>(),
        look.data_ptr<float>(),
        do_reset.data_ptr<bool>(),
        forward_in.data_ptr<float>(),
        strafe_in.data_ptr<float>(),
        delta_yaw_in.data_ptr<float>(),
        delta_pitch_in.data_ptr<float>(),
        jump_in.data_ptr<bool>(),
        do_break.data_ptr<bool>(),
        do_place.data_ptr<bool>(),
        speed_mult.data_ptr<float>(),
        cameras.data_ptr<float>(),
        basis.data_ptr<float>(),
        obs_buffer.data_ptr<uint8_t>(),
        rewards.data_ptr<float>(),
        batch_size, world_x, world_y, world_z, width, height,
        max_steps, view_distance, fov_degrees,
        dt, gravity, walk_speed, sprint_mult, jump_vel,
        target_block, reward_value,
        spawn_x, spawn_y, spawn_z
    );
}

// Fast step: episode check/update in C++, world regen handled by Python
void py_step_uint8_prebasis_full(
    torch::Tensor voxels,
    torch::Tensor positions, torch::Tensor velocities,
    torch::Tensor yaws, torch::Tensor pitches, torch::Tensor on_ground,
    torch::Tensor buttons, torch::Tensor look,
    torch::Tensor do_reset,   // Writable: set by episode_check_kernel
    torch::Tensor step_count, // [B] int32 step counter
    torch::Tensor forward_in, torch::Tensor strafe_in,
    torch::Tensor delta_yaw_in, torch::Tensor delta_pitch_in,
    torch::Tensor jump_in, torch::Tensor do_break, torch::Tensor do_place, torch::Tensor speed_mult,
    torch::Tensor cameras,
    torch::Tensor basis,
    torch::Tensor obs_buffer, torch::Tensor rewards,
    int width, int height,
    int max_steps, float view_distance, float fov_degrees,
    float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
    int8_t target_block, float reward_value,
    float spawn_x, float spawn_y, float spawn_z,
    int episode_length  // 0 = no limit
) {
    TORCH_CHECK(voxels.is_cuda(), "voxels must be CUDA tensor");
    TORCH_CHECK(obs_buffer.dtype() == torch::kUInt8, "obs_buffer must be uint8");
    TORCH_CHECK(basis.dtype() == torch::kFloat32, "basis must be float32");
    TORCH_CHECK(step_count.dtype() == torch::kInt32, "step_count must be int32");

    int batch_size = voxels.size(0);
    int world_y = voxels.size(1);
    int world_x = voxels.size(2);
    int world_z = voxels.size(3);

    launch_step_uint8_prebasis_full(
        voxels.data_ptr<int8_t>(),
        positions.data_ptr<float>(),
        velocities.data_ptr<float>(),
        yaws.data_ptr<float>(),
        pitches.data_ptr<float>(),
        on_ground.data_ptr<bool>(),
        buttons.data_ptr<int8_t>(),
        look.data_ptr<float>(),
        do_reset.data_ptr<bool>(),
        step_count.data_ptr<int>(),
        forward_in.data_ptr<float>(),
        strafe_in.data_ptr<float>(),
        delta_yaw_in.data_ptr<float>(),
        delta_pitch_in.data_ptr<float>(),
        jump_in.data_ptr<bool>(),
        do_break.data_ptr<bool>(),
        do_place.data_ptr<bool>(),
        speed_mult.data_ptr<float>(),
        cameras.data_ptr<float>(),
        basis.data_ptr<float>(),
        obs_buffer.data_ptr<uint8_t>(),
        rewards.data_ptr<float>(),
        batch_size, world_x, world_y, world_z, width, height,
        max_steps, view_distance, fov_degrees,
        dt, gravity, walk_speed, sprint_mult, jump_vel,
        target_block, reward_value,
        spawn_x, spawn_y, spawn_z,
        episode_length
    );
}

// Standalone render functions for benchmarking
void py_precompute_camera_basis(
    torch::Tensor cameras,  // [B, 5] float32
    torch::Tensor basis,    // [B, 14] float32 (output)
    int width, int height,
    float fov_degrees
) {
    TORCH_CHECK(cameras.is_cuda(), "cameras must be CUDA tensor");
    TORCH_CHECK(basis.is_cuda(), "basis must be CUDA tensor");
    TORCH_CHECK(cameras.dtype() == torch::kFloat32, "cameras must be float32");
    TORCH_CHECK(basis.dtype() == torch::kFloat32, "basis must be float32");

    int batch_size = cameras.size(0);

    launch_precompute_camera_basis(
        cameras.data_ptr<float>(),
        basis.data_ptr<float>(),
        batch_size, width, height, fov_degrees
    );
}

void py_render_uint8_prebasis(
    torch::Tensor voxels,   // [B, Y, X, Z] int8
    torch::Tensor basis,    // [B, 14] float32
    torch::Tensor output,   // [B, H, W, 3] uint8
    int max_steps
) {
    TORCH_CHECK(voxels.is_cuda(), "voxels must be CUDA tensor");
    TORCH_CHECK(basis.is_cuda(), "basis must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(voxels.dtype() == torch::kInt8, "voxels must be int8");
    TORCH_CHECK(basis.dtype() == torch::kFloat32, "basis must be float32");
    TORCH_CHECK(output.dtype() == torch::kUInt8, "output must be uint8");

    int batch_size = voxels.size(0);
    int world_size = voxels.size(1);
    int height = output.size(1);
    int width = output.size(2);

    launch_render_uint8_prebasis(
        voxels.data_ptr<int8_t>(),
        basis.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        batch_size, world_size, width, height, max_steps
    );
}

void py_render_uint8_coalesced(
    torch::Tensor voxels,   // [Y, X, Z, B] int8 - transposed layout!
    torch::Tensor basis,    // [B, 14] float32
    torch::Tensor output,   // [B, H, W, 3] uint8
    int max_steps
) {
    TORCH_CHECK(voxels.is_cuda(), "voxels must be CUDA tensor");
    TORCH_CHECK(basis.is_cuda(), "basis must be CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
    TORCH_CHECK(voxels.dtype() == torch::kInt8, "voxels must be int8");
    TORCH_CHECK(basis.dtype() == torch::kFloat32, "basis must be float32");
    TORCH_CHECK(output.dtype() == torch::kUInt8, "output must be uint8");

    // Voxels: [Y, X, Z, B] - batch is innermost dimension
    int world_size = voxels.size(0);  // Y dimension
    int batch_size = voxels.size(3);  // B dimension
    int height = output.size(1);
    int width = output.size(2);

    launch_render_uint8_coalesced(
        voxels.data_ptr<int8_t>(),
        basis.data_ptr<float>(),
        output.data_ptr<uint8_t>(),
        batch_size, world_size, width, height, max_steps
    );
}

void py_step_uint8_smem(
    torch::Tensor voxels,
    torch::Tensor positions, torch::Tensor velocities,
    torch::Tensor yaws, torch::Tensor pitches, torch::Tensor on_ground,
    torch::Tensor buttons, torch::Tensor look,
    torch::Tensor do_reset,
    torch::Tensor forward_in, torch::Tensor strafe_in,
    torch::Tensor delta_yaw_in, torch::Tensor delta_pitch_in,
    torch::Tensor jump_in, torch::Tensor do_break, torch::Tensor do_place, torch::Tensor speed_mult,
    torch::Tensor cameras,
    torch::Tensor basis,  // [B, 14] precomputed camera basis
    torch::Tensor obs_buffer, torch::Tensor rewards,
    int width, int height,
    int max_steps, float view_distance, float fov_degrees,
    float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
    int8_t target_block, float reward_value,
    float spawn_x, float spawn_y, float spawn_z
) {
    TORCH_CHECK(voxels.is_cuda(), "voxels must be CUDA tensor");
    TORCH_CHECK(obs_buffer.dtype() == torch::kUInt8, "obs_buffer must be uint8");
    TORCH_CHECK(basis.dtype() == torch::kFloat32, "basis must be float32");

    int batch_size = voxels.size(0);
    int world_size = voxels.size(1);

    launch_step_uint8_smem(
        voxels.data_ptr<int8_t>(),
        positions.data_ptr<float>(),
        velocities.data_ptr<float>(),
        yaws.data_ptr<float>(),
        pitches.data_ptr<float>(),
        on_ground.data_ptr<bool>(),
        buttons.data_ptr<int8_t>(),
        look.data_ptr<float>(),
        do_reset.data_ptr<bool>(),
        forward_in.data_ptr<float>(),
        strafe_in.data_ptr<float>(),
        delta_yaw_in.data_ptr<float>(),
        delta_pitch_in.data_ptr<float>(),
        jump_in.data_ptr<bool>(),
        do_break.data_ptr<bool>(),
        do_place.data_ptr<bool>(),
        speed_mult.data_ptr<float>(),
        cameras.data_ptr<float>(),
        basis.data_ptr<float>(),
        obs_buffer.data_ptr<uint8_t>(),
        rewards.data_ptr<float>(),
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees,
        dt, gravity, walk_speed, sprint_mult, jump_vel,
        target_block, reward_value,
        spawn_x, spawn_y, spawn_z
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "mine.cu - High-performance batched voxel RL environment";

    m.def("render", &py_render, "Batched raymarching render",
          py::arg("voxels"), py::arg("cameras"), py::arg("output"),
          py::arg("max_steps") = 64,
          py::arg("view_distance") = 32.0f,
          py::arg("fov_degrees") = 70.0f);

    m.def("render_uint8", &py_render_uint8, "Batched raymarching render (uint8 output)",
          py::arg("voxels"), py::arg("cameras"), py::arg("output"),
          py::arg("max_steps") = 64,
          py::arg("view_distance") = 32.0f,
          py::arg("fov_degrees") = 70.0f);

    m.def("physics", &py_physics, "Batched physics step",
          py::arg("positions"), py::arg("velocities"),
          py::arg("yaws"), py::arg("pitches"), py::arg("on_ground"),
          py::arg("forward_in"), py::arg("strafe_in"),
          py::arg("delta_yaw_in"), py::arg("delta_pitch_in"), py::arg("jump_in"),
          py::arg("speed_mult"),
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

    m.def("place_house", &py_place_house, "Place a house at given position",
          py::arg("voxels"), py::arg("house_x"), py::arg("house_z"), py::arg("ground_height"));

    // New unified step function
    m.def("step", &py_step, "Unified step function with all kernels",
          py::arg("voxels"),
          py::arg("positions"), py::arg("velocities"),
          py::arg("yaws"), py::arg("pitches"), py::arg("on_ground"),
          py::arg("buttons"), py::arg("look"),
          py::arg("do_reset"),
          py::arg("forward_in"), py::arg("strafe_in"),
          py::arg("delta_yaw_in"), py::arg("delta_pitch_in"),
          py::arg("jump_in"), py::arg("do_break"), py::arg("do_place"), py::arg("speed_mult"),
          py::arg("cameras"),
          py::arg("obs_buffer"), py::arg("rewards"),
          py::arg("width"), py::arg("height"),
          py::arg("max_steps") = 64,
          py::arg("view_distance") = 32.0f,
          py::arg("fov_degrees") = 70.0f,
          py::arg("dt") = 0.05f,
          py::arg("gravity") = -20.0f,
          py::arg("walk_speed") = 4.0f,
          py::arg("sprint_mult") = 1.5f,
          py::arg("jump_vel") = 8.0f,
          py::arg("target_block") = -1,
          py::arg("reward_value") = 1.0f,
          py::arg("spawn_x") = 16.0f,
          py::arg("spawn_y") = 9.0f,
          py::arg("spawn_z") = 16.0f);

    // UINT8 version of step function
    m.def("step_uint8", &py_step_uint8, "Unified step function with uint8 output",
          py::arg("voxels"),
          py::arg("positions"), py::arg("velocities"),
          py::arg("yaws"), py::arg("pitches"), py::arg("on_ground"),
          py::arg("buttons"), py::arg("look"),
          py::arg("do_reset"),
          py::arg("forward_in"), py::arg("strafe_in"),
          py::arg("delta_yaw_in"), py::arg("delta_pitch_in"),
          py::arg("jump_in"), py::arg("do_break"), py::arg("do_place"), py::arg("speed_mult"),
          py::arg("cameras"),
          py::arg("obs_buffer"), py::arg("rewards"),
          py::arg("width"), py::arg("height"),
          py::arg("max_steps") = 64,
          py::arg("view_distance") = 32.0f,
          py::arg("fov_degrees") = 70.0f,
          py::arg("dt") = 0.05f,
          py::arg("gravity") = -20.0f,
          py::arg("walk_speed") = 4.0f,
          py::arg("sprint_mult") = 1.5f,
          py::arg("jump_vel") = 8.0f,
          py::arg("target_block") = -1,
          py::arg("reward_value") = 1.0f,
          py::arg("spawn_x") = 16.0f,
          py::arg("spawn_y") = 9.0f,
          py::arg("spawn_z") = 16.0f);

    // UINT8 Fast version with ground-accelerated rendering
    m.def("step_uint8_fast", &py_step_uint8_fast, "Fast step with ground-accelerated rendering",
          py::arg("voxels"),
          py::arg("positions"), py::arg("velocities"),
          py::arg("yaws"), py::arg("pitches"), py::arg("on_ground"),
          py::arg("buttons"), py::arg("look"),
          py::arg("do_reset"),
          py::arg("forward_in"), py::arg("strafe_in"),
          py::arg("delta_yaw_in"), py::arg("delta_pitch_in"),
          py::arg("jump_in"), py::arg("do_break"), py::arg("do_place"), py::arg("speed_mult"),
          py::arg("cameras"),
          py::arg("obs_buffer"), py::arg("rewards"),
          py::arg("width"), py::arg("height"),
          py::arg("max_steps") = 64,
          py::arg("view_distance") = 32.0f,
          py::arg("fov_degrees") = 70.0f,
          py::arg("dt") = 0.05f,
          py::arg("gravity") = -20.0f,
          py::arg("walk_speed") = 4.0f,
          py::arg("sprint_mult") = 1.5f,
          py::arg("jump_vel") = 8.0f,
          py::arg("target_block") = -1,
          py::arg("reward_value") = 1.0f,
          py::arg("spawn_x") = 16.0f,
          py::arg("spawn_y") = 9.0f,
          py::arg("spawn_z") = 16.0f,
          py::arg("ground_height") = 8);

    // UINT8 Minimal version - flat colors, no fog/shading for max perf
    m.def("step_uint8_minimal", &py_step_uint8_minimal, "Minimal step with flat colors (no fog/shading)",
          py::arg("voxels"),
          py::arg("positions"), py::arg("velocities"),
          py::arg("yaws"), py::arg("pitches"), py::arg("on_ground"),
          py::arg("buttons"), py::arg("look"),
          py::arg("do_reset"),
          py::arg("forward_in"), py::arg("strafe_in"),
          py::arg("delta_yaw_in"), py::arg("delta_pitch_in"),
          py::arg("jump_in"), py::arg("do_break"), py::arg("do_place"), py::arg("speed_mult"),
          py::arg("cameras"),
          py::arg("obs_buffer"), py::arg("rewards"),
          py::arg("width"), py::arg("height"),
          py::arg("max_steps") = 64,
          py::arg("view_distance") = 32.0f,
          py::arg("fov_degrees") = 70.0f,
          py::arg("dt") = 0.05f,
          py::arg("gravity") = -20.0f,
          py::arg("walk_speed") = 4.0f,
          py::arg("sprint_mult") = 1.5f,
          py::arg("jump_vel") = 8.0f,
          py::arg("target_block") = -1,
          py::arg("reward_value") = 1.0f,
          py::arg("spawn_x") = 16.0f,
          py::arg("spawn_y") = 9.0f,
          py::arg("spawn_z") = 16.0f);

    // UINT8 FP16 version - half precision DDA for max throughput
    m.def("step_uint8_fp16", &py_step_uint8_fp16, "FP16 DDA step with flat colors",
          py::arg("voxels"),
          py::arg("positions"), py::arg("velocities"),
          py::arg("yaws"), py::arg("pitches"), py::arg("on_ground"),
          py::arg("buttons"), py::arg("look"),
          py::arg("do_reset"),
          py::arg("forward_in"), py::arg("strafe_in"),
          py::arg("delta_yaw_in"), py::arg("delta_pitch_in"),
          py::arg("jump_in"), py::arg("do_break"), py::arg("do_place"), py::arg("speed_mult"),
          py::arg("cameras"),
          py::arg("obs_buffer"), py::arg("rewards"),
          py::arg("width"), py::arg("height"),
          py::arg("max_steps") = 64,
          py::arg("view_distance") = 32.0f,
          py::arg("fov_degrees") = 70.0f,
          py::arg("dt") = 0.05f,
          py::arg("gravity") = -20.0f,
          py::arg("walk_speed") = 4.0f,
          py::arg("sprint_mult") = 1.5f,
          py::arg("jump_vel") = 8.0f,
          py::arg("target_block") = -1,
          py::arg("reward_value") = 1.0f,
          py::arg("spawn_x") = 16.0f,
          py::arg("spawn_y") = 9.0f,
          py::arg("spawn_z") = 16.0f);

    // UINT8 Prebasis version - precomputed camera basis, no per-pixel trig
    m.def("step_uint8_prebasis", &py_step_uint8_prebasis, "Prebasis step with precomputed camera trig",
          py::arg("voxels"),
          py::arg("positions"), py::arg("velocities"),
          py::arg("yaws"), py::arg("pitches"), py::arg("on_ground"),
          py::arg("buttons"), py::arg("look"),
          py::arg("do_reset"),
          py::arg("forward_in"), py::arg("strafe_in"),
          py::arg("delta_yaw_in"), py::arg("delta_pitch_in"),
          py::arg("jump_in"), py::arg("do_break"), py::arg("do_place"), py::arg("speed_mult"),
          py::arg("cameras"),
          py::arg("basis"),
          py::arg("obs_buffer"), py::arg("rewards"),
          py::arg("width"), py::arg("height"),
          py::arg("max_steps") = 64,
          py::arg("view_distance") = 32.0f,
          py::arg("fov_degrees") = 70.0f,
          py::arg("dt") = 0.05f,
          py::arg("gravity") = -20.0f,
          py::arg("walk_speed") = 4.0f,
          py::arg("sprint_mult") = 1.5f,
          py::arg("jump_vel") = 8.0f,
          py::arg("target_block") = -1,
          py::arg("reward_value") = 1.0f,
          py::arg("spawn_x") = 16.0f,
          py::arg("spawn_y") = 9.0f,
          py::arg("spawn_z") = 16.0f);

    // UINT8 Prebasis Full version - episode management in C++, world regen in Python
    m.def("step_uint8_prebasis_full", &py_step_uint8_prebasis_full, "Fast step with episode management in C++",
          py::arg("voxels"),
          py::arg("positions"), py::arg("velocities"),
          py::arg("yaws"), py::arg("pitches"), py::arg("on_ground"),
          py::arg("buttons"), py::arg("look"),
          py::arg("do_reset"),
          py::arg("step_count"),
          py::arg("forward_in"), py::arg("strafe_in"),
          py::arg("delta_yaw_in"), py::arg("delta_pitch_in"),
          py::arg("jump_in"), py::arg("do_break"), py::arg("do_place"), py::arg("speed_mult"),
          py::arg("cameras"),
          py::arg("basis"),
          py::arg("obs_buffer"), py::arg("rewards"),
          py::arg("width"), py::arg("height"),
          py::arg("max_steps") = 64,
          py::arg("view_distance") = 32.0f,
          py::arg("fov_degrees") = 70.0f,
          py::arg("dt") = 0.05f,
          py::arg("gravity") = -20.0f,
          py::arg("walk_speed") = 4.0f,
          py::arg("sprint_mult") = 1.5f,
          py::arg("jump_vel") = 8.0f,
          py::arg("target_block") = -1,
          py::arg("reward_value") = 1.0f,
          py::arg("spawn_x") = 16.0f,
          py::arg("spawn_y") = 9.0f,
          py::arg("spawn_z") = 16.0f,
          py::arg("episode_length") = 0);

    // UINT8 Shared Memory version - entire voxel grid in shared memory
    m.def("step_uint8_smem", &py_step_uint8_smem, "Smem step with voxel grid in shared memory",
          py::arg("voxels"),
          py::arg("positions"), py::arg("velocities"),
          py::arg("yaws"), py::arg("pitches"), py::arg("on_ground"),
          py::arg("buttons"), py::arg("look"),
          py::arg("do_reset"),
          py::arg("forward_in"), py::arg("strafe_in"),
          py::arg("delta_yaw_in"), py::arg("delta_pitch_in"),
          py::arg("jump_in"), py::arg("do_break"), py::arg("do_place"), py::arg("speed_mult"),
          py::arg("cameras"),
          py::arg("basis"),
          py::arg("obs_buffer"), py::arg("rewards"),
          py::arg("width"), py::arg("height"),
          py::arg("max_steps") = 64,
          py::arg("view_distance") = 32.0f,
          py::arg("fov_degrees") = 70.0f,
          py::arg("dt") = 0.05f,
          py::arg("gravity") = -20.0f,
          py::arg("walk_speed") = 4.0f,
          py::arg("sprint_mult") = 1.5f,
          py::arg("jump_vel") = 8.0f,
          py::arg("target_block") = -1,
          py::arg("reward_value") = 1.0f,
          py::arg("spawn_x") = 16.0f,
          py::arg("spawn_y") = 9.0f,
          py::arg("spawn_z") = 16.0f);

    // Standalone render functions for benchmarking layout optimizations
    m.def("precompute_camera_basis", &py_precompute_camera_basis, "Precompute camera basis vectors",
          py::arg("cameras"), py::arg("basis"),
          py::arg("width"), py::arg("height"),
          py::arg("fov_degrees") = 70.0f);

    m.def("render_uint8_prebasis", &py_render_uint8_prebasis, "Render with prebasis [B,Y,X,Z] layout",
          py::arg("voxels"), py::arg("basis"), py::arg("output"),
          py::arg("max_steps") = 64);

    m.def("render_uint8_coalesced", &py_render_uint8_coalesced, "Render with coalesced [Y,X,Z,B] layout",
          py::arg("voxels"), py::arg("basis"), py::arg("output"),
          py::arg("max_steps") = 64);
}
