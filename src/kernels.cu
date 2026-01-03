// mine.cu - High-performance batched voxel RL environment
// All environment logic runs on GPU via custom CUDA kernels

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// Block types
constexpr int8_t AIR = -1;
constexpr int8_t GRASS = 0;
constexpr int8_t DIRT = 1;
constexpr int8_t STONE = 2;
constexpr int8_t OAKLOG = 3;
constexpr int8_t LEAVES = 4;
constexpr int8_t SAND = 5;
constexpr int8_t WATER = 6;
constexpr int8_t GLASS = 7;
constexpr int8_t BRICK = 8;
constexpr int8_t COBBLESTONE = 9;
constexpr int8_t PLANKS = 10;
constexpr int8_t SNOW = 11;
constexpr int8_t BEDROCK = 12;

// Block colors (R, G, B) normalized to [0, 1]
__constant__ float BLOCK_COLORS[13][3] = {
    {0.30f, 0.65f, 0.20f},  // GRASS
    {0.55f, 0.35f, 0.20f},  // DIRT
    {0.50f, 0.50f, 0.50f},  // STONE
    {0.55f, 0.40f, 0.25f},  // OAKLOG
    {0.20f, 0.55f, 0.15f},  // LEAVES
    {0.90f, 0.85f, 0.60f},  // SAND
    {0.20f, 0.40f, 0.80f},  // WATER
    {0.85f, 0.90f, 0.95f},  // GLASS
    {0.70f, 0.35f, 0.30f},  // BRICK
    {0.40f, 0.40f, 0.40f},  // COBBLESTONE
    {0.75f, 0.60f, 0.40f},  // PLANKS
    {0.95f, 0.95f, 0.98f},  // SNOW
    {0.15f, 0.15f, 0.15f},  // BEDROCK
};

// Sky color
__constant__ float SKY_COLOR[3] = {0.53f, 0.81f, 0.92f};


// =============================================================================
// RENDER KERNEL - Batched raymarching with DDA
// =============================================================================

__global__ void render_kernel(
    const int8_t* __restrict__ voxels,    // [B, Y, X, Z]
    const float* __restrict__ cameras,     // [B, 5] = (x, y, z, yaw, pitch)
    float* __restrict__ output,            // [B, H, W, 3]
    int batch_size,
    int world_size,
    int width,
    int height,
    int max_steps,
    float view_distance,
    float fov_degrees
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = batch_size * height * width;
    if (idx >= total_pixels) return;

    // Decompose index
    int b = idx / (height * width);
    int pixel_idx = idx % (height * width);
    int py = pixel_idx / width;
    int px = pixel_idx % width;

    // Camera params
    float cam_x = cameras[b * 5 + 0];
    float cam_y = cameras[b * 5 + 1];
    float cam_z = cameras[b * 5 + 2];
    float yaw = cameras[b * 5 + 3];
    float pitch = cameras[b * 5 + 4];

    // Eye position (offset for head height)
    float eye_x = cam_x;
    float eye_y = cam_y + 1.6f;
    float eye_z = cam_z;

    // Camera basis vectors
    float cos_yaw = cosf(yaw);
    float sin_yaw = sinf(yaw);
    float cos_pitch = cosf(pitch);
    float sin_pitch = sinf(pitch);

    float forward_x = sin_yaw * cos_pitch;
    float forward_y = sin_pitch;
    float forward_z = cos_yaw * cos_pitch;

    float right_x = cos_yaw;
    float right_y = 0.0f;
    float right_z = -sin_yaw;

    float up_x = -sin_yaw * sin_pitch;
    float up_y = cos_pitch;
    float up_z = -cos_yaw * sin_pitch;

    // Ray direction from pixel
    float fov_rad = fov_degrees * 3.14159265f / 180.0f;
    float aspect = (float)width / (float)height;
    float half_fov = tanf(fov_rad * 0.5f);

    float u = (2.0f * (px + 0.5f) / width - 1.0f) * half_fov * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * half_fov;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + u * right_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    // Normalize ray
    float ray_len = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    ray_x /= ray_len;
    ray_y /= ray_len;
    ray_z /= ray_len;

    // DDA raymarching
    int ws = world_size;
    int ws2 = ws * ws;
    int ws3 = ws * ws * ws;

    float pos_x = eye_x;
    float pos_y = eye_y;
    float pos_z = eye_z;

    int voxel_x = (int)floorf(pos_x);
    int voxel_y = (int)floorf(pos_y);
    int voxel_z = (int)floorf(pos_z);

    int step_x = (ray_x >= 0) ? 1 : -1;
    int step_y = (ray_y >= 0) ? 1 : -1;
    int step_z = (ray_z >= 0) ? 1 : -1;

    float inv_ray_x = (ray_x != 0.0f) ? 1.0f / ray_x : 1e30f;
    float inv_ray_y = (ray_y != 0.0f) ? 1.0f / ray_y : 1e30f;
    float inv_ray_z = (ray_z != 0.0f) ? 1.0f / ray_z : 1e30f;

    float t_delta_x = fabsf(inv_ray_x);
    float t_delta_y = fabsf(inv_ray_y);
    float t_delta_z = fabsf(inv_ray_z);

    float t_max_x, t_max_y, t_max_z;
    if (ray_x >= 0) {
        t_max_x = ((float)(voxel_x + 1) - pos_x) * inv_ray_x;
    } else {
        t_max_x = ((float)voxel_x - pos_x) * inv_ray_x;
    }
    if (ray_y >= 0) {
        t_max_y = ((float)(voxel_y + 1) - pos_y) * inv_ray_y;
    } else {
        t_max_y = ((float)voxel_y - pos_y) * inv_ray_y;
    }
    if (ray_z >= 0) {
        t_max_z = ((float)(voxel_z + 1) - pos_z) * inv_ray_z;
    } else {
        t_max_z = ((float)voxel_z - pos_z) * inv_ray_z;
    }

    float t = 0.0f;
    int hit_face = -1;  // 0=x, 1=y, 2=z
    int8_t hit_block = AIR;

    for (int step = 0; step < max_steps && t < view_distance; ++step) {
        // Check current voxel
        if (voxel_x >= 0 && voxel_x < ws &&
            voxel_y >= 0 && voxel_y < ws &&
            voxel_z >= 0 && voxel_z < ws) {

            int voxel_idx = b * ws3 + voxel_y * ws2 + voxel_x * ws + voxel_z;
            int8_t block = voxels[voxel_idx];

            if (block >= 0) {
                hit_block = block;
                break;
            }
        }

        // Step to next voxel
        if (t_max_x < t_max_y && t_max_x < t_max_z) {
            t = t_max_x;
            t_max_x += t_delta_x;
            voxel_x += step_x;
            hit_face = 0;
        } else if (t_max_y < t_max_z) {
            t = t_max_y;
            t_max_y += t_delta_y;
            voxel_y += step_y;
            hit_face = 1;
        } else {
            t = t_max_z;
            t_max_z += t_delta_z;
            voxel_z += step_z;
            hit_face = 2;
        }
    }

    // Output color
    int out_idx = (b * height * width + py * width + px) * 3;

    if (hit_block >= 0) {
        float r = BLOCK_COLORS[hit_block][0];
        float g = BLOCK_COLORS[hit_block][1];
        float b_col = BLOCK_COLORS[hit_block][2];

        // Simple face shading
        float shade = 1.0f;
        if (hit_face == 0) shade = 0.8f;       // X face
        else if (hit_face == 2) shade = 0.9f;  // Z face
        else if (hit_face == 1 && step_y < 0) shade = 0.6f;  // Bottom face

        // Distance fog
        float fog = fminf(t / view_distance, 1.0f);
        fog = fog * fog;

        output[out_idx + 0] = r * shade * (1.0f - fog) + SKY_COLOR[0] * fog;
        output[out_idx + 1] = g * shade * (1.0f - fog) + SKY_COLOR[1] * fog;
        output[out_idx + 2] = b_col * shade * (1.0f - fog) + SKY_COLOR[2] * fog;
    } else {
        output[out_idx + 0] = SKY_COLOR[0];
        output[out_idx + 1] = SKY_COLOR[1];
        output[out_idx + 2] = SKY_COLOR[2];
    }
}


// =============================================================================
// PHYSICS KERNEL - Movement, gravity, collision
// =============================================================================

__global__ void physics_kernel(
    float* __restrict__ positions,      // [B, 3]
    float* __restrict__ velocities,     // [B, 3]
    float* __restrict__ yaws,           // [B]
    float* __restrict__ pitches,        // [B]
    bool* __restrict__ on_ground,       // [B]
    const float* __restrict__ forward_in,     // [B]
    const float* __restrict__ strafe_in,      // [B]
    const float* __restrict__ delta_yaw_in,   // [B]
    const float* __restrict__ delta_pitch_in, // [B]
    const bool* __restrict__ jump_in,         // [B]
    int batch_size,
    int world_size,
    float dt,
    float gravity,
    float walk_speed,
    float jump_vel
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    // Update rotation
    yaws[i] += delta_yaw_in[i];
    float new_pitch = pitches[i] + delta_pitch_in[i];
    pitches[i] = fminf(fmaxf(new_pitch, -1.56f), 1.56f);  // Clamp to ~90 degrees

    // Movement direction
    float cos_yaw = cosf(yaws[i]);
    float sin_yaw = sinf(yaws[i]);

    float fwd = forward_in[i];
    float str = strafe_in[i];

    float move_x = sin_yaw * fwd * walk_speed * dt + cos_yaw * str * walk_speed * dt;
    float move_z = cos_yaw * fwd * walk_speed * dt - sin_yaw * str * walk_speed * dt;

    // Vertical velocity
    float vel_y = velocities[i * 3 + 1];

    if (jump_in[i] && on_ground[i]) {
        vel_y = jump_vel;
    }

    vel_y += gravity * dt;

    // Update position
    float new_x = positions[i * 3 + 0] + move_x;
    float new_y = positions[i * 3 + 1] + vel_y * dt;
    float new_z = positions[i * 3 + 2] + move_z;

    // Simple ground collision (flat world at y = world_size/4)
    float ground_y = (float)(world_size / 4);
    if (new_y <= ground_y) {
        new_y = ground_y;
        vel_y = 0.0f;
        on_ground[i] = true;
    } else {
        on_ground[i] = false;
    }

    // World boundary clamping
    float margin = 1.0f;
    new_x = fminf(fmaxf(new_x, margin), (float)world_size - margin);
    new_z = fminf(fmaxf(new_z, margin), (float)world_size - margin);

    positions[i * 3 + 0] = new_x;
    positions[i * 3 + 1] = new_y;
    positions[i * 3 + 2] = new_z;
    velocities[i * 3 + 1] = vel_y;
}


// =============================================================================
// RAYCAST BREAK KERNEL - Block breaking with reward
// =============================================================================

__global__ void raycast_break_kernel(
    int8_t* __restrict__ voxels,         // [B, Y, X, Z]
    const float* __restrict__ positions, // [B, 3]
    const float* __restrict__ yaws,      // [B]
    const float* __restrict__ pitches,   // [B]
    const bool* __restrict__ do_break,   // [B]
    float* __restrict__ rewards,         // [B]
    int batch_size,
    int world_size,
    int8_t target_block,    // Block type that gives reward (-1 for any)
    float reward_value      // Reward for breaking target block
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    rewards[i] = 0.0f;
    if (!do_break[i]) return;

    float yaw = yaws[i];
    float pitch = pitches[i];

    float cos_yaw = cosf(yaw);
    float sin_yaw = sinf(yaw);
    float cos_pitch = cosf(pitch);
    float sin_pitch = sinf(pitch);

    float dir_x = sin_yaw * cos_pitch;
    float dir_y = sin_pitch;
    float dir_z = cos_yaw * cos_pitch;

    float eye_x = positions[i * 3 + 0];
    float eye_y = positions[i * 3 + 1] + 1.6f;
    float eye_z = positions[i * 3 + 2];

    float reach = 4.0f;
    int ws = world_size;
    int ws2 = ws * ws;
    int ws3 = ws * ws * ws;

    // Simple raycast (could use DDA for accuracy)
    for (int step = 0; step < 32; ++step) {
        float t = 0.1f + step * (reach / 32.0f);
        float px = eye_x + dir_x * t;
        float py = eye_y + dir_y * t;
        float pz = eye_z + dir_z * t;

        int vx = (int)floorf(px);
        int vy = (int)floorf(py);
        int vz = (int)floorf(pz);

        if (vx >= 0 && vx < ws && vy >= 0 && vy < ws && vz >= 0 && vz < ws) {
            int idx = i * ws3 + vy * ws2 + vx * ws + vz;
            int8_t block = voxels[idx];

            if (block >= 0) {
                // Hit a solid block
                if (target_block < 0 || block == target_block) {
                    rewards[i] = reward_value;
                }
                voxels[idx] = AIR;
                return;
            }
        }
    }
}


// =============================================================================
// RESET KERNEL - Episode reset
// =============================================================================

__global__ void reset_kernel(
    float* __restrict__ positions,   // [B, 3]
    float* __restrict__ velocities,  // [B, 3]
    float* __restrict__ yaws,        // [B]
    float* __restrict__ pitches,     // [B]
    bool* __restrict__ on_ground,    // [B]
    const bool* __restrict__ do_reset, // [B]
    int batch_size,
    float spawn_x,
    float spawn_y,
    float spawn_z
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    if (do_reset[i]) {
        positions[i * 3 + 0] = spawn_x;
        positions[i * 3 + 1] = spawn_y;
        positions[i * 3 + 2] = spawn_z;
        velocities[i * 3 + 0] = 0.0f;
        velocities[i * 3 + 1] = 0.0f;
        velocities[i * 3 + 2] = 0.0f;
        yaws[i] = 0.0f;
        pitches[i] = 0.0f;
        on_ground[i] = false;
    }
}


// =============================================================================
// WORLD GENERATION KERNEL
// =============================================================================

__global__ void generate_world_kernel(
    int8_t* __restrict__ voxels,  // [B, Y, X, Z]
    int batch_size,
    int world_size,
    int ground_height,
    unsigned int seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ws = world_size;
    int ws2 = ws * ws;
    int ws3 = ws * ws * ws;
    int total = batch_size * ws3;
    if (idx >= total) return;

    int b = idx / ws3;
    int voxel_idx = idx % ws3;
    int y = voxel_idx / ws2;
    int x = (voxel_idx % ws2) / ws;
    int z = voxel_idx % ws;

    int8_t block = AIR;

    if (y == 0) {
        block = BEDROCK;
    } else if (y < ground_height - 1) {
        block = DIRT;
    } else if (y == ground_height - 1) {
        block = GRASS;
    }

    voxels[idx] = block;
}


__global__ void place_tree_kernel(
    int8_t* __restrict__ voxels,  // [B, Y, X, Z]
    int batch_size,
    int world_size,
    int tree_x,
    int tree_z,
    int ground_height
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int ws = world_size;
    int ws2 = ws * ws;
    int ws3 = ws * ws * ws;

    // Place 3-block tall log
    for (int dy = 0; dy < 3; ++dy) {
        int y = ground_height + dy;
        if (y < ws) {
            int idx = b * ws3 + y * ws2 + tree_x * ws + tree_z;
            voxels[idx] = OAKLOG;
        }
    }

    // Place leaves on top
    int leaf_y = ground_height + 3;
    if (leaf_y < ws) {
        int idx = b * ws3 + leaf_y * ws2 + tree_x * ws + tree_z;
        voxels[idx] = LEAVES;
    }
}


// =============================================================================
// C++ LAUNCHER FUNCTIONS - Called from Python bindings
// =============================================================================

extern "C" {

void launch_render(
    const int8_t* voxels, const float* cameras, float* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees
) {
    int total_pixels = batch_size * height * width;
    int block_size = 256;
    int num_blocks = (total_pixels + block_size - 1) / block_size;

    render_kernel<<<num_blocks, block_size>>>(
        voxels, cameras, output,
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void launch_physics(
    float* positions, float* velocities, float* yaws, float* pitches,
    bool* on_ground, const float* forward_in, const float* strafe_in,
    const float* delta_yaw_in, const float* delta_pitch_in, const bool* jump_in,
    int batch_size, int world_size, float dt, float gravity, float walk_speed, float jump_vel
) {
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;

    physics_kernel<<<num_blocks, block_size>>>(
        positions, velocities, yaws, pitches, on_ground,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in, jump_in,
        batch_size, world_size, dt, gravity, walk_speed, jump_vel
    );
}

void launch_raycast_break(
    int8_t* voxels, const float* positions, const float* yaws, const float* pitches,
    const bool* do_break, float* rewards,
    int batch_size, int world_size, int8_t target_block, float reward_value
) {
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;

    raycast_break_kernel<<<num_blocks, block_size>>>(
        voxels, positions, yaws, pitches, do_break, rewards,
        batch_size, world_size, target_block, reward_value
    );
}

void launch_reset(
    float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
    const bool* do_reset, int batch_size, float spawn_x, float spawn_y, float spawn_z
) {
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;

    reset_kernel<<<num_blocks, block_size>>>(
        positions, velocities, yaws, pitches, on_ground,
        do_reset, batch_size, spawn_x, spawn_y, spawn_z
    );
}

void launch_generate_world(
    int8_t* voxels, int batch_size, int world_size, int ground_height, unsigned int seed
) {
    int total = batch_size * world_size * world_size * world_size;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    generate_world_kernel<<<num_blocks, block_size>>>(
        voxels, batch_size, world_size, ground_height, seed
    );
}

void launch_place_tree(
    int8_t* voxels, int batch_size, int world_size, int tree_x, int tree_z, int ground_height
) {
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;

    place_tree_kernel<<<num_blocks, block_size>>>(
        voxels, batch_size, world_size, tree_x, tree_z, ground_height
    );
}

}  // extern "C"
