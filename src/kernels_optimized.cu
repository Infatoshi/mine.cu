// kernels_optimized.cu - Optimized render kernel variants for benchmarking
// Each variant tests specific optimization techniques for DDA raymarching

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// Block types
constexpr int8_t AIR = -1;

// Block colors (R, G, B) normalized to [0, 1]
__constant__ float BLOCK_COLORS_OPT[13][3] = {
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

__constant__ float SKY_COLOR_OPT[3] = {0.53f, 0.81f, 0.92f};


// =============================================================================
// VARIANT 1: Shared Memory for Block Colors
// Optimization: Load block colors into shared memory once per block
// =============================================================================

__global__ void render_kernel_v1_shared_colors(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ cameras,
    float* __restrict__ output,
    int batch_size,
    int world_size,
    int width,
    int height,
    int max_steps,
    float view_distance,
    float fov_degrees
) {
    // Shared memory for block colors
    __shared__ float s_colors[13][3];
    __shared__ float s_sky[3];

    // Load colors into shared memory (first 13*3 = 39 threads do it)
    int tid = threadIdx.x;
    if (tid < 39) {
        int block_idx = tid / 3;
        int channel = tid % 3;
        s_colors[block_idx][channel] = BLOCK_COLORS_OPT[block_idx][channel];
    }
    if (tid < 3) {
        s_sky[tid] = SKY_COLOR_OPT[tid];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = batch_size * height * width;
    if (idx >= total_pixels) return;

    int b = idx / (height * width);
    int pixel_idx = idx % (height * width);
    int py = pixel_idx / width;
    int px = pixel_idx % width;

    float cam_x = cameras[b * 5 + 0];
    float cam_y = cameras[b * 5 + 1];
    float cam_z = cameras[b * 5 + 2];
    float yaw = cameras[b * 5 + 3];
    float pitch = cameras[b * 5 + 4];

    float eye_x = cam_x;
    float eye_y = cam_y + 1.6f;
    float eye_z = cam_z;

    float cos_yaw = cosf(yaw);
    float sin_yaw = sinf(yaw);
    float cos_pitch = cosf(pitch);
    float sin_pitch = sinf(pitch);

    float forward_x = sin_yaw * cos_pitch;
    float forward_y = sin_pitch;
    float forward_z = cos_yaw * cos_pitch;

    float right_x = cos_yaw;
    float right_z = -sin_yaw;

    float up_x = -sin_yaw * sin_pitch;
    float up_y = cos_pitch;
    float up_z = -cos_yaw * sin_pitch;

    float fov_rad = fov_degrees * 3.14159265f / 180.0f;
    float aspect = (float)width / (float)height;
    float half_fov = tanf(fov_rad * 0.5f);

    float u = (2.0f * (px + 0.5f) / width - 1.0f) * half_fov * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * half_fov;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    float ray_len = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    ray_x /= ray_len;
    ray_y /= ray_len;
    ray_z /= ray_len;

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
    int hit_face = -1;
    int8_t hit_block = AIR;

    for (int step = 0; step < max_steps && t < view_distance; ++step) {
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

    int out_idx = (b * height * width + py * width + px) * 3;

    if (hit_block >= 0) {
        float r = s_colors[hit_block][0];
        float g = s_colors[hit_block][1];
        float b_col = s_colors[hit_block][2];

        float shade = 1.0f;
        if (hit_face == 0) shade = 0.8f;
        else if (hit_face == 2) shade = 0.9f;
        else if (hit_face == 1 && step_y < 0) shade = 0.6f;

        float fog = fminf(t / view_distance, 1.0f);
        fog = fog * fog;

        output[out_idx + 0] = r * shade * (1.0f - fog) + s_sky[0] * fog;
        output[out_idx + 1] = g * shade * (1.0f - fog) + s_sky[1] * fog;
        output[out_idx + 2] = b_col * shade * (1.0f - fog) + s_sky[2] * fog;
    } else {
        output[out_idx + 0] = s_sky[0];
        output[out_idx + 1] = s_sky[1];
        output[out_idx + 2] = s_sky[2];
    }
}


// =============================================================================
// VARIANT 2: __ldg() Intrinsic for Read-Only Voxel Data
// Optimization: Use texture cache path for voxel reads
// =============================================================================

__global__ void render_kernel_v2_ldg(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ cameras,
    float* __restrict__ output,
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

    int b = idx / (height * width);
    int pixel_idx = idx % (height * width);
    int py = pixel_idx / width;
    int px = pixel_idx % width;

    // Use __ldg for camera reads
    float cam_x = __ldg(&cameras[b * 5 + 0]);
    float cam_y = __ldg(&cameras[b * 5 + 1]);
    float cam_z = __ldg(&cameras[b * 5 + 2]);
    float yaw = __ldg(&cameras[b * 5 + 3]);
    float pitch = __ldg(&cameras[b * 5 + 4]);

    float eye_x = cam_x;
    float eye_y = cam_y + 1.6f;
    float eye_z = cam_z;

    float cos_yaw = cosf(yaw);
    float sin_yaw = sinf(yaw);
    float cos_pitch = cosf(pitch);
    float sin_pitch = sinf(pitch);

    float forward_x = sin_yaw * cos_pitch;
    float forward_y = sin_pitch;
    float forward_z = cos_yaw * cos_pitch;

    float right_x = cos_yaw;
    float right_z = -sin_yaw;

    float up_x = -sin_yaw * sin_pitch;
    float up_y = cos_pitch;
    float up_z = -cos_yaw * sin_pitch;

    float fov_rad = fov_degrees * 3.14159265f / 180.0f;
    float aspect = (float)width / (float)height;
    float half_fov = tanf(fov_rad * 0.5f);

    float u = (2.0f * (px + 0.5f) / width - 1.0f) * half_fov * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * half_fov;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    float ray_len = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    ray_x /= ray_len;
    ray_y /= ray_len;
    ray_z /= ray_len;

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
    int hit_face = -1;
    int8_t hit_block = AIR;

    for (int step = 0; step < max_steps && t < view_distance; ++step) {
        if (voxel_x >= 0 && voxel_x < ws &&
            voxel_y >= 0 && voxel_y < ws &&
            voxel_z >= 0 && voxel_z < ws) {

            int voxel_idx = b * ws3 + voxel_y * ws2 + voxel_x * ws + voxel_z;
            // Use __ldg for read-only voxel access
            int8_t block = __ldg(&voxels[voxel_idx]);

            if (block >= 0) {
                hit_block = block;
                break;
            }
        }

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

    int out_idx = (b * height * width + py * width + px) * 3;

    if (hit_block >= 0) {
        float r = BLOCK_COLORS_OPT[hit_block][0];
        float g = BLOCK_COLORS_OPT[hit_block][1];
        float b_col = BLOCK_COLORS_OPT[hit_block][2];

        float shade = 1.0f;
        if (hit_face == 0) shade = 0.8f;
        else if (hit_face == 2) shade = 0.9f;
        else if (hit_face == 1 && step_y < 0) shade = 0.6f;

        float fog = fminf(t / view_distance, 1.0f);
        fog = fog * fog;

        output[out_idx + 0] = r * shade * (1.0f - fog) + SKY_COLOR_OPT[0] * fog;
        output[out_idx + 1] = g * shade * (1.0f - fog) + SKY_COLOR_OPT[1] * fog;
        output[out_idx + 2] = b_col * shade * (1.0f - fog) + SKY_COLOR_OPT[2] * fog;
    } else {
        output[out_idx + 0] = SKY_COLOR_OPT[0];
        output[out_idx + 1] = SKY_COLOR_OPT[1];
        output[out_idx + 2] = SKY_COLOR_OPT[2];
    }
}


// =============================================================================
// VARIANT 3: Branchless DDA Stepping
// Optimization: Use fminf-based selection to reduce warp divergence
// =============================================================================

__global__ void render_kernel_v3_branchless(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ cameras,
    float* __restrict__ output,
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

    int b = idx / (height * width);
    int pixel_idx = idx % (height * width);
    int py = pixel_idx / width;
    int px = pixel_idx % width;

    float cam_x = cameras[b * 5 + 0];
    float cam_y = cameras[b * 5 + 1];
    float cam_z = cameras[b * 5 + 2];
    float yaw = cameras[b * 5 + 3];
    float pitch = cameras[b * 5 + 4];

    float eye_x = cam_x;
    float eye_y = cam_y + 1.6f;
    float eye_z = cam_z;

    float cos_yaw = cosf(yaw);
    float sin_yaw = sinf(yaw);
    float cos_pitch = cosf(pitch);
    float sin_pitch = sinf(pitch);

    float forward_x = sin_yaw * cos_pitch;
    float forward_y = sin_pitch;
    float forward_z = cos_yaw * cos_pitch;

    float right_x = cos_yaw;
    float right_z = -sin_yaw;

    float up_x = -sin_yaw * sin_pitch;
    float up_y = cos_pitch;
    float up_z = -cos_yaw * sin_pitch;

    float fov_rad = fov_degrees * 3.14159265f / 180.0f;
    float aspect = (float)width / (float)height;
    float half_fov = tanf(fov_rad * 0.5f);

    float u = (2.0f * (px + 0.5f) / width - 1.0f) * half_fov * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * half_fov;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    float ray_len = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    ray_x /= ray_len;
    ray_y /= ray_len;
    ray_z /= ray_len;

    int ws = world_size;
    int ws2 = ws * ws;
    int ws3 = ws * ws * ws;

    float pos_x = eye_x;
    float pos_y = eye_y;
    float pos_z = eye_z;

    int voxel_x = (int)floorf(pos_x);
    int voxel_y = (int)floorf(pos_y);
    int voxel_z = (int)floorf(pos_z);

    // Branchless step direction
    int step_x = (ray_x >= 0) ? 1 : -1;
    int step_y = (ray_y >= 0) ? 1 : -1;
    int step_z = (ray_z >= 0) ? 1 : -1;

    float inv_ray_x = (ray_x != 0.0f) ? 1.0f / ray_x : 1e30f;
    float inv_ray_y = (ray_y != 0.0f) ? 1.0f / ray_y : 1e30f;
    float inv_ray_z = (ray_z != 0.0f) ? 1.0f / ray_z : 1e30f;

    float t_delta_x = fabsf(inv_ray_x);
    float t_delta_y = fabsf(inv_ray_y);
    float t_delta_z = fabsf(inv_ray_z);

    // Branchless t_max initialization
    float t_max_x = ((float)(voxel_x + (step_x > 0)) - pos_x) * inv_ray_x;
    float t_max_y = ((float)(voxel_y + (step_y > 0)) - pos_y) * inv_ray_y;
    float t_max_z = ((float)(voxel_z + (step_z > 0)) - pos_z) * inv_ray_z;

    float t = 0.0f;
    int hit_face = -1;
    int8_t hit_block = AIR;

    for (int step = 0; step < max_steps && t < view_distance; ++step) {
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

        // Branchless DDA stepping using fminf
        float t_min = fminf(fminf(t_max_x, t_max_y), t_max_z);
        t = t_min;

        // Determine which axis was chosen (branchless-ish with predication)
        int is_x = (t_max_x <= t_max_y) && (t_max_x <= t_max_z);
        int is_y = (t_max_y < t_max_x) && (t_max_y <= t_max_z);
        int is_z = !is_x && !is_y;

        // Update state based on chosen axis
        t_max_x += is_x ? t_delta_x : 0.0f;
        t_max_y += is_y ? t_delta_y : 0.0f;
        t_max_z += is_z ? t_delta_z : 0.0f;

        voxel_x += is_x ? step_x : 0;
        voxel_y += is_y ? step_y : 0;
        voxel_z += is_z ? step_z : 0;

        hit_face = is_x * 0 + is_y * 1 + is_z * 2;
    }

    int out_idx = (b * height * width + py * width + px) * 3;

    if (hit_block >= 0) {
        float r = BLOCK_COLORS_OPT[hit_block][0];
        float g = BLOCK_COLORS_OPT[hit_block][1];
        float b_col = BLOCK_COLORS_OPT[hit_block][2];

        float shade = 1.0f;
        if (hit_face == 0) shade = 0.8f;
        else if (hit_face == 2) shade = 0.9f;
        else if (hit_face == 1 && step_y < 0) shade = 0.6f;

        float fog = fminf(t / view_distance, 1.0f);
        fog = fog * fog;

        output[out_idx + 0] = r * shade * (1.0f - fog) + SKY_COLOR_OPT[0] * fog;
        output[out_idx + 1] = g * shade * (1.0f - fog) + SKY_COLOR_OPT[1] * fog;
        output[out_idx + 2] = b_col * shade * (1.0f - fog) + SKY_COLOR_OPT[2] * fog;
    } else {
        output[out_idx + 0] = SKY_COLOR_OPT[0];
        output[out_idx + 1] = SKY_COLOR_OPT[1];
        output[out_idx + 2] = SKY_COLOR_OPT[2];
    }
}


// =============================================================================
// VARIANT 4: Loop Unrolling with #pragma unroll
// Optimization: Unroll DDA loop partially (4x)
// =============================================================================

__global__ void render_kernel_v4_unrolled(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ cameras,
    float* __restrict__ output,
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

    int b = idx / (height * width);
    int pixel_idx = idx % (height * width);
    int py = pixel_idx / width;
    int px = pixel_idx % width;

    float cam_x = cameras[b * 5 + 0];
    float cam_y = cameras[b * 5 + 1];
    float cam_z = cameras[b * 5 + 2];
    float yaw = cameras[b * 5 + 3];
    float pitch = cameras[b * 5 + 4];

    float eye_x = cam_x;
    float eye_y = cam_y + 1.6f;
    float eye_z = cam_z;

    float cos_yaw = cosf(yaw);
    float sin_yaw = sinf(yaw);
    float cos_pitch = cosf(pitch);
    float sin_pitch = sinf(pitch);

    float forward_x = sin_yaw * cos_pitch;
    float forward_y = sin_pitch;
    float forward_z = cos_yaw * cos_pitch;

    float right_x = cos_yaw;
    float right_z = -sin_yaw;

    float up_x = -sin_yaw * sin_pitch;
    float up_y = cos_pitch;
    float up_z = -cos_yaw * sin_pitch;

    float fov_rad = fov_degrees * 3.14159265f / 180.0f;
    float aspect = (float)width / (float)height;
    float half_fov = tanf(fov_rad * 0.5f);

    float u = (2.0f * (px + 0.5f) / width - 1.0f) * half_fov * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * half_fov;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    float ray_len = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    ray_x /= ray_len;
    ray_y /= ray_len;
    ray_z /= ray_len;

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
    int hit_face = -1;
    int8_t hit_block = AIR;

    // Unrolled loop with early exit checks
    int step = 0;

    #define DDA_STEP() \
        if (step >= max_steps || t >= view_distance) goto done; \
        if (voxel_x >= 0 && voxel_x < ws && \
            voxel_y >= 0 && voxel_y < ws && \
            voxel_z >= 0 && voxel_z < ws) { \
            int voxel_idx = b * ws3 + voxel_y * ws2 + voxel_x * ws + voxel_z; \
            int8_t block = voxels[voxel_idx]; \
            if (block >= 0) { \
                hit_block = block; \
                goto done; \
            } \
        } \
        if (t_max_x < t_max_y && t_max_x < t_max_z) { \
            t = t_max_x; \
            t_max_x += t_delta_x; \
            voxel_x += step_x; \
            hit_face = 0; \
        } else if (t_max_y < t_max_z) { \
            t = t_max_y; \
            t_max_y += t_delta_y; \
            voxel_y += step_y; \
            hit_face = 1; \
        } else { \
            t = t_max_z; \
            t_max_z += t_delta_z; \
            voxel_z += step_z; \
            hit_face = 2; \
        } \
        ++step;

    // Unroll 4x per iteration (max_steps is typically 64, so 16 iterations)
    #pragma unroll 4
    for (int i = 0; i < max_steps; i += 4) {
        DDA_STEP();
        DDA_STEP();
        DDA_STEP();
        DDA_STEP();
    }

    #undef DDA_STEP

done:
    int out_idx = (b * height * width + py * width + px) * 3;

    if (hit_block >= 0) {
        float r = BLOCK_COLORS_OPT[hit_block][0];
        float g = BLOCK_COLORS_OPT[hit_block][1];
        float b_col = BLOCK_COLORS_OPT[hit_block][2];

        float shade = 1.0f;
        if (hit_face == 0) shade = 0.8f;
        else if (hit_face == 2) shade = 0.9f;
        else if (hit_face == 1 && step_y < 0) shade = 0.6f;

        float fog = fminf(t / view_distance, 1.0f);
        fog = fog * fog;

        output[out_idx + 0] = r * shade * (1.0f - fog) + SKY_COLOR_OPT[0] * fog;
        output[out_idx + 1] = g * shade * (1.0f - fog) + SKY_COLOR_OPT[1] * fog;
        output[out_idx + 2] = b_col * shade * (1.0f - fog) + SKY_COLOR_OPT[2] * fog;
    } else {
        output[out_idx + 0] = SKY_COLOR_OPT[0];
        output[out_idx + 1] = SKY_COLOR_OPT[1];
        output[out_idx + 2] = SKY_COLOR_OPT[2];
    }
}


// =============================================================================
// VARIANT 5: Combined - __ldg + Shared Colors + Branchless t_max init
// Optimization: Combine multiple optimizations
// =============================================================================

__global__ void render_kernel_v5_combined(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ cameras,
    float* __restrict__ output,
    int batch_size,
    int world_size,
    int width,
    int height,
    int max_steps,
    float view_distance,
    float fov_degrees
) {
    __shared__ float s_colors[13][3];
    __shared__ float s_sky[3];

    int tid = threadIdx.x;
    if (tid < 39) {
        int block_idx = tid / 3;
        int channel = tid % 3;
        s_colors[block_idx][channel] = BLOCK_COLORS_OPT[block_idx][channel];
    }
    if (tid < 3) {
        s_sky[tid] = SKY_COLOR_OPT[tid];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = batch_size * height * width;
    if (idx >= total_pixels) return;

    int b = idx / (height * width);
    int pixel_idx = idx % (height * width);
    int py = pixel_idx / width;
    int px = pixel_idx % width;

    float cam_x = __ldg(&cameras[b * 5 + 0]);
    float cam_y = __ldg(&cameras[b * 5 + 1]);
    float cam_z = __ldg(&cameras[b * 5 + 2]);
    float yaw = __ldg(&cameras[b * 5 + 3]);
    float pitch = __ldg(&cameras[b * 5 + 4]);

    float eye_x = cam_x;
    float eye_y = cam_y + 1.6f;
    float eye_z = cam_z;

    float cos_yaw = cosf(yaw);
    float sin_yaw = sinf(yaw);
    float cos_pitch = cosf(pitch);
    float sin_pitch = sinf(pitch);

    float forward_x = sin_yaw * cos_pitch;
    float forward_y = sin_pitch;
    float forward_z = cos_yaw * cos_pitch;

    float right_x = cos_yaw;
    float right_z = -sin_yaw;

    float up_x = -sin_yaw * sin_pitch;
    float up_y = cos_pitch;
    float up_z = -cos_yaw * sin_pitch;

    float fov_rad = fov_degrees * 3.14159265f / 180.0f;
    float aspect = (float)width / (float)height;
    float half_fov = tanf(fov_rad * 0.5f);

    float u = (2.0f * (px + 0.5f) / width - 1.0f) * half_fov * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * half_fov;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    float ray_len = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    ray_x /= ray_len;
    ray_y /= ray_len;
    ray_z /= ray_len;

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

    // Branchless t_max initialization
    float t_max_x = ((float)(voxel_x + (step_x > 0)) - pos_x) * inv_ray_x;
    float t_max_y = ((float)(voxel_y + (step_y > 0)) - pos_y) * inv_ray_y;
    float t_max_z = ((float)(voxel_z + (step_z > 0)) - pos_z) * inv_ray_z;

    float t = 0.0f;
    int hit_face = -1;
    int8_t hit_block = AIR;

    for (int step = 0; step < max_steps && t < view_distance; ++step) {
        if (voxel_x >= 0 && voxel_x < ws &&
            voxel_y >= 0 && voxel_y < ws &&
            voxel_z >= 0 && voxel_z < ws) {

            int voxel_idx = b * ws3 + voxel_y * ws2 + voxel_x * ws + voxel_z;
            int8_t block = __ldg(&voxels[voxel_idx]);

            if (block >= 0) {
                hit_block = block;
                break;
            }
        }

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

    int out_idx = (b * height * width + py * width + px) * 3;

    if (hit_block >= 0) {
        float r = s_colors[hit_block][0];
        float g = s_colors[hit_block][1];
        float b_col = s_colors[hit_block][2];

        float shade = 1.0f;
        if (hit_face == 0) shade = 0.8f;
        else if (hit_face == 2) shade = 0.9f;
        else if (hit_face == 1 && step_y < 0) shade = 0.6f;

        float fog = fminf(t / view_distance, 1.0f);
        fog = fog * fog;

        output[out_idx + 0] = r * shade * (1.0f - fog) + s_sky[0] * fog;
        output[out_idx + 1] = g * shade * (1.0f - fog) + s_sky[1] * fog;
        output[out_idx + 2] = b_col * shade * (1.0f - fog) + s_sky[2] * fog;
    } else {
        output[out_idx + 0] = s_sky[0];
        output[out_idx + 1] = s_sky[1];
        output[out_idx + 2] = s_sky[2];
    }
}


// =============================================================================
// VARIANT 6: Early Ray Termination + Faster Trig
// Optimization: Use sincosf, early world bounds check
// =============================================================================

__global__ void render_kernel_v6_fast_math(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ cameras,
    float* __restrict__ output,
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

    int b = idx / (height * width);
    int pixel_idx = idx % (height * width);
    int py = pixel_idx / width;
    int px = pixel_idx % width;

    float cam_x = cameras[b * 5 + 0];
    float cam_y = cameras[b * 5 + 1];
    float cam_z = cameras[b * 5 + 2];
    float yaw = cameras[b * 5 + 3];
    float pitch = cameras[b * 5 + 4];

    float eye_x = cam_x;
    float eye_y = cam_y + 1.6f;
    float eye_z = cam_z;

    // Use sincosf for combined sin/cos computation
    float cos_yaw, sin_yaw;
    __sincosf(yaw, &sin_yaw, &cos_yaw);
    float cos_pitch, sin_pitch;
    __sincosf(pitch, &sin_pitch, &cos_pitch);

    float forward_x = sin_yaw * cos_pitch;
    float forward_y = sin_pitch;
    float forward_z = cos_yaw * cos_pitch;

    float right_x = cos_yaw;
    float right_z = -sin_yaw;

    float up_x = -sin_yaw * sin_pitch;
    float up_y = cos_pitch;
    float up_z = -cos_yaw * sin_pitch;

    float fov_rad = fov_degrees * 3.14159265f / 180.0f;
    float aspect = (float)width / (float)height;
    float half_fov = __tanf(fov_rad * 0.5f);

    float u = (2.0f * (px + 0.5f) / width - 1.0f) * half_fov * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * half_fov;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    // Use fast reciprocal sqrt
    float ray_len_sq = ray_x * ray_x + ray_y * ray_y + ray_z * ray_z;
    float inv_ray_len = __frsqrt_rn(ray_len_sq);
    ray_x *= inv_ray_len;
    ray_y *= inv_ray_len;
    ray_z *= inv_ray_len;

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

    // Use fast reciprocal
    float inv_ray_x = (fabsf(ray_x) > 1e-6f) ? __frcp_rn(ray_x) : 1e30f;
    float inv_ray_y = (fabsf(ray_y) > 1e-6f) ? __frcp_rn(ray_y) : 1e30f;
    float inv_ray_z = (fabsf(ray_z) > 1e-6f) ? __frcp_rn(ray_z) : 1e30f;

    float t_delta_x = fabsf(inv_ray_x);
    float t_delta_y = fabsf(inv_ray_y);
    float t_delta_z = fabsf(inv_ray_z);

    float t_max_x = ((float)(voxel_x + (step_x > 0)) - pos_x) * inv_ray_x;
    float t_max_y = ((float)(voxel_y + (step_y > 0)) - pos_y) * inv_ray_y;
    float t_max_z = ((float)(voxel_z + (step_z > 0)) - pos_z) * inv_ray_z;

    float t = 0.0f;
    int hit_face = -1;
    int8_t hit_block = AIR;

    for (int step = 0; step < max_steps && t < view_distance; ++step) {
        // Early termination if completely outside world bounds
        bool outside_x = (voxel_x < 0 && step_x < 0) || (voxel_x >= ws && step_x > 0);
        bool outside_y = (voxel_y < 0 && step_y < 0) || (voxel_y >= ws && step_y > 0);
        bool outside_z = (voxel_z < 0 && step_z < 0) || (voxel_z >= ws && step_z > 0);
        if (outside_x || outside_y || outside_z) break;

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

    int out_idx = (b * height * width + py * width + px) * 3;

    if (hit_block >= 0) {
        float r = BLOCK_COLORS_OPT[hit_block][0];
        float g = BLOCK_COLORS_OPT[hit_block][1];
        float b_col = BLOCK_COLORS_OPT[hit_block][2];

        float shade = 1.0f;
        if (hit_face == 0) shade = 0.8f;
        else if (hit_face == 2) shade = 0.9f;
        else if (hit_face == 1 && step_y < 0) shade = 0.6f;

        float fog = fminf(t / view_distance, 1.0f);
        fog = fog * fog;

        output[out_idx + 0] = r * shade * (1.0f - fog) + SKY_COLOR_OPT[0] * fog;
        output[out_idx + 1] = g * shade * (1.0f - fog) + SKY_COLOR_OPT[1] * fog;
        output[out_idx + 2] = b_col * shade * (1.0f - fog) + SKY_COLOR_OPT[2] * fog;
    } else {
        output[out_idx + 0] = SKY_COLOR_OPT[0];
        output[out_idx + 1] = SKY_COLOR_OPT[1];
        output[out_idx + 2] = SKY_COLOR_OPT[2];
    }
}


// =============================================================================
// VARIANT 7: Full Combined - All optimizations together
// =============================================================================

__global__ void render_kernel_v7_full(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ cameras,
    float* __restrict__ output,
    int batch_size,
    int world_size,
    int width,
    int height,
    int max_steps,
    float view_distance,
    float fov_degrees
) {
    __shared__ float s_colors[13][3];
    __shared__ float s_sky[3];

    int tid = threadIdx.x;
    if (tid < 39) {
        int block_idx = tid / 3;
        int channel = tid % 3;
        s_colors[block_idx][channel] = BLOCK_COLORS_OPT[block_idx][channel];
    }
    if (tid < 3) {
        s_sky[tid] = SKY_COLOR_OPT[tid];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = batch_size * height * width;
    if (idx >= total_pixels) return;

    int b = idx / (height * width);
    int pixel_idx = idx % (height * width);
    int py = pixel_idx / width;
    int px = pixel_idx % width;

    float cam_x = __ldg(&cameras[b * 5 + 0]);
    float cam_y = __ldg(&cameras[b * 5 + 1]);
    float cam_z = __ldg(&cameras[b * 5 + 2]);
    float yaw = __ldg(&cameras[b * 5 + 3]);
    float pitch = __ldg(&cameras[b * 5 + 4]);

    float eye_x = cam_x;
    float eye_y = cam_y + 1.6f;
    float eye_z = cam_z;

    float cos_yaw, sin_yaw;
    __sincosf(yaw, &sin_yaw, &cos_yaw);
    float cos_pitch, sin_pitch;
    __sincosf(pitch, &sin_pitch, &cos_pitch);

    float forward_x = sin_yaw * cos_pitch;
    float forward_y = sin_pitch;
    float forward_z = cos_yaw * cos_pitch;

    float right_x = cos_yaw;
    float right_z = -sin_yaw;

    float up_x = -sin_yaw * sin_pitch;
    float up_y = cos_pitch;
    float up_z = -cos_yaw * sin_pitch;

    float fov_rad = fov_degrees * 3.14159265f / 180.0f;
    float aspect = (float)width / (float)height;
    float half_fov = __tanf(fov_rad * 0.5f);

    float u = (2.0f * (px + 0.5f) / width - 1.0f) * half_fov * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * half_fov;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    float ray_len_sq = ray_x * ray_x + ray_y * ray_y + ray_z * ray_z;
    float inv_ray_len = __frsqrt_rn(ray_len_sq);
    ray_x *= inv_ray_len;
    ray_y *= inv_ray_len;
    ray_z *= inv_ray_len;

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

    float inv_ray_x = (fabsf(ray_x) > 1e-6f) ? __frcp_rn(ray_x) : 1e30f;
    float inv_ray_y = (fabsf(ray_y) > 1e-6f) ? __frcp_rn(ray_y) : 1e30f;
    float inv_ray_z = (fabsf(ray_z) > 1e-6f) ? __frcp_rn(ray_z) : 1e30f;

    float t_delta_x = fabsf(inv_ray_x);
    float t_delta_y = fabsf(inv_ray_y);
    float t_delta_z = fabsf(inv_ray_z);

    float t_max_x = ((float)(voxel_x + (step_x > 0)) - pos_x) * inv_ray_x;
    float t_max_y = ((float)(voxel_y + (step_y > 0)) - pos_y) * inv_ray_y;
    float t_max_z = ((float)(voxel_z + (step_z > 0)) - pos_z) * inv_ray_z;

    float t = 0.0f;
    int hit_face = -1;
    int8_t hit_block = AIR;

    for (int step = 0; step < max_steps && t < view_distance; ++step) {
        bool outside_x = (voxel_x < 0 && step_x < 0) || (voxel_x >= ws && step_x > 0);
        bool outside_y = (voxel_y < 0 && step_y < 0) || (voxel_y >= ws && step_y > 0);
        bool outside_z = (voxel_z < 0 && step_z < 0) || (voxel_z >= ws && step_z > 0);
        if (outside_x || outside_y || outside_z) break;

        if (voxel_x >= 0 && voxel_x < ws &&
            voxel_y >= 0 && voxel_y < ws &&
            voxel_z >= 0 && voxel_z < ws) {

            int voxel_idx = b * ws3 + voxel_y * ws2 + voxel_x * ws + voxel_z;
            int8_t block = __ldg(&voxels[voxel_idx]);

            if (block >= 0) {
                hit_block = block;
                break;
            }
        }

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

    int out_idx = (b * height * width + py * width + px) * 3;

    if (hit_block >= 0) {
        float r = s_colors[hit_block][0];
        float g = s_colors[hit_block][1];
        float b_col = s_colors[hit_block][2];

        float shade = 1.0f;
        if (hit_face == 0) shade = 0.8f;
        else if (hit_face == 2) shade = 0.9f;
        else if (hit_face == 1 && step_y < 0) shade = 0.6f;

        float fog = fminf(t / view_distance, 1.0f);
        fog = fog * fog;

        output[out_idx + 0] = r * shade * (1.0f - fog) + s_sky[0] * fog;
        output[out_idx + 1] = g * shade * (1.0f - fog) + s_sky[1] * fog;
        output[out_idx + 2] = b_col * shade * (1.0f - fog) + s_sky[2] * fog;
    } else {
        output[out_idx + 0] = s_sky[0];
        output[out_idx + 1] = s_sky[1];
        output[out_idx + 2] = s_sky[2];
    }
}


// =============================================================================
// VARIANT 8: Unrolled + __ldg (combining top two optimizations)
// =============================================================================

__global__ void render_kernel_v8_unrolled_ldg(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ cameras,
    float* __restrict__ output,
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

    int b = idx / (height * width);
    int pixel_idx = idx % (height * width);
    int py = pixel_idx / width;
    int px = pixel_idx % width;

    float cam_x = __ldg(&cameras[b * 5 + 0]);
    float cam_y = __ldg(&cameras[b * 5 + 1]);
    float cam_z = __ldg(&cameras[b * 5 + 2]);
    float yaw = __ldg(&cameras[b * 5 + 3]);
    float pitch = __ldg(&cameras[b * 5 + 4]);

    float eye_x = cam_x;
    float eye_y = cam_y + 1.6f;
    float eye_z = cam_z;

    float cos_yaw = cosf(yaw);
    float sin_yaw = sinf(yaw);
    float cos_pitch = cosf(pitch);
    float sin_pitch = sinf(pitch);

    float forward_x = sin_yaw * cos_pitch;
    float forward_y = sin_pitch;
    float forward_z = cos_yaw * cos_pitch;

    float right_x = cos_yaw;
    float right_z = -sin_yaw;

    float up_x = -sin_yaw * sin_pitch;
    float up_y = cos_pitch;
    float up_z = -cos_yaw * sin_pitch;

    float fov_rad = fov_degrees * 3.14159265f / 180.0f;
    float aspect = (float)width / (float)height;
    float half_fov = tanf(fov_rad * 0.5f);

    float u = (2.0f * (px + 0.5f) / width - 1.0f) * half_fov * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * half_fov;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    float ray_len = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    ray_x /= ray_len;
    ray_y /= ray_len;
    ray_z /= ray_len;

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
    int hit_face = -1;
    int8_t hit_block = AIR;

    int step = 0;

    #define DDA_STEP_LDG() \
        if (step >= max_steps || t >= view_distance) goto done8; \
        if (voxel_x >= 0 && voxel_x < ws && \
            voxel_y >= 0 && voxel_y < ws && \
            voxel_z >= 0 && voxel_z < ws) { \
            int voxel_idx = b * ws3 + voxel_y * ws2 + voxel_x * ws + voxel_z; \
            int8_t block = __ldg(&voxels[voxel_idx]); \
            if (block >= 0) { \
                hit_block = block; \
                goto done8; \
            } \
        } \
        if (t_max_x < t_max_y && t_max_x < t_max_z) { \
            t = t_max_x; \
            t_max_x += t_delta_x; \
            voxel_x += step_x; \
            hit_face = 0; \
        } else if (t_max_y < t_max_z) { \
            t = t_max_y; \
            t_max_y += t_delta_y; \
            voxel_y += step_y; \
            hit_face = 1; \
        } else { \
            t = t_max_z; \
            t_max_z += t_delta_z; \
            voxel_z += step_z; \
            hit_face = 2; \
        } \
        ++step;

    #pragma unroll 4
    for (int i = 0; i < max_steps; i += 4) {
        DDA_STEP_LDG();
        DDA_STEP_LDG();
        DDA_STEP_LDG();
        DDA_STEP_LDG();
    }

    #undef DDA_STEP_LDG

done8:
    int out_idx = (b * height * width + py * width + px) * 3;

    if (hit_block >= 0) {
        float r = BLOCK_COLORS_OPT[hit_block][0];
        float g = BLOCK_COLORS_OPT[hit_block][1];
        float b_col = BLOCK_COLORS_OPT[hit_block][2];

        float shade = 1.0f;
        if (hit_face == 0) shade = 0.8f;
        else if (hit_face == 2) shade = 0.9f;
        else if (hit_face == 1 && step_y < 0) shade = 0.6f;

        float fog = fminf(t / view_distance, 1.0f);
        fog = fog * fog;

        output[out_idx + 0] = r * shade * (1.0f - fog) + SKY_COLOR_OPT[0] * fog;
        output[out_idx + 1] = g * shade * (1.0f - fog) + SKY_COLOR_OPT[1] * fog;
        output[out_idx + 2] = b_col * shade * (1.0f - fog) + SKY_COLOR_OPT[2] * fog;
    } else {
        output[out_idx + 0] = SKY_COLOR_OPT[0];
        output[out_idx + 1] = SKY_COLOR_OPT[1];
        output[out_idx + 2] = SKY_COLOR_OPT[2];
    }
}


// =============================================================================
// VARIANT 9: Unrolled + __ldg + fast intrinsics (best combination)
// =============================================================================

__global__ void render_kernel_v9_best(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ cameras,
    float* __restrict__ output,
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

    int b = idx / (height * width);
    int pixel_idx = idx % (height * width);
    int py = pixel_idx / width;
    int px = pixel_idx % width;

    float cam_x = __ldg(&cameras[b * 5 + 0]);
    float cam_y = __ldg(&cameras[b * 5 + 1]);
    float cam_z = __ldg(&cameras[b * 5 + 2]);
    float yaw = __ldg(&cameras[b * 5 + 3]);
    float pitch = __ldg(&cameras[b * 5 + 4]);

    float eye_x = cam_x;
    float eye_y = cam_y + 1.6f;
    float eye_z = cam_z;

    float cos_yaw, sin_yaw;
    __sincosf(yaw, &sin_yaw, &cos_yaw);
    float cos_pitch, sin_pitch;
    __sincosf(pitch, &sin_pitch, &cos_pitch);

    float forward_x = sin_yaw * cos_pitch;
    float forward_y = sin_pitch;
    float forward_z = cos_yaw * cos_pitch;

    float right_x = cos_yaw;
    float right_z = -sin_yaw;

    float up_x = -sin_yaw * sin_pitch;
    float up_y = cos_pitch;
    float up_z = -cos_yaw * sin_pitch;

    float fov_rad = fov_degrees * 3.14159265f / 180.0f;
    float aspect = (float)width / (float)height;
    float half_fov = __tanf(fov_rad * 0.5f);

    float u = (2.0f * (px + 0.5f) / width - 1.0f) * half_fov * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * half_fov;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    float ray_len_sq = ray_x * ray_x + ray_y * ray_y + ray_z * ray_z;
    float inv_ray_len = __frsqrt_rn(ray_len_sq);
    ray_x *= inv_ray_len;
    ray_y *= inv_ray_len;
    ray_z *= inv_ray_len;

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

    float inv_ray_x = (fabsf(ray_x) > 1e-6f) ? __frcp_rn(ray_x) : 1e30f;
    float inv_ray_y = (fabsf(ray_y) > 1e-6f) ? __frcp_rn(ray_y) : 1e30f;
    float inv_ray_z = (fabsf(ray_z) > 1e-6f) ? __frcp_rn(ray_z) : 1e30f;

    float t_delta_x = fabsf(inv_ray_x);
    float t_delta_y = fabsf(inv_ray_y);
    float t_delta_z = fabsf(inv_ray_z);

    float t_max_x = ((float)(voxel_x + (step_x > 0)) - pos_x) * inv_ray_x;
    float t_max_y = ((float)(voxel_y + (step_y > 0)) - pos_y) * inv_ray_y;
    float t_max_z = ((float)(voxel_z + (step_z > 0)) - pos_z) * inv_ray_z;

    float t = 0.0f;
    int hit_face = -1;
    int8_t hit_block = AIR;

    int step = 0;

    #define DDA_STEP_BEST() \
        if (step >= max_steps || t >= view_distance) goto done9; \
        if (voxel_x >= 0 && voxel_x < ws && \
            voxel_y >= 0 && voxel_y < ws && \
            voxel_z >= 0 && voxel_z < ws) { \
            int voxel_idx = b * ws3 + voxel_y * ws2 + voxel_x * ws + voxel_z; \
            int8_t block = __ldg(&voxels[voxel_idx]); \
            if (block >= 0) { \
                hit_block = block; \
                goto done9; \
            } \
        } \
        if (t_max_x < t_max_y && t_max_x < t_max_z) { \
            t = t_max_x; \
            t_max_x += t_delta_x; \
            voxel_x += step_x; \
            hit_face = 0; \
        } else if (t_max_y < t_max_z) { \
            t = t_max_y; \
            t_max_y += t_delta_y; \
            voxel_y += step_y; \
            hit_face = 1; \
        } else { \
            t = t_max_z; \
            t_max_z += t_delta_z; \
            voxel_z += step_z; \
            hit_face = 2; \
        } \
        ++step;

    #pragma unroll 4
    for (int i = 0; i < max_steps; i += 4) {
        DDA_STEP_BEST();
        DDA_STEP_BEST();
        DDA_STEP_BEST();
        DDA_STEP_BEST();
    }

    #undef DDA_STEP_BEST

done9:
    int out_idx = (b * height * width + py * width + px) * 3;

    if (hit_block >= 0) {
        float r = BLOCK_COLORS_OPT[hit_block][0];
        float g = BLOCK_COLORS_OPT[hit_block][1];
        float b_col = BLOCK_COLORS_OPT[hit_block][2];

        float shade = 1.0f;
        if (hit_face == 0) shade = 0.8f;
        else if (hit_face == 2) shade = 0.9f;
        else if (hit_face == 1 && step_y < 0) shade = 0.6f;

        float fog = fminf(t / view_distance, 1.0f);
        fog = fog * fog;

        output[out_idx + 0] = r * shade * (1.0f - fog) + SKY_COLOR_OPT[0] * fog;
        output[out_idx + 1] = g * shade * (1.0f - fog) + SKY_COLOR_OPT[1] * fog;
        output[out_idx + 2] = b_col * shade * (1.0f - fog) + SKY_COLOR_OPT[2] * fog;
    } else {
        output[out_idx + 0] = SKY_COLOR_OPT[0];
        output[out_idx + 1] = SKY_COLOR_OPT[1];
        output[out_idx + 2] = SKY_COLOR_OPT[2];
    }
}


// =============================================================================
// LAUNCHER FUNCTIONS - For each variant with configurable block size
// =============================================================================

extern "C" {

// Baseline (copy from original for comparison)
void launch_render_baseline(
    const int8_t* voxels, const float* cameras, float* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees, int block_size
);

void launch_render_v1_shared_colors(
    const int8_t* voxels, const float* cameras, float* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees, int block_size
) {
    int total_pixels = batch_size * height * width;
    int num_blocks = (total_pixels + block_size - 1) / block_size;
    render_kernel_v1_shared_colors<<<num_blocks, block_size>>>(
        voxels, cameras, output, batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void launch_render_v2_ldg(
    const int8_t* voxels, const float* cameras, float* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees, int block_size
) {
    int total_pixels = batch_size * height * width;
    int num_blocks = (total_pixels + block_size - 1) / block_size;
    render_kernel_v2_ldg<<<num_blocks, block_size>>>(
        voxels, cameras, output, batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void launch_render_v3_branchless(
    const int8_t* voxels, const float* cameras, float* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees, int block_size
) {
    int total_pixels = batch_size * height * width;
    int num_blocks = (total_pixels + block_size - 1) / block_size;
    render_kernel_v3_branchless<<<num_blocks, block_size>>>(
        voxels, cameras, output, batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void launch_render_v4_unrolled(
    const int8_t* voxels, const float* cameras, float* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees, int block_size
) {
    int total_pixels = batch_size * height * width;
    int num_blocks = (total_pixels + block_size - 1) / block_size;
    render_kernel_v4_unrolled<<<num_blocks, block_size>>>(
        voxels, cameras, output, batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void launch_render_v5_combined(
    const int8_t* voxels, const float* cameras, float* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees, int block_size
) {
    int total_pixels = batch_size * height * width;
    int num_blocks = (total_pixels + block_size - 1) / block_size;
    render_kernel_v5_combined<<<num_blocks, block_size>>>(
        voxels, cameras, output, batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void launch_render_v6_fast_math(
    const int8_t* voxels, const float* cameras, float* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees, int block_size
) {
    int total_pixels = batch_size * height * width;
    int num_blocks = (total_pixels + block_size - 1) / block_size;
    render_kernel_v6_fast_math<<<num_blocks, block_size>>>(
        voxels, cameras, output, batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void launch_render_v7_full(
    const int8_t* voxels, const float* cameras, float* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees, int block_size
) {
    int total_pixels = batch_size * height * width;
    int num_blocks = (total_pixels + block_size - 1) / block_size;
    render_kernel_v7_full<<<num_blocks, block_size>>>(
        voxels, cameras, output, batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void launch_render_v8_unrolled_ldg(
    const int8_t* voxels, const float* cameras, float* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees, int block_size
) {
    int total_pixels = batch_size * height * width;
    int num_blocks = (total_pixels + block_size - 1) / block_size;
    render_kernel_v8_unrolled_ldg<<<num_blocks, block_size>>>(
        voxels, cameras, output, batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void launch_render_v9_best(
    const int8_t* voxels, const float* cameras, float* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees, int block_size
) {
    int total_pixels = batch_size * height * width;
    int num_blocks = (total_pixels + block_size - 1) / block_size;
    render_kernel_v9_best<<<num_blocks, block_size>>>(
        voxels, cameras, output, batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

}  // extern "C"
