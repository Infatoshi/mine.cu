// mine.cu - High-performance batched voxel RL environment
// All environment logic runs on GPU via custom CUDA kernels

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>  // For cp.async (Ampere+)
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
// AABB RAY INTERSECTION - Early termination for rays missing the world
// =============================================================================

__device__ __forceinline__ bool ray_intersects_aabb(
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float minx, float miny, float minz,
    float maxx, float maxy, float maxz,
    float* t_entry, float* t_exit
) {
    float inv_dx = (dx != 0.0f) ? 1.0f / dx : 1e30f;
    float inv_dy = (dy != 0.0f) ? 1.0f / dy : 1e30f;
    float inv_dz = (dz != 0.0f) ? 1.0f / dz : 1e30f;

    float t1 = (minx - ox) * inv_dx;
    float t2 = (maxx - ox) * inv_dx;
    float t3 = (miny - oy) * inv_dy;
    float t4 = (maxy - oy) * inv_dy;
    float t5 = (minz - oz) * inv_dz;
    float t6 = (maxz - oz) * inv_dz;

    float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
    float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

    *t_entry = tmin;
    *t_exit = tmax;

    if (tmax < 0 || tmin > tmax) {
        return false;
    }
    return true;
}


// =============================================================================
// RENDER KERNEL - Optimized DDA raymarching
// Optimizations: AABB early termination, #pragma unroll 4, __ldg(), block_size=128
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

    // Camera params - use __ldg for read-only texture cache path
    float cam_x = __ldg(&cameras[b * 5 + 0]);
    float cam_y = __ldg(&cameras[b * 5 + 1]);
    float cam_z = __ldg(&cameras[b * 5 + 2]);
    float yaw = __ldg(&cameras[b * 5 + 3]);
    float pitch = __ldg(&cameras[b * 5 + 4]);

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
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    // Normalize ray
    float ray_len = sqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    ray_x /= ray_len;
    ray_y /= ray_len;
    ray_z /= ray_len;

    // Early termination: test if ray intersects world AABB
    int ws = world_size;
    float t_entry, t_exit;
    bool hits_world = ray_intersects_aabb(
        eye_x, eye_y, eye_z,
        ray_x, ray_y, ray_z,
        0.0f, 0.0f, 0.0f,
        (float)ws, (float)ws, (float)ws,
        &t_entry, &t_exit
    );

    long long out_idx = ((long long)b * height * width + py * width + px) * 3;

    // Skip DDA entirely if ray misses world or entry point is beyond view distance
    if (!hits_world || t_entry > view_distance) {
        output[out_idx + 0] = SKY_COLOR[0];
        output[out_idx + 1] = SKY_COLOR[1];
        output[out_idx + 2] = SKY_COLOR[2];
        return;
    }

    // DDA raymarching
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

    // Unrolled DDA loop with __ldg for voxel reads
    int step = 0;

    #define DDA_STEP() \
        if (step >= max_steps || t >= view_distance) goto render_done; \
        if (voxel_x >= 0 && voxel_x < ws && \
            voxel_y >= 0 && voxel_y < ws && \
            voxel_z >= 0 && voxel_z < ws) { \
            long long voxel_idx = (long long)b * ws3 + voxel_y * ws2 + voxel_x * ws + voxel_z; \
            int8_t block = __ldg(&voxels[voxel_idx]); \
            if (block >= 0) { \
                hit_block = block; \
                goto render_done; \
            } \
        } else if (t > t_exit) { \
            goto render_done; \
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

    // Unroll 4x per iteration for better instruction-level parallelism
    #pragma unroll 4
    for (int i = 0; i < max_steps; i += 4) {
        DDA_STEP();
        DDA_STEP();
        DDA_STEP();
        DDA_STEP();
    }

    #undef DDA_STEP

render_done:
    // Output color
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
// RENDER KERNEL (UINT8 output) - Memory-optimized version for high throughput
// =============================================================================

__global__ void render_kernel_uint8(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ cameras,
    uint8_t* __restrict__ output,  // [B, H, W, 3] uint8
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
    float t_entry, t_exit;
    bool hits_world = ray_intersects_aabb(
        eye_x, eye_y, eye_z,
        ray_x, ray_y, ray_z,
        0.0f, 0.0f, 0.0f,
        (float)ws, (float)ws, (float)ws,
        &t_entry, &t_exit
    );

    long long out_idx = ((long long)b * height * width + py * width + px) * 3;

    if (!hits_world || t_entry > view_distance) {
        output[out_idx + 0] = (uint8_t)(SKY_COLOR[0] * 255.0f);
        output[out_idx + 1] = (uint8_t)(SKY_COLOR[1] * 255.0f);
        output[out_idx + 2] = (uint8_t)(SKY_COLOR[2] * 255.0f);
        return;
    }

    int ws2 = ws * ws;
    int ws3 = ws * ws * ws;

    int voxel_x = (int)floorf(eye_x);
    int voxel_y = (int)floorf(eye_y);
    int voxel_z = (int)floorf(eye_z);

    int step_x = (ray_x >= 0) ? 1 : -1;
    int step_y = (ray_y >= 0) ? 1 : -1;
    int step_z = (ray_z >= 0) ? 1 : -1;

    float inv_ray_x = (ray_x != 0.0f) ? 1.0f / ray_x : 1e30f;
    float inv_ray_y = (ray_y != 0.0f) ? 1.0f / ray_y : 1e30f;
    float inv_ray_z = (ray_z != 0.0f) ? 1.0f / ray_z : 1e30f;

    float t_delta_x = fabsf(inv_ray_x);
    float t_delta_y = fabsf(inv_ray_y);
    float t_delta_z = fabsf(inv_ray_z);

    float t_max_x = (ray_x >= 0) ? ((float)(voxel_x + 1) - eye_x) * inv_ray_x
                                 : ((float)voxel_x - eye_x) * inv_ray_x;
    float t_max_y = (ray_y >= 0) ? ((float)(voxel_y + 1) - eye_y) * inv_ray_y
                                 : ((float)voxel_y - eye_y) * inv_ray_y;
    float t_max_z = (ray_z >= 0) ? ((float)(voxel_z + 1) - eye_z) * inv_ray_z
                                 : ((float)voxel_z - eye_z) * inv_ray_z;

    float t = 0.0f;
    int hit_face = -1;
    int8_t hit_block = AIR;
    int step = 0;

    #pragma unroll 4
    for (int i = 0; i < max_steps; i += 4) {
        #define DDA_STEP_U8() \
            if (step >= max_steps || t >= view_distance) goto render_done_u8; \
            if (voxel_x >= 0 && voxel_x < ws && \
                voxel_y >= 0 && voxel_y < ws && \
                voxel_z >= 0 && voxel_z < ws) { \
                long long voxel_idx = (long long)b * ws3 + voxel_y * ws2 + voxel_x * ws + voxel_z; \
                int8_t block = __ldg(&voxels[voxel_idx]); \
                if (block >= 0) { \
                    hit_block = block; \
                    goto render_done_u8; \
                } \
            } else if (t > t_exit) { \
                goto render_done_u8; \
            } \
            if (t_max_x < t_max_y && t_max_x < t_max_z) { \
                t = t_max_x; t_max_x += t_delta_x; voxel_x += step_x; hit_face = 0; \
            } else if (t_max_y < t_max_z) { \
                t = t_max_y; t_max_y += t_delta_y; voxel_y += step_y; hit_face = 1; \
            } else { \
                t = t_max_z; t_max_z += t_delta_z; voxel_z += step_z; hit_face = 2; \
            } \
            ++step;
        DDA_STEP_U8();
        DDA_STEP_U8();
        DDA_STEP_U8();
        DDA_STEP_U8();
        #undef DDA_STEP_U8
    }

render_done_u8:
    if (hit_block >= 0) {
        float r = BLOCK_COLORS[hit_block][0];
        float g = BLOCK_COLORS[hit_block][1];
        float b_col = BLOCK_COLORS[hit_block][2];

        float shade = 1.0f;
        if (hit_face == 0) shade = 0.8f;
        else if (hit_face == 2) shade = 0.9f;
        else if (hit_face == 1 && step_y < 0) shade = 0.6f;

        float fog = fminf(t / view_distance, 1.0f);
        fog = fog * fog;

        float final_r = r * shade * (1.0f - fog) + SKY_COLOR[0] * fog;
        float final_g = g * shade * (1.0f - fog) + SKY_COLOR[1] * fog;
        float final_b = b_col * shade * (1.0f - fog) + SKY_COLOR[2] * fog;

        output[out_idx + 0] = (uint8_t)(final_r * 255.0f);
        output[out_idx + 1] = (uint8_t)(final_g * 255.0f);
        output[out_idx + 2] = (uint8_t)(final_b * 255.0f);
    } else {
        output[out_idx + 0] = (uint8_t)(SKY_COLOR[0] * 255.0f);
        output[out_idx + 1] = (uint8_t)(SKY_COLOR[1] * 255.0f);
        output[out_idx + 2] = (uint8_t)(SKY_COLOR[2] * 255.0f);
    }
}


// =============================================================================
// RENDER KERNEL (UINT8, Minimal) - Maximum throughput, flat shading
// Removes: fog, face shading, distance fog. Just block colors.
// =============================================================================

// Pre-computed uint8 block colors (avoids float->uint8 conversion per pixel)
__constant__ uint8_t BLOCK_COLORS_U8[13][3] = {
    {76, 166, 51},    // GRASS
    {140, 89, 51},    // DIRT
    {127, 127, 127},  // STONE
    {140, 102, 64},   // OAKLOG
    {51, 140, 38},    // LEAVES
    {229, 217, 153},  // SAND
    {51, 102, 204},   // WATER
    {217, 229, 242},  // GLASS
    {178, 89, 76},    // BRICK
    {102, 102, 102},  // COBBLESTONE
    {191, 153, 102},  // PLANKS
    {242, 242, 250},  // SNOW
    {38, 38, 38},     // BEDROCK
};

__constant__ uint8_t SKY_COLOR_U8[3] = {135, 206, 235};


// =============================================================================
// PRECOMPUTE CAMERA BASIS - Compute trig once per camera, not per pixel
// =============================================================================

// Camera basis layout: [B, 14] floats per camera
// [0-2]: eye position (x, y+1.6, z)
// [3-5]: forward vector
// [6-7]: right vector (x, z) - right_y is always 0
// [8-10]: up vector
// [11]: scale_u = half_fov * aspect
// [12]: scale_v = half_fov
// [13]: (padding)

__global__ void precompute_camera_basis_kernel(
    const float* __restrict__ cameras,  // [B, 5]: x, y, z, yaw, pitch
    float* __restrict__ basis,          // [B, 14]
    int batch_size,
    int width,
    int height,
    float fov_degrees
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    float cam_x = cameras[b * 5 + 0];
    float cam_y = cameras[b * 5 + 1];
    float cam_z = cameras[b * 5 + 2];
    float yaw = cameras[b * 5 + 3];
    float pitch = cameras[b * 5 + 4];

    // Use __sincosf for efficiency - computes both in one call
    float sin_yaw, cos_yaw, sin_pitch, cos_pitch;
    __sincosf(yaw, &sin_yaw, &cos_yaw);
    __sincosf(pitch, &sin_pitch, &cos_pitch);

    float fov_rad = fov_degrees * 0.01745329f;
    float half_fov = tanf(fov_rad * 0.5f);
    float aspect = (float)width / (float)height;

    int idx = b * 14;

    // Eye position
    basis[idx + 0] = cam_x;
    basis[idx + 1] = cam_y + 1.6f;
    basis[idx + 2] = cam_z;

    // Forward vector
    basis[idx + 3] = sin_yaw * cos_pitch;
    basis[idx + 4] = sin_pitch;
    basis[idx + 5] = cos_yaw * cos_pitch;

    // Right vector (right_y = 0, stored implicitly)
    basis[idx + 6] = cos_yaw;
    basis[idx + 7] = -sin_yaw;

    // Up vector
    basis[idx + 8] = -sin_yaw * sin_pitch;
    basis[idx + 9] = cos_pitch;
    basis[idx + 10] = -cos_yaw * sin_pitch;

    // FOV scaling
    basis[idx + 11] = half_fov * aspect;  // scale_u
    basis[idx + 12] = half_fov;           // scale_v
}


// =============================================================================
// RENDER KERNEL (UINT8, Precomputed Basis) - No per-pixel trig
// =============================================================================

__global__ void render_kernel_uint8_prebasis(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ basis,  // [B, 14] precomputed camera basis
    uint8_t* __restrict__ output,
    int batch_size,
    int world_x,
    int world_y,
    int world_z,
    int width,
    int height,
    int max_steps
) {
    // Shared memory for camera basis - loaded once per block, shared by all threads
    __shared__ float s_basis[14];

    // 2D grid: blockIdx.y = batch element, blockIdx.x = pixel block within batch
    int b = blockIdx.y;
    int pixel_offset = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels_per_batch = height * width;

    if (b >= batch_size || pixel_offset >= pixels_per_batch) return;

    int py = pixel_offset / width;
    int px = pixel_offset % width;

    // Load basis into shared memory (first 14 threads load, then sync)
    if (threadIdx.x < 14) {
        s_basis[threadIdx.x] = basis[b * 14 + threadIdx.x];
    }
    __syncthreads();

    // Read from shared memory (fast!)
    float eye_x = s_basis[0];
    float eye_y = s_basis[1];
    float eye_z = s_basis[2];

    float forward_x = s_basis[3];
    float forward_y = s_basis[4];
    float forward_z = s_basis[5];

    float right_x = s_basis[6];
    float right_z = s_basis[7];

    float up_x = s_basis[8];
    float up_y = s_basis[9];
    float up_z = s_basis[10];

    float scale_u = s_basis[11];
    float scale_v = s_basis[12];

    // Ray direction from pixel coordinates
    float u = (2.0f * (px + 0.5f) / width - 1.0f) * scale_u;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * scale_v;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    float ray_len = rsqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    ray_x *= ray_len;
    ray_y *= ray_len;
    ray_z *= ray_len;

    int wx = world_x;
    int wy = world_y;
    int wz = world_z;
    long long out_idx = ((long long)b * height * width + py * width + px) * 3;
    int wxz = wx * wz;  // Y stride for [Y, X, Z] layout
    long long wxyz = (long long)wx * wy * wz;  // Total voxels per batch

    int voxel_x = (int)floorf(eye_x);
    int voxel_y = (int)floorf(eye_y);
    int voxel_z = (int)floorf(eye_z);

    int step_x = (ray_x >= 0) ? 1 : -1;
    int step_y = (ray_y >= 0) ? 1 : -1;
    int step_z = (ray_z >= 0) ? 1 : -1;

    float inv_ray_x = (ray_x != 0.0f) ? 1.0f / ray_x : 1e30f;
    float inv_ray_y = (ray_y != 0.0f) ? 1.0f / ray_y : 1e30f;
    float inv_ray_z = (ray_z != 0.0f) ? 1.0f / ray_z : 1e30f;

    float t_delta_x = fabsf(inv_ray_x);
    float t_delta_y = fabsf(inv_ray_y);
    float t_delta_z = fabsf(inv_ray_z);

    float t_max_x = (ray_x >= 0) ? ((float)(voxel_x + 1) - eye_x) * inv_ray_x
                                 : ((float)voxel_x - eye_x) * inv_ray_x;
    float t_max_y = (ray_y >= 0) ? ((float)(voxel_y + 1) - eye_y) * inv_ray_y
                                 : ((float)voxel_y - eye_y) * inv_ray_y;
    float t_max_z = (ray_z >= 0) ? ((float)(voxel_z + 1) - eye_z) * inv_ray_z
                                 : ((float)voxel_z - eye_z) * inv_ray_z;

    int8_t hit_block = AIR;
    int step = 0;

    #define DDA_STEP_PRE() \
        if (step >= max_steps) goto done_pre; \
        if ((unsigned)voxel_x < (unsigned)wx && \
            (unsigned)voxel_y < (unsigned)wy && \
            (unsigned)voxel_z < (unsigned)wz) { \
            int8_t block = __ldg(&voxels[b * wxyz + voxel_y * wxz + voxel_x * wz + voxel_z]); \
            if (block >= 0) { hit_block = block; goto done_pre; } \
        } \
        if (t_max_x < t_max_y && t_max_x < t_max_z) { \
            t_max_x += t_delta_x; voxel_x += step_x; \
        } else if (t_max_y < t_max_z) { \
            t_max_y += t_delta_y; voxel_y += step_y; \
        } else { \
            t_max_z += t_delta_z; voxel_z += step_z; \
        } \
        ++step;

    #pragma unroll 4
    for (int i = 0; i < max_steps; i += 4) {
        DDA_STEP_PRE();
        DDA_STEP_PRE();
        DDA_STEP_PRE();
        DDA_STEP_PRE();
    }
    #undef DDA_STEP_PRE

done_pre:
    if (hit_block >= 0) {
        output[out_idx + 0] = BLOCK_COLORS_U8[hit_block][0];
        output[out_idx + 1] = BLOCK_COLORS_U8[hit_block][1];
        output[out_idx + 2] = BLOCK_COLORS_U8[hit_block][2];
    } else {
        output[out_idx + 0] = SKY_COLOR_U8[0];
        output[out_idx + 1] = SKY_COLOR_U8[1];
        output[out_idx + 2] = SKY_COLOR_U8[2];
    }
}


// =============================================================================
// RENDER KERNEL (UINT8, Coalesced Layout) - [Y,X,Z,B] voxel layout for coalescing
// Thread mapping: adjacent threads handle same pixel across different batches
// When rays are coherent, voxel reads are coalesced
// =============================================================================

__global__ void render_kernel_uint8_coalesced(
    const int8_t* __restrict__ voxels,  // [Y, X, Z, B] layout!
    const float* __restrict__ basis,     // [B, 14] precomputed camera basis
    uint8_t* __restrict__ output,        // [B, H, W, 3]
    int batch_size,
    int world_size,
    int width,
    int height,
    int max_steps
) {
    // Thread mapping: warp handles same pixel across batches for coalescing
    // blockIdx.x = pixel index, blockIdx.y * blockDim.x + threadIdx.x = batch offset
    int pixel_idx = blockIdx.x;
    int b = blockIdx.y * blockDim.x + threadIdx.x;

    if (pixel_idx >= height * width || b >= batch_size) return;

    int py = pixel_idx / width;
    int px = pixel_idx % width;

    // Load basis for this batch element (each thread loads its own - no sharing)
    int bidx = b * 14;
    float eye_x = __ldg(&basis[bidx + 0]);
    float eye_y = __ldg(&basis[bidx + 1]);
    float eye_z = __ldg(&basis[bidx + 2]);
    float forward_x = __ldg(&basis[bidx + 3]);
    float forward_y = __ldg(&basis[bidx + 4]);
    float forward_z = __ldg(&basis[bidx + 5]);
    float right_x = __ldg(&basis[bidx + 6]);
    float right_z = __ldg(&basis[bidx + 7]);
    float up_x = __ldg(&basis[bidx + 8]);
    float up_y = __ldg(&basis[bidx + 9]);
    float up_z = __ldg(&basis[bidx + 10]);
    float scale_u = __ldg(&basis[bidx + 11]);
    float scale_v = __ldg(&basis[bidx + 12]);

    // Ray direction from pixel coordinates
    float u = (2.0f * (px + 0.5f) / width - 1.0f) * scale_u;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * scale_v;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    float ray_len = rsqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    ray_x *= ray_len;
    ray_y *= ray_len;
    ray_z *= ray_len;

    int ws = world_size;
    long long out_idx = ((long long)b * height * width + py * width + px) * 3;

    int voxel_x = (int)floorf(eye_x);
    int voxel_y = (int)floorf(eye_y);
    int voxel_z = (int)floorf(eye_z);

    int step_x = (ray_x >= 0) ? 1 : -1;
    int step_y = (ray_y >= 0) ? 1 : -1;
    int step_z = (ray_z >= 0) ? 1 : -1;

    float inv_ray_x = (ray_x != 0.0f) ? 1.0f / ray_x : 1e30f;
    float inv_ray_y = (ray_y != 0.0f) ? 1.0f / ray_y : 1e30f;
    float inv_ray_z = (ray_z != 0.0f) ? 1.0f / ray_z : 1e30f;

    float t_delta_x = fabsf(inv_ray_x);
    float t_delta_y = fabsf(inv_ray_y);
    float t_delta_z = fabsf(inv_ray_z);

    float t_max_x = (ray_x >= 0) ? ((float)(voxel_x + 1) - eye_x) * inv_ray_x
                                 : ((float)voxel_x - eye_x) * inv_ray_x;
    float t_max_y = (ray_y >= 0) ? ((float)(voxel_y + 1) - eye_y) * inv_ray_y
                                 : ((float)voxel_y - eye_y) * inv_ray_y;
    float t_max_z = (ray_z >= 0) ? ((float)(voxel_z + 1) - eye_z) * inv_ray_z
                                 : ((float)voxel_z - eye_z) * inv_ray_z;

    int8_t hit_block = AIR;
    int step = 0;

    // DDA loop with [Y,X,Z,B] layout - coalesced when rays hit same position
    // Address: voxels[voxel_y * ws * ws * B + voxel_x * ws * B + voxel_z * B + b]
    #define DDA_STEP_COAL() \
        if (step >= max_steps) goto done_coal; \
        if ((unsigned)voxel_x < (unsigned)ws && \
            (unsigned)voxel_y < (unsigned)ws && \
            (unsigned)voxel_z < (unsigned)ws) { \
            int idx = voxel_y * ws * ws * batch_size + voxel_x * ws * batch_size + voxel_z * batch_size + b; \
            int8_t block = __ldg(&voxels[idx]); \
            if (block >= 0) { hit_block = block; goto done_coal; } \
        } \
        if (t_max_x < t_max_y && t_max_x < t_max_z) { \
            t_max_x += t_delta_x; voxel_x += step_x; \
        } else if (t_max_y < t_max_z) { \
            t_max_y += t_delta_y; voxel_y += step_y; \
        } else { \
            t_max_z += t_delta_z; voxel_z += step_z; \
        } \
        ++step;

    #pragma unroll 4
    for (int i = 0; i < max_steps; i += 4) {
        DDA_STEP_COAL();
        DDA_STEP_COAL();
        DDA_STEP_COAL();
        DDA_STEP_COAL();
    }
    #undef DDA_STEP_COAL

done_coal:
    if (hit_block >= 0) {
        output[out_idx + 0] = BLOCK_COLORS_U8[hit_block][0];
        output[out_idx + 1] = BLOCK_COLORS_U8[hit_block][1];
        output[out_idx + 2] = BLOCK_COLORS_U8[hit_block][2];
    } else {
        output[out_idx + 0] = SKY_COLOR_U8[0];
        output[out_idx + 1] = SKY_COLOR_U8[1];
        output[out_idx + 2] = SKY_COLOR_U8[2];
    }
}


// =============================================================================
// RENDER KERNEL (UINT8, Shared Memory Voxels) - Entire voxel grid in shared mem
// Uses cp.async for async global->shared copy, all DDA lookups hit shared memory
// Requires: world_size <= 16 (4KB shared mem for voxels)
// =============================================================================

__global__ void render_kernel_uint8_smem(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ basis,  // [B, 14] precomputed camera basis
    uint8_t* __restrict__ output,
    int batch_size,
    int world_size,
    int width,
    int height,
    int max_steps
) {
    // Shared memory: 14 floats for basis + 4096 bytes for voxels (16^3)
    extern __shared__ int8_t shared_mem[];
    float* s_basis = (float*)shared_mem;
    int8_t* s_voxels = shared_mem + 64;  // Align to 64 bytes (14 floats = 56 bytes, pad to 64)

    // 2D grid: blockIdx.y = batch element, blockIdx.x = pixel block within batch
    int b = blockIdx.y;
    int pixel_offset = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels_per_batch = height * width;

    if (b >= batch_size) return;

    int ws = world_size;
    int ws3 = ws * ws * ws;

    // Load basis into shared memory (first 14 threads)
    if (threadIdx.x < 14) {
        s_basis[threadIdx.x] = basis[b * 14 + threadIdx.x];
    }

    // Load entire voxel grid into shared memory
    // With 128 threads and 4096 bytes, each thread loads 32 bytes (8 int32s)
    // Use coalesced int4 loads for maximum bandwidth
    const int4* src = (const int4*)(voxels + (long long)b * ws3);
    int4* dst = (int4*)s_voxels;
    const int int4_per_thread = (ws3 / 16 + blockDim.x - 1) / blockDim.x;  // 16 bytes per int4
    const int total_int4 = ws3 / 16;  // 4096/16 = 256 int4s

    #pragma unroll
    for (int i = 0; i < int4_per_thread; i++) {
        int idx = threadIdx.x + i * blockDim.x;
        if (idx < total_int4) {
            dst[idx] = src[idx];
        }
    }
    __syncthreads();

    // Early exit for threads beyond pixel count
    if (pixel_offset >= pixels_per_batch) return;

    int py = pixel_offset / width;
    int px = pixel_offset % width;

    // Read basis from shared memory
    float eye_x = s_basis[0];
    float eye_y = s_basis[1];
    float eye_z = s_basis[2];
    float forward_x = s_basis[3];
    float forward_y = s_basis[4];
    float forward_z = s_basis[5];
    float right_x = s_basis[6];
    float right_z = s_basis[7];
    float up_x = s_basis[8];
    float up_y = s_basis[9];
    float up_z = s_basis[10];
    float scale_u = s_basis[11];
    float scale_v = s_basis[12];

    // Ray direction from pixel coordinates
    float u = (2.0f * (px + 0.5f) / width - 1.0f) * scale_u;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * scale_v;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    float ray_len = rsqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    ray_x *= ray_len;
    ray_y *= ray_len;
    ray_z *= ray_len;

    long long out_idx = ((long long)b * height * width + py * width + px) * 3;
    int ws2 = ws * ws;

    int voxel_x = (int)floorf(eye_x);
    int voxel_y = (int)floorf(eye_y);
    int voxel_z = (int)floorf(eye_z);

    int step_x = (ray_x >= 0) ? 1 : -1;
    int step_y = (ray_y >= 0) ? 1 : -1;
    int step_z = (ray_z >= 0) ? 1 : -1;

    float inv_ray_x = (ray_x != 0.0f) ? 1.0f / ray_x : 1e30f;
    float inv_ray_y = (ray_y != 0.0f) ? 1.0f / ray_y : 1e30f;
    float inv_ray_z = (ray_z != 0.0f) ? 1.0f / ray_z : 1e30f;

    float t_delta_x = fabsf(inv_ray_x);
    float t_delta_y = fabsf(inv_ray_y);
    float t_delta_z = fabsf(inv_ray_z);

    float t_max_x = (ray_x >= 0) ? ((float)(voxel_x + 1) - eye_x) * inv_ray_x
                                 : ((float)voxel_x - eye_x) * inv_ray_x;
    float t_max_y = (ray_y >= 0) ? ((float)(voxel_y + 1) - eye_y) * inv_ray_y
                                 : ((float)voxel_y - eye_y) * inv_ray_y;
    float t_max_z = (ray_z >= 0) ? ((float)(voxel_z + 1) - eye_z) * inv_ray_z
                                 : ((float)voxel_z - eye_z) * inv_ray_z;

    int8_t hit_block = AIR;
    int step = 0;

    // DDA loop with shared memory voxel lookups (no global memory access!)
    #define DDA_STEP_SMEM() \
        if (step >= max_steps) goto done_smem; \
        if ((unsigned)voxel_x < (unsigned)ws && \
            (unsigned)voxel_y < (unsigned)ws && \
            (unsigned)voxel_z < (unsigned)ws) { \
            int8_t block = s_voxels[voxel_y * ws2 + voxel_x * ws + voxel_z]; \
            if (block >= 0) { hit_block = block; goto done_smem; } \
        } \
        if (t_max_x < t_max_y && t_max_x < t_max_z) { \
            t_max_x += t_delta_x; voxel_x += step_x; \
        } else if (t_max_y < t_max_z) { \
            t_max_y += t_delta_y; voxel_y += step_y; \
        } else { \
            t_max_z += t_delta_z; voxel_z += step_z; \
        } \
        ++step;

    #pragma unroll 4
    for (int i = 0; i < max_steps; i += 4) {
        DDA_STEP_SMEM();
        DDA_STEP_SMEM();
        DDA_STEP_SMEM();
        DDA_STEP_SMEM();
    }
    #undef DDA_STEP_SMEM

done_smem:
    if (hit_block >= 0) {
        output[out_idx + 0] = BLOCK_COLORS_U8[hit_block][0];
        output[out_idx + 1] = BLOCK_COLORS_U8[hit_block][1];
        output[out_idx + 2] = BLOCK_COLORS_U8[hit_block][2];
    } else {
        output[out_idx + 0] = SKY_COLOR_U8[0];
        output[out_idx + 1] = SKY_COLOR_U8[1];
        output[out_idx + 2] = SKY_COLOR_U8[2];
    }
}


__global__ void render_kernel_uint8_minimal(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ cameras,
    uint8_t* __restrict__ output,
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

    // Trig - unavoidable for camera rotation
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

    float fov_rad = fov_degrees * 0.01745329f;  // pi/180 as constant
    float aspect = (float)width / (float)height;
    float half_fov = tanf(fov_rad * 0.5f);

    float u = (2.0f * (px + 0.5f) / width - 1.0f) * half_fov * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * half_fov;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    // Fast inverse sqrt approximation could go here, but rsqrtf is already fast
    float ray_len = rsqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    ray_x *= ray_len;
    ray_y *= ray_len;
    ray_z *= ray_len;

    int ws = world_size;
    long long out_idx = ((long long)b * height * width + py * width + px) * 3;

    // Simplified bounds check - skip AABB for small worlds
    int ws2 = ws * ws;
    int ws3 = ws * ws * ws;

    int voxel_x = (int)floorf(eye_x);
    int voxel_y = (int)floorf(eye_y);
    int voxel_z = (int)floorf(eye_z);

    int step_x = (ray_x >= 0) ? 1 : -1;
    int step_y = (ray_y >= 0) ? 1 : -1;
    int step_z = (ray_z >= 0) ? 1 : -1;

    float inv_ray_x = (ray_x != 0.0f) ? 1.0f / ray_x : 1e30f;
    float inv_ray_y = (ray_y != 0.0f) ? 1.0f / ray_y : 1e30f;
    float inv_ray_z = (ray_z != 0.0f) ? 1.0f / ray_z : 1e30f;

    float t_delta_x = fabsf(inv_ray_x);
    float t_delta_y = fabsf(inv_ray_y);
    float t_delta_z = fabsf(inv_ray_z);

    float t_max_x = (ray_x >= 0) ? ((float)(voxel_x + 1) - eye_x) * inv_ray_x
                                 : ((float)voxel_x - eye_x) * inv_ray_x;
    float t_max_y = (ray_y >= 0) ? ((float)(voxel_y + 1) - eye_y) * inv_ray_y
                                 : ((float)voxel_y - eye_y) * inv_ray_y;
    float t_max_z = (ray_z >= 0) ? ((float)(voxel_z + 1) - eye_z) * inv_ray_z
                                 : ((float)voxel_z - eye_z) * inv_ray_z;

    int8_t hit_block = AIR;
    int step = 0;

    // Minimal DDA - no fog, no shading computation during traversal
    #define DDA_STEP_MIN() \
        if (step >= max_steps) goto done_min; \
        if ((unsigned)voxel_x < (unsigned)ws && \
            (unsigned)voxel_y < (unsigned)ws && \
            (unsigned)voxel_z < (unsigned)ws) { \
            int8_t block = __ldg(&voxels[(long long)b * ws3 + voxel_y * ws2 + voxel_x * ws + voxel_z]); \
            if (block >= 0) { hit_block = block; goto done_min; } \
        } \
        if (t_max_x < t_max_y && t_max_x < t_max_z) { \
            t_max_x += t_delta_x; voxel_x += step_x; \
        } else if (t_max_y < t_max_z) { \
            t_max_y += t_delta_y; voxel_y += step_y; \
        } else { \
            t_max_z += t_delta_z; voxel_z += step_z; \
        } \
        ++step;

    #pragma unroll 4
    for (int i = 0; i < max_steps; i += 4) {
        DDA_STEP_MIN();
        DDA_STEP_MIN();
        DDA_STEP_MIN();
        DDA_STEP_MIN();
    }
    #undef DDA_STEP_MIN

done_min:
    // Direct uint8 color output - no float math
    if (hit_block >= 0) {
        output[out_idx + 0] = BLOCK_COLORS_U8[hit_block][0];
        output[out_idx + 1] = BLOCK_COLORS_U8[hit_block][1];
        output[out_idx + 2] = BLOCK_COLORS_U8[hit_block][2];
    } else {
        output[out_idx + 0] = SKY_COLOR_U8[0];
        output[out_idx + 1] = SKY_COLOR_U8[1];
        output[out_idx + 2] = SKY_COLOR_U8[2];
    }
}


// =============================================================================
// RENDER KERNEL (UINT8, FP16) - Half precision DDA for maximum throughput
// Uses fp16 for t_max/t_delta accumulation, fp32 for camera/ray setup
// =============================================================================

__global__ void render_kernel_uint8_fp16(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ cameras,
    uint8_t* __restrict__ output,
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

    // Camera setup stays fp32 for precision
    float cam_x = __ldg(&cameras[b * 5 + 0]);
    float cam_y = __ldg(&cameras[b * 5 + 1]);
    float cam_z = __ldg(&cameras[b * 5 + 2]);
    float yaw = __ldg(&cameras[b * 5 + 3]);
    float pitch = __ldg(&cameras[b * 5 + 4]);

    float eye_x = cam_x;
    float eye_y = cam_y + 1.6f;
    float eye_z = cam_z;

    // Trig in fp32
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

    float fov_rad = fov_degrees * 0.01745329f;
    float aspect = (float)width / (float)height;
    float half_fov = tanf(fov_rad * 0.5f);

    float u = (2.0f * (px + 0.5f) / width - 1.0f) * half_fov * aspect;
    float v = (1.0f - 2.0f * (py + 0.5f) / height) * half_fov;

    float ray_x = forward_x + u * right_x + v * up_x;
    float ray_y = forward_y + v * up_y;
    float ray_z = forward_z + u * right_z + v * up_z;

    float ray_len = rsqrtf(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z);
    ray_x *= ray_len;
    ray_y *= ray_len;
    ray_z *= ray_len;

    int ws = world_size;
    long long out_idx = ((long long)b * height * width + py * width + px) * 3;
    int ws2 = ws * ws;
    int ws3 = ws * ws * ws;

    int voxel_x = (int)floorf(eye_x);
    int voxel_y = (int)floorf(eye_y);
    int voxel_z = (int)floorf(eye_z);

    int step_x = (ray_x >= 0) ? 1 : -1;
    int step_y = (ray_y >= 0) ? 1 : -1;
    int step_z = (ray_z >= 0) ? 1 : -1;

    // Compute initial values in fp32, then convert to fp16 for DDA loop
    float inv_ray_x_f = (ray_x != 0.0f) ? 1.0f / ray_x : 65504.0f;  // fp16 max
    float inv_ray_y_f = (ray_y != 0.0f) ? 1.0f / ray_y : 65504.0f;
    float inv_ray_z_f = (ray_z != 0.0f) ? 1.0f / ray_z : 65504.0f;

    // Convert to half precision for DDA
    __half t_delta_x = __float2half(fabsf(inv_ray_x_f));
    __half t_delta_y = __float2half(fabsf(inv_ray_y_f));
    __half t_delta_z = __float2half(fabsf(inv_ray_z_f));

    __half t_max_x = __float2half((ray_x >= 0) ? ((float)(voxel_x + 1) - eye_x) * inv_ray_x_f
                                               : ((float)voxel_x - eye_x) * inv_ray_x_f);
    __half t_max_y = __float2half((ray_y >= 0) ? ((float)(voxel_y + 1) - eye_y) * inv_ray_y_f
                                               : ((float)voxel_y - eye_y) * inv_ray_y_f);
    __half t_max_z = __float2half((ray_z >= 0) ? ((float)(voxel_z + 1) - eye_z) * inv_ray_z_f
                                               : ((float)voxel_z - eye_z) * inv_ray_z_f);

    int8_t hit_block = AIR;
    int step = 0;

    // FP16 DDA loop
    #define DDA_STEP_FP16() \
        if (step >= max_steps) goto done_fp16; \
        if ((unsigned)voxel_x < (unsigned)ws && \
            (unsigned)voxel_y < (unsigned)ws && \
            (unsigned)voxel_z < (unsigned)ws) { \
            int8_t block = __ldg(&voxels[(long long)b * ws3 + voxel_y * ws2 + voxel_x * ws + voxel_z]); \
            if (block >= 0) { hit_block = block; goto done_fp16; } \
        } \
        if (__hlt(t_max_x, t_max_y) && __hlt(t_max_x, t_max_z)) { \
            t_max_x = __hadd(t_max_x, t_delta_x); voxel_x += step_x; \
        } else if (__hlt(t_max_y, t_max_z)) { \
            t_max_y = __hadd(t_max_y, t_delta_y); voxel_y += step_y; \
        } else { \
            t_max_z = __hadd(t_max_z, t_delta_z); voxel_z += step_z; \
        } \
        ++step;

    #pragma unroll 4
    for (int i = 0; i < max_steps; i += 4) {
        DDA_STEP_FP16();
        DDA_STEP_FP16();
        DDA_STEP_FP16();
        DDA_STEP_FP16();
    }
    #undef DDA_STEP_FP16

done_fp16:
    if (hit_block >= 0) {
        output[out_idx + 0] = BLOCK_COLORS_U8[hit_block][0];
        output[out_idx + 1] = BLOCK_COLORS_U8[hit_block][1];
        output[out_idx + 2] = BLOCK_COLORS_U8[hit_block][2];
    } else {
        output[out_idx + 0] = SKY_COLOR_U8[0];
        output[out_idx + 1] = SKY_COLOR_U8[1];
        output[out_idx + 2] = SKY_COLOR_U8[2];
    }
}


// =============================================================================
// RENDER KERNEL (UINT8, Ground-Accelerated) - Optimized for flat worlds
// Adds early ground termination within the existing efficient DDA loop
// =============================================================================

__global__ void render_kernel_uint8_fast(
    const int8_t* __restrict__ voxels,
    const float* __restrict__ cameras,
    uint8_t* __restrict__ output,
    int batch_size,
    int world_size,
    int width,
    int height,
    int max_steps,
    float view_distance,
    float fov_degrees,
    int ground_height
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
    float t_entry, t_exit;
    bool hits_world = ray_intersects_aabb(
        eye_x, eye_y, eye_z,
        ray_x, ray_y, ray_z,
        0.0f, 0.0f, 0.0f,
        (float)ws, (float)ws, (float)ws,
        &t_entry, &t_exit
    );

    long long out_idx = ((long long)b * height * width + py * width + px) * 3;

    if (!hits_world || t_entry > view_distance) {
        output[out_idx + 0] = (uint8_t)(SKY_COLOR[0] * 255.0f);
        output[out_idx + 1] = (uint8_t)(SKY_COLOR[1] * 255.0f);
        output[out_idx + 2] = (uint8_t)(SKY_COLOR[2] * 255.0f);
        return;
    }

    int ws2 = ws * ws;
    int ws3 = ws * ws * ws;
    int gh = ground_height;

    int voxel_x = (int)floorf(eye_x);
    int voxel_y = (int)floorf(eye_y);
    int voxel_z = (int)floorf(eye_z);

    int step_x = (ray_x >= 0) ? 1 : -1;
    int step_y = (ray_y >= 0) ? 1 : -1;
    int step_z = (ray_z >= 0) ? 1 : -1;

    float inv_ray_x = (ray_x != 0.0f) ? 1.0f / ray_x : 1e30f;
    float inv_ray_y = (ray_y != 0.0f) ? 1.0f / ray_y : 1e30f;
    float inv_ray_z = (ray_z != 0.0f) ? 1.0f / ray_z : 1e30f;

    float t_delta_x = fabsf(inv_ray_x);
    float t_delta_y = fabsf(inv_ray_y);
    float t_delta_z = fabsf(inv_ray_z);

    float t_max_x = (ray_x >= 0) ? ((float)(voxel_x + 1) - eye_x) * inv_ray_x
                                 : ((float)voxel_x - eye_x) * inv_ray_x;
    float t_max_y = (ray_y >= 0) ? ((float)(voxel_y + 1) - eye_y) * inv_ray_y
                                 : ((float)voxel_y - eye_y) * inv_ray_y;
    float t_max_z = (ray_z >= 0) ? ((float)(voxel_z + 1) - eye_z) * inv_ray_z
                                 : ((float)voxel_z - eye_z) * inv_ray_z;

    float t = 0.0f;
    int hit_face = -1;
    int8_t hit_block = AIR;
    int step = 0;

    // DDA with ground early-termination using same macro pattern as original
    #define DDA_STEP_FAST() \
        if (step >= max_steps || t >= view_distance) goto render_done_fast; \
        if (voxel_y < gh) { \
            hit_block = (voxel_y == gh - 1) ? GRASS : DIRT; \
            goto render_done_fast; \
        } \
        if (voxel_x >= 0 && voxel_x < ws && \
            voxel_y >= 0 && voxel_y < ws && \
            voxel_z >= 0 && voxel_z < ws) { \
            long long voxel_idx = (long long)b * ws3 + voxel_y * ws2 + voxel_x * ws + voxel_z; \
            int8_t block = __ldg(&voxels[voxel_idx]); \
            if (block >= 0) { \
                hit_block = block; \
                goto render_done_fast; \
            } \
        } else if (t > t_exit) { \
            goto render_done_fast; \
        } \
        if (t_max_x < t_max_y && t_max_x < t_max_z) { \
            t = t_max_x; t_max_x += t_delta_x; voxel_x += step_x; hit_face = 0; \
        } else if (t_max_y < t_max_z) { \
            t = t_max_y; t_max_y += t_delta_y; voxel_y += step_y; hit_face = 1; \
        } else { \
            t = t_max_z; t_max_z += t_delta_z; voxel_z += step_z; hit_face = 2; \
        } \
        ++step;

    #pragma unroll 4
    for (int i = 0; i < max_steps; i += 4) {
        DDA_STEP_FAST();
        DDA_STEP_FAST();
        DDA_STEP_FAST();
        DDA_STEP_FAST();
    }
    #undef DDA_STEP_FAST

render_done_fast:
    if (hit_block >= 0) {
        float r = BLOCK_COLORS[hit_block][0];
        float g = BLOCK_COLORS[hit_block][1];
        float b_col = BLOCK_COLORS[hit_block][2];

        float shade = 1.0f;
        if (hit_face == 0) shade = 0.8f;
        else if (hit_face == 2) shade = 0.9f;
        else if (hit_face == 1 && step_y < 0) shade = 0.6f;

        float fog = fminf(t / view_distance, 1.0f);
        fog = fog * fog;

        output[out_idx + 0] = (uint8_t)((r * shade * (1.0f - fog) + SKY_COLOR[0] * fog) * 255.0f);
        output[out_idx + 1] = (uint8_t)((g * shade * (1.0f - fog) + SKY_COLOR[1] * fog) * 255.0f);
        output[out_idx + 2] = (uint8_t)((b_col * shade * (1.0f - fog) + SKY_COLOR[2] * fog) * 255.0f);
    } else {
        output[out_idx + 0] = (uint8_t)(SKY_COLOR[0] * 255.0f);
        output[out_idx + 1] = (uint8_t)(SKY_COLOR[1] * 255.0f);
        output[out_idx + 2] = (uint8_t)(SKY_COLOR[2] * 255.0f);
    }
}


// =============================================================================
// DECODE ACTIONS KERNEL - Convert multi-hot buttons + continuous look to control signals
// =============================================================================

// Button indices for multi-hot encoding
constexpr int BTN_FORWARD = 0;
constexpr int BTN_BACKWARD = 1;
constexpr int BTN_STRAFE_LEFT = 2;
constexpr int BTN_STRAFE_RIGHT = 3;
constexpr int BTN_JUMP = 4;
constexpr int BTN_BREAK = 5;
constexpr int BTN_PLACE = 6;
constexpr int BTN_SPRINT = 7;

__global__ void decode_actions_kernel(
    const int8_t* __restrict__ buttons,  // [B, 8] multi-hot
    const float* __restrict__ look,      // [B, 2] radians (yaw, pitch)
    float* __restrict__ forward_in,      // [B] output
    float* __restrict__ strafe_in,       // [B] output
    float* __restrict__ delta_yaw_in,    // [B] output
    float* __restrict__ delta_pitch_in,  // [B] output
    bool* __restrict__ jump_in,          // [B] output
    bool* __restrict__ do_break,         // [B] output
    bool* __restrict__ do_place,         // [B] output
    float* __restrict__ speed_mult,      // [B] output
    int batch_size,
    float sprint_multiplier
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    // Read buttons
    int8_t fwd = buttons[i * 8 + BTN_FORWARD];
    int8_t back = buttons[i * 8 + BTN_BACKWARD];
    int8_t left = buttons[i * 8 + BTN_STRAFE_LEFT];
    int8_t right = buttons[i * 8 + BTN_STRAFE_RIGHT];
    int8_t jump = buttons[i * 8 + BTN_JUMP];
    int8_t brk = buttons[i * 8 + BTN_BREAK];
    int8_t place = buttons[i * 8 + BTN_PLACE];
    int8_t sprint = buttons[i * 8 + BTN_SPRINT];

    // Convert to control signals
    // Forward/backward: fwd=+1, back=-1, both=0
    forward_in[i] = (float)fwd - (float)back;

    // Strafe: left=-1, right=+1, both=0
    strafe_in[i] = (float)right - (float)left;

    // Look from continuous input
    delta_yaw_in[i] = look[i * 2 + 0];
    delta_pitch_in[i] = look[i * 2 + 1];

    // Boolean actions
    jump_in[i] = (jump != 0);
    do_break[i] = (brk != 0);
    do_place[i] = (place != 0);

    // Sprint only applies when moving forward
    speed_mult[i] = (fwd && sprint) ? sprint_multiplier : 1.0f;
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
    const float* __restrict__ speed_mult,     // [B] - NEW: sprint multiplier
    int batch_size,
    int world_x,
    int world_y,
    int world_z,
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
    float speed = walk_speed * speed_mult[i];  // Apply sprint multiplier

    float move_x = sin_yaw * fwd * speed * dt + cos_yaw * str * walk_speed * dt;
    float move_z = cos_yaw * fwd * speed * dt - sin_yaw * str * walk_speed * dt;

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

    // Simple ground collision (flat world at y = world_y/4)
    float ground_y = (float)(world_y / 4);
    if (new_y <= ground_y) {
        new_y = ground_y;
        vel_y = 0.0f;
        on_ground[i] = true;
    } else {
        on_ground[i] = false;
    }

    // World boundary clamping
    float margin = 1.0f;
    new_x = fminf(fmaxf(new_x, margin), (float)world_x - margin);
    new_z = fminf(fmaxf(new_z, margin), (float)world_z - margin);

    positions[i * 3 + 0] = new_x;
    positions[i * 3 + 1] = new_y;
    positions[i * 3 + 2] = new_z;
    velocities[i * 3 + 1] = vel_y;
}


// =============================================================================
// RAYCAST INTERACT KERNEL - Block breaking and placing (combined)
// =============================================================================

__global__ void raycast_interact_kernel(
    int8_t* __restrict__ voxels,         // [B, Y, X, Z]
    const float* __restrict__ positions, // [B, 3]
    const float* __restrict__ yaws,      // [B]
    const float* __restrict__ pitches,   // [B]
    const bool* __restrict__ do_break,   // [B]
    const bool* __restrict__ do_place,   // [B]
    float* __restrict__ rewards,         // [B]
    int batch_size,
    int world_x,
    int world_y,
    int world_z,
    int8_t target_block,    // Block type that gives reward (-1 for any)
    float reward_value      // Reward for breaking target block
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    rewards[i] = 0.0f;

    bool wants_break = do_break[i];
    bool wants_place = do_place[i];

    // Early exit if no interaction requested
    if (!wants_break && !wants_place) return;

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
    int wx = world_x;
    int wy = world_y;
    int wz = world_z;
    int wxz = wx * wz;
    long long wxyz = (long long)wx * wy * wz;

    // Track previous air voxel for block placement
    int prev_vx = -1, prev_vy = -1, prev_vz = -1;

    // Raycast to find hit block
    for (int step = 0; step < 32; ++step) {
        float t = 0.1f + step * (reach / 32.0f);
        float px = eye_x + dir_x * t;
        float py = eye_y + dir_y * t;
        float pz = eye_z + dir_z * t;

        int vx = (int)floorf(px);
        int vy = (int)floorf(py);
        int vz = (int)floorf(pz);

        if (vx >= 0 && vx < wx && vy >= 0 && vy < wy && vz >= 0 && vz < wz) {
            long long idx = i * wxyz + vy * wxz + vx * wz + vz;
            int8_t block = voxels[idx];

            if (block >= 0) {
                // Hit a solid block
                if (wants_break) {
                    // Break the block
                    if (target_block < 0 || block == target_block) {
                        rewards[i] = reward_value;
                    }
                    voxels[idx] = AIR;
                } else if (wants_place && prev_vx >= 0) {
                    // Place block at previous air voxel (the face we're looking at)
                    long long place_idx = i * wxyz + prev_vy * wxz + prev_vx * wz + prev_vz;
                    voxels[place_idx] = DIRT;
                }
                return;
            }

            // Remember this air voxel for potential placement
            prev_vx = vx;
            prev_vy = vy;
            prev_vz = vz;
        }
    }
}


// =============================================================================
// RAYCAST BREAK KERNEL - Legacy, kept for backward compatibility
// =============================================================================

__global__ void raycast_break_kernel(
    int8_t* __restrict__ voxels,         // [B, Y, X, Z]
    const float* __restrict__ positions, // [B, 3]
    const float* __restrict__ yaws,      // [B]
    const float* __restrict__ pitches,   // [B]
    const bool* __restrict__ do_break,   // [B]
    float* __restrict__ rewards,         // [B]
    int batch_size,
    int world_x,
    int world_y,
    int world_z,
    int8_t target_block,
    float reward_value
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
    int wx = world_x;
    int wy = world_y;
    int wz = world_z;
    int wxz = wx * wz;
    long long wxyz = (long long)wx * wy * wz;

    for (int step = 0; step < 32; ++step) {
        float t = 0.1f + step * (reach / 32.0f);
        float px = eye_x + dir_x * t;
        float py = eye_y + dir_y * t;
        float pz = eye_z + dir_z * t;

        int vx = (int)floorf(px);
        int vy = (int)floorf(py);
        int vz = (int)floorf(pz);

        if (vx >= 0 && vx < wx && vy >= 0 && vy < wy && vz >= 0 && vz < wz) {
            long long idx = (long long)i * wxyz + vy * wxz + vx * wz + vz;
            int8_t block = voxels[idx];

            if (block >= 0) {
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
    int world_x,
    int world_y,
    int world_z,
    int ground_height,
    unsigned int seed
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int wx = world_x;
    int wy = world_y;
    int wz = world_z;
    int wxz = wx * wz;
    long long wxyz = (long long)wx * wy * wz;
    long long total = (long long)batch_size * wxyz;
    if (idx >= total) return;

    int b = idx / wxyz;
    int voxel_idx = idx % wxyz;
    int y = voxel_idx / wxz;
    int x = (voxel_idx % wxz) / wz;
    int z = voxel_idx % wz;

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
    int world_x,
    int world_y,
    int world_z,
    int tree_x,
    int tree_z,
    int ground_height
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int wx = world_x;
    int wy = world_y;
    int wz = world_z;
    int wxz = wx * wz;
    long long wxyz = (long long)wx * wy * wz;

    // Place 3-block tall log
    for (int dy = 0; dy < 3; ++dy) {
        int y = ground_height + dy;
        if (y < wy) {
            long long idx = (long long)b * wxyz + y * wxz + tree_x * wz + tree_z;
            voxels[idx] = OAKLOG;
        }
    }

    // Place leaves on top
    int leaf_y = ground_height + 3;
    if (leaf_y < wy) {
        long long idx = (long long)b * wxyz + leaf_y * wxz + tree_x * wz + tree_z;
        voxels[idx] = LEAVES;
    }
}

// Place a simple house structure
__global__ void place_house_kernel(
    int8_t* __restrict__ voxels,  // [B, Y, X, Z]
    int batch_size,
    int world_x,
    int world_y,
    int world_z,
    int house_x,  // corner X position
    int house_z,  // corner Z position
    int ground_height
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    int wx = world_x;
    int wy = world_y;
    int wz = world_z;
    int wxz = wx * wz;
    long long wxyz = (long long)wx * wy * wz;

    // House dimensions: 5 wide (X) x 4 deep (Z) x 4 tall (Y)
    const int W = 5, D = 4, H = 4;
    int base_y = ground_height;

    // Helper to set voxel
    #define SET_VOXEL(dy, dx, dz, block) do { \
        int vx = house_x + (dx); \
        int vy = base_y + (dy); \
        int vz = house_z + (dz); \
        if (vx >= 0 && vx < wx && vy >= 0 && vy < wy && vz >= 0 && vz < wz) { \
            long long idx = (long long)b * wxyz + vy * wxz + vx * wz + vz; \
            voxels[idx] = (block); \
        } \
    } while(0)

    // Floor (cobblestone)
    for (int dx = 0; dx < W; ++dx) {
        for (int dz = 0; dz < D; ++dz) {
            SET_VOXEL(0, dx, dz, COBBLESTONE);
        }
    }

    // Walls (planks) - 3 blocks tall
    for (int dy = 1; dy <= 3; ++dy) {
        // Front wall (Z = 0) with door opening in middle
        for (int dx = 0; dx < W; ++dx) {
            if (dx == 2 && dy <= 2) continue;  // door opening
            SET_VOXEL(dy, dx, 0, PLANKS);
        }
        // Back wall (Z = D-1)
        for (int dx = 0; dx < W; ++dx) {
            SET_VOXEL(dy, dx, D-1, PLANKS);
        }
        // Left wall (X = 0)
        for (int dz = 1; dz < D-1; ++dz) {
            SET_VOXEL(dy, 0, dz, PLANKS);
        }
        // Right wall (X = W-1)
        for (int dz = 1; dz < D-1; ++dz) {
            SET_VOXEL(dy, W-1, dz, PLANKS);
        }
    }

    // Corner pillars (oak log)
    for (int dy = 1; dy <= 3; ++dy) {
        SET_VOXEL(dy, 0, 0, OAKLOG);
        SET_VOXEL(dy, W-1, 0, OAKLOG);
        SET_VOXEL(dy, 0, D-1, OAKLOG);
        SET_VOXEL(dy, W-1, D-1, OAKLOG);
    }

    // Roof (brick) - flat roof
    for (int dx = 0; dx < W; ++dx) {
        for (int dz = 0; dz < D; ++dz) {
            SET_VOXEL(H, dx, dz, BRICK);
        }
    }

    #undef SET_VOXEL
}


// =============================================================================
// EPISODE MANAGEMENT KERNELS - For C++ step logic
// =============================================================================

// Called BEFORE step: checks if step_count >= episode_length, sets do_reset
__global__ void episode_check_kernel(
    const int* __restrict__ step_count,  // [B]
    bool* __restrict__ do_reset,         // [B]
    int batch_size,
    int episode_length  // 0 means no limit
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    if (episode_length > 0) {
        do_reset[i] = (step_count[i] >= episode_length);
    } else {
        do_reset[i] = false;
    }
}

// Called AFTER step: increments step_count, resets to 1 for done envs
__global__ void episode_update_kernel(
    int* __restrict__ step_count,        // [B]
    const bool* __restrict__ do_reset,   // [B]
    int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    // If this env was reset, step_count becomes 1 (first step after reset)
    // Otherwise increment
    if (do_reset[i]) {
        step_count[i] = 1;
    } else {
        step_count[i] = step_count[i] + 1;
    }
}

// Masked version: only regenerates voxels for environments where do_reset is true
__global__ void generate_world_kernel_masked(
    int8_t* __restrict__ voxels,         // [B, Y, X, Z]
    const bool* __restrict__ do_reset,   // [B]
    int batch_size,
    int world_x,
    int world_y,
    int world_z,
    int ground_height
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int wx = world_x;
    int wy = world_y;
    int wz = world_z;
    int wxz = wx * wz;
    long long wxyz = (long long)wx * wy * wz;
    long long total = (long long)batch_size * wxyz;
    if (idx >= total) return;

    int b = idx / wxyz;

    // Only regenerate for reset environments
    if (!do_reset[b]) return;

    int voxel_idx = idx % wxyz;
    int y = voxel_idx / wxz;
    int x = (voxel_idx % wxz) / wz;
    int z = voxel_idx % wz;

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

// Masked version: only places tree for environments where do_reset is true
__global__ void place_tree_kernel_masked(
    int8_t* __restrict__ voxels,         // [B, Y, X, Z]
    const bool* __restrict__ do_reset,   // [B]
    int batch_size,
    int world_x,
    int world_y,
    int world_z,
    int tree_x,
    int tree_z,
    int ground_height
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;

    // Only place tree for reset environments
    if (!do_reset[b]) return;

    int wx = world_x;
    int wy = world_y;
    int wz = world_z;
    int wxz = wx * wz;
    long long wxyz = (long long)wx * wy * wz;

    // Place 3-block tall log
    for (int dy = 0; dy < 3; ++dy) {
        int y = ground_height + dy;
        if (y < wy) {
            long long idx = (long long)b * wxyz + y * wxz + tree_x * wz + tree_z;
            voxels[idx] = OAKLOG;
        }
    }

    // Place leaves on top
    int leaf_y = ground_height + 3;
    if (leaf_y < wy) {
        long long idx = (long long)b * wxyz + leaf_y * wxz + tree_x * wz + tree_z;
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
    int block_size = 128;
    int num_blocks = (total_pixels + block_size - 1) / block_size;

    render_kernel<<<num_blocks, block_size>>>(
        voxels, cameras, output,
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void launch_render_uint8(
    const int8_t* voxels, const float* cameras, uint8_t* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees
) {
    int total_pixels = batch_size * height * width;
    int block_size = 128;
    int num_blocks = (total_pixels + block_size - 1) / block_size;

    render_kernel_uint8<<<num_blocks, block_size>>>(
        voxels, cameras, output,
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void launch_render_uint8_fast(
    const int8_t* voxels, const float* cameras, uint8_t* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees,
    int ground_height
) {
    int total_pixels = batch_size * height * width;
    int block_size = 128;
    int num_blocks = (total_pixels + block_size - 1) / block_size;

    render_kernel_uint8_fast<<<num_blocks, block_size>>>(
        voxels, cameras, output,
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees,
        ground_height
    );
}

void launch_render_uint8_minimal(
    const int8_t* voxels, const float* cameras, uint8_t* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees
) {
    int total_pixels = batch_size * height * width;
    int block_size = 128;
    int num_blocks = (total_pixels + block_size - 1) / block_size;

    render_kernel_uint8_minimal<<<num_blocks, block_size>>>(
        voxels, cameras, output,
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void launch_render_uint8_fp16(
    const int8_t* voxels, const float* cameras, uint8_t* output,
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees
) {
    int total_pixels = batch_size * height * width;
    int block_size = 128;
    int num_blocks = (total_pixels + block_size - 1) / block_size;

    render_kernel_uint8_fp16<<<num_blocks, block_size>>>(
        voxels, cameras, output,
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void launch_precompute_camera_basis(
    const float* cameras, float* basis,
    int batch_size, int width, int height, float fov_degrees
) {
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;

    precompute_camera_basis_kernel<<<num_blocks, block_size>>>(
        cameras, basis, batch_size, width, height, fov_degrees
    );
}

void launch_render_uint8_prebasis(
    const int8_t* voxels, const float* basis, uint8_t* output,
    int batch_size, int world_size, int width, int height, int max_steps
) {
    // 2D grid: y = batch element, x = pixel blocks within batch
    // This allows shared memory for camera basis per block
    int pixels_per_batch = height * width;
    int block_size = 128;
    int blocks_per_batch = (pixels_per_batch + block_size - 1) / block_size;

    dim3 grid(blocks_per_batch, batch_size);
    dim3 block(block_size);

    render_kernel_uint8_prebasis<<<grid, block>>>(
        voxels, basis, output,
        batch_size, world_size, world_size, world_size, width, height, max_steps
    );
}

void launch_render_uint8_smem(
    const int8_t* voxels, const float* basis, uint8_t* output,
    int batch_size, int world_size, int width, int height, int max_steps
) {
    // 2D grid: y = batch element, x = pixel blocks within batch
    int pixels_per_batch = height * width;
    int block_size = 128;
    int blocks_per_batch = (pixels_per_batch + block_size - 1) / block_size;

    dim3 grid(blocks_per_batch, batch_size);
    dim3 block(block_size);

    // Shared memory: 64 bytes for basis (aligned) + world_size^3 bytes for voxels
    int ws3 = world_size * world_size * world_size;
    int shared_mem_size = 64 + ws3;

    render_kernel_uint8_smem<<<grid, block, shared_mem_size>>>(
        voxels, basis, output,
        batch_size, world_size, width, height, max_steps
    );
}

void launch_render_uint8_coalesced(
    const int8_t* voxels,  // [Y, X, Z, B] layout!
    const float* basis,
    uint8_t* output,
    int batch_size, int world_size, int width, int height, int max_steps
) {
    // Grid: x = pixel index, y = batch blocks
    // Threads in warp handle same pixel across different batches
    int pixels_per_batch = height * width;
    int block_size = 128;  // 4 warps
    int batch_blocks = (batch_size + block_size - 1) / block_size;

    dim3 grid(pixels_per_batch, batch_blocks);
    dim3 block(block_size);

    render_kernel_uint8_coalesced<<<grid, block>>>(
        voxels, basis, output,
        batch_size, world_size, width, height, max_steps
    );
}

void launch_physics(
    float* positions, float* velocities, float* yaws, float* pitches,
    bool* on_ground, const float* forward_in, const float* strafe_in,
    const float* delta_yaw_in, const float* delta_pitch_in, const bool* jump_in,
    const float* speed_mult,
    int batch_size, int world_size, float dt, float gravity, float walk_speed, float jump_vel
) {
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;

    // Cubic world: pass world_size as all 3 dimensions
    physics_kernel<<<num_blocks, block_size>>>(
        positions, velocities, yaws, pitches, on_ground,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in, jump_in,
        speed_mult,
        batch_size, world_size, world_size, world_size, dt, gravity, walk_speed, jump_vel
    );
}

void launch_raycast_break(
    int8_t* voxels, const float* positions, const float* yaws, const float* pitches,
    const bool* do_break, float* rewards,
    int batch_size, int world_size, int8_t target_block, float reward_value
) {
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;

    // Cubic world: pass world_size as all 3 dimensions
    raycast_break_kernel<<<num_blocks, block_size>>>(
        voxels, positions, yaws, pitches, do_break, rewards,
        batch_size, world_size, world_size, world_size, target_block, reward_value
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
    int8_t* voxels, int batch_size, int world_x, int world_y, int world_z, int ground_height, unsigned int seed
) {
    long long total = (long long)batch_size * world_x * world_y * world_z;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    generate_world_kernel<<<num_blocks, block_size>>>(
        voxels, batch_size, world_x, world_y, world_z, ground_height, seed
    );
}

void launch_place_tree(
    int8_t* voxels, int batch_size, int world_x, int world_y, int world_z, int tree_x, int tree_z, int ground_height
) {
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;

    place_tree_kernel<<<num_blocks, block_size>>>(
        voxels, batch_size, world_x, world_y, world_z, tree_x, tree_z, ground_height
    );
}

void launch_place_house(
    int8_t* voxels, int batch_size, int world_x, int world_y, int world_z, int house_x, int house_z, int ground_height
) {
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;

    place_house_kernel<<<num_blocks, block_size>>>(
        voxels, batch_size, world_x, world_y, world_z, house_x, house_z, ground_height
    );
}

// =============================================================================
// NEW STEP API - Unified step function with internal buffers
// =============================================================================

// Simple kernel to update camera buffer from agent state
__global__ void update_cameras_kernel(
    float* __restrict__ cameras,           // [B, 5] output
    const float* __restrict__ positions,   // [B, 3]
    const float* __restrict__ yaws,        // [B]
    const float* __restrict__ pitches,     // [B]
    int batch_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_size) return;

    cameras[i * 5 + 0] = positions[i * 3 + 0];  // x
    cameras[i * 5 + 1] = positions[i * 3 + 1];  // y
    cameras[i * 5 + 2] = positions[i * 3 + 2];  // z
    cameras[i * 5 + 3] = yaws[i];
    cameras[i * 5 + 4] = pitches[i];
}

// CUDA Graph state for step function
struct StepGraphState {
    bool captured = false;
    bool initialized = false;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t instance = nullptr;
    cudaStream_t stream = nullptr;  // Dedicated stream for graph operations
    int warmup_count = 0;
    static constexpr int WARMUP_THRESHOLD = 10;

    // Cached parameters to detect changes requiring re-capture
    int last_batch_size = 0;
    int last_width = 0;
    int last_height = 0;

    // Track tensor pointers - graph captures these, so changes require recapture
    void* last_voxels = nullptr;
    void* last_positions = nullptr;
    void* last_obs_buffer = nullptr;

    void init() {
        if (!initialized) {
            cudaStreamCreate(&stream);
            initialized = true;
        }
    }

    void reset() {
        if (captured) {
            cudaGraphExecDestroy(instance);
            cudaGraphDestroy(graph);
            captured = false;
            instance = nullptr;
            graph = nullptr;
        }
        warmup_count = 0;
    }
};

static StepGraphState g_step_graph;

// Separate graph state for uint8_prebasis step
static StepGraphState g_step_graph_prebasis;

// Graph state for full step (includes episode management and world regen)
static StepGraphState g_step_graph_prebasis_full;

void launch_decode_actions(
    const int8_t* buttons, const float* look,
    float* forward_in, float* strafe_in,
    float* delta_yaw_in, float* delta_pitch_in,
    bool* jump_in, bool* do_break, bool* do_place, float* speed_mult,
    int batch_size, float sprint_multiplier
) {
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;

    decode_actions_kernel<<<num_blocks, block_size>>>(
        buttons, look,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
        jump_in, do_break, do_place, speed_mult,
        batch_size, sprint_multiplier
    );
}

void launch_physics_v2(
    float* positions, float* velocities, float* yaws, float* pitches,
    bool* on_ground, const float* forward_in, const float* strafe_in,
    const float* delta_yaw_in, const float* delta_pitch_in, const bool* jump_in,
    const float* speed_mult,
    int batch_size, int world_size, float dt, float gravity, float walk_speed, float jump_vel
) {
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;

    // Cubic world: pass world_size as all 3 dimensions
    physics_kernel<<<num_blocks, block_size>>>(
        positions, velocities, yaws, pitches, on_ground,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in, jump_in, speed_mult,
        batch_size, world_size, world_size, world_size, dt, gravity, walk_speed, jump_vel
    );
}

void launch_raycast_interact(
    int8_t* voxels, const float* positions, const float* yaws, const float* pitches,
    const bool* do_break, const bool* do_place, float* rewards,
    int batch_size, int world_size, int8_t target_block, float reward_value
) {
    int block_size = 256;
    int num_blocks = (batch_size + block_size - 1) / block_size;

    // Cubic world: pass world_size as all 3 dimensions
    raycast_interact_kernel<<<num_blocks, block_size>>>(
        voxels, positions, yaws, pitches, do_break, do_place, rewards,
        batch_size, world_size, world_size, world_size, target_block, reward_value
    );
}

// Helper to launch all step kernels (used for both eager and graph capture)
static void launch_step_kernels(
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
    float spawn_x, float spawn_y, float spawn_z,
    cudaStream_t stream
) {
    int block_size = 256;
    int num_blocks_batch = (batch_size + block_size - 1) / block_size;
    int total_pixels = batch_size * height * width;
    int num_blocks_render = (total_pixels + 127) / 128;

    // 1. Decode actions -> control signals
    decode_actions_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        buttons, look,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
        jump_in, do_break, do_place, speed_mult,
        batch_size, sprint_mult
    );

    // 2. Physics step (cubic world)
    physics_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        positions, velocities, yaws, pitches, on_ground,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in, jump_in, speed_mult,
        batch_size, world_size, world_size, world_size, dt, gravity, walk_speed, jump_vel
    );

    // 3. Block interaction (break/place, cubic world)
    raycast_interact_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        voxels, positions, yaws, pitches, do_break, do_place, rewards,
        batch_size, world_size, world_size, world_size, target_block, reward_value
    );

    // 4. Reset (always run with mask)
    reset_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        positions, velocities, yaws, pitches, on_ground,
        do_reset, batch_size, spawn_x, spawn_y, spawn_z
    );

    // 5. Update camera buffer from agent state
    update_cameras_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        cameras, positions, yaws, pitches, batch_size
    );

    // 6. Render (cubic world - render_kernel takes single world_size)
    render_kernel<<<num_blocks_render, 128, 0, stream>>>(
        voxels, cameras, obs_buffer,
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

void launch_step(
    // World state
    int8_t* voxels,
    // Agent state
    float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
    // Actions (input)
    const int8_t* buttons, const float* look,
    // Reset mask
    const bool* do_reset,
    // Internal buffers (pre-allocated by caller)
    float* forward_in, float* strafe_in, float* delta_yaw_in, float* delta_pitch_in,
    bool* jump_in, bool* do_break, bool* do_place, float* speed_mult,
    // Camera buffer for render
    float* cameras,
    // Output
    float* obs_buffer, float* rewards,
    // Config
    int batch_size, int world_size, int width, int height,
    int max_steps, float view_distance, float fov_degrees,
    float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
    int8_t target_block, float reward_value,
    float spawn_x, float spawn_y, float spawn_z
) {
    // Initialize stream on first call
    g_step_graph.init();
    cudaStream_t stream = g_step_graph.stream;

    // Check if we need to invalidate the graph (config or tensor pointers changed)
    // CUDA graphs capture memory addresses, so different envs need recapture
    bool config_changed = (batch_size != g_step_graph.last_batch_size ||
                          width != g_step_graph.last_width ||
                          height != g_step_graph.last_height);
    bool pointers_changed = (voxels != g_step_graph.last_voxels ||
                            positions != g_step_graph.last_positions ||
                            obs_buffer != g_step_graph.last_obs_buffer);

    if (config_changed || pointers_changed) {
        g_step_graph.reset();
    }

    g_step_graph.last_batch_size = batch_size;
    g_step_graph.last_width = width;
    g_step_graph.last_height = height;
    g_step_graph.last_voxels = voxels;
    g_step_graph.last_positions = positions;
    g_step_graph.last_obs_buffer = obs_buffer;

    if (g_step_graph.captured) {
        // Fast path: replay captured graph
        cudaGraphLaunch(g_step_graph.instance, stream);
        cudaStreamSynchronize(stream);  // Ensure graph completes before returning
    } else if (g_step_graph.warmup_count >= StepGraphState::WARMUP_THRESHOLD) {
        // Capture graph on dedicated stream
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        launch_step_kernels(
            voxels, positions, velocities, yaws, pitches, on_ground,
            buttons, look, do_reset,
            forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
            jump_in, do_break, do_place, speed_mult,
            cameras, obs_buffer, rewards,
            batch_size, world_size, width, height,
            max_steps, view_distance, fov_degrees,
            dt, gravity, walk_speed, sprint_mult, jump_vel,
            target_block, reward_value,
            spawn_x, spawn_y, spawn_z,
            stream
        );

        cudaStreamEndCapture(stream, &g_step_graph.graph);
        cudaGraphInstantiate(&g_step_graph.instance, g_step_graph.graph, nullptr, nullptr, 0);
        g_step_graph.captured = true;

        // Launch the newly captured graph
        cudaGraphLaunch(g_step_graph.instance, stream);
        cudaStreamSynchronize(stream);
    } else {
        // Warmup: eager execution on dedicated stream
        launch_step_kernels(
            voxels, positions, velocities, yaws, pitches, on_ground,
            buttons, look, do_reset,
            forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
            jump_in, do_break, do_place, speed_mult,
            cameras, obs_buffer, rewards,
            batch_size, world_size, width, height,
            max_steps, view_distance, fov_degrees,
            dt, gravity, walk_speed, sprint_mult, jump_vel,
            target_block, reward_value,
            spawn_x, spawn_y, spawn_z,
            stream
        );
        cudaStreamSynchronize(stream);  // Ensure kernels complete
        g_step_graph.warmup_count++;
    }
}

// UINT8 version of step - no CUDA graphs (bandwidth reduction is the main win)
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
) {
    int block_size = 256;
    int num_blocks_batch = (batch_size + block_size - 1) / block_size;
    int total_pixels = batch_size * height * width;
    int num_blocks_render = (total_pixels + 127) / 128;

    // 1. Decode actions
    decode_actions_kernel<<<num_blocks_batch, block_size>>>(
        buttons, look,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
        jump_in, do_break, do_place, speed_mult,
        batch_size, sprint_mult
    );

    // 2. Physics (cubic world: pass world_size for all three dims)
    physics_kernel<<<num_blocks_batch, block_size>>>(
        positions, velocities, yaws, pitches, on_ground,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in, jump_in, speed_mult,
        batch_size, world_size, world_size, world_size, dt, gravity, walk_speed, jump_vel
    );

    // 3. Block interaction (cubic world)
    raycast_interact_kernel<<<num_blocks_batch, block_size>>>(
        voxels, positions, yaws, pitches, do_break, do_place, rewards,
        batch_size, world_size, world_size, world_size, target_block, reward_value
    );

    // 4. Reset
    reset_kernel<<<num_blocks_batch, block_size>>>(
        positions, velocities, yaws, pitches, on_ground,
        do_reset, batch_size, spawn_x, spawn_y, spawn_z
    );

    // 5. Update cameras
    update_cameras_kernel<<<num_blocks_batch, block_size>>>(
        cameras, positions, yaws, pitches, batch_size
    );

    // 6. Render (uint8 output - takes single world_size)
    render_kernel_uint8<<<num_blocks_render, 128>>>(
        voxels, cameras, obs_buffer,
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

// UINT8 Fast version - uses ground-accelerated rendering for flat worlds
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
    int ground_height  // NEW: for ground-accelerated rendering
) {
    int block_size = 256;
    int num_blocks_batch = (batch_size + block_size - 1) / block_size;
    int total_pixels = batch_size * height * width;
    int num_blocks_render = (total_pixels + 127) / 128;

    // 1-5: Same as regular step
    decode_actions_kernel<<<num_blocks_batch, block_size>>>(
        buttons, look,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
        jump_in, do_break, do_place, speed_mult,
        batch_size, sprint_mult
    );

    // Physics (cubic world)
    physics_kernel<<<num_blocks_batch, block_size>>>(
        positions, velocities, yaws, pitches, on_ground,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in, jump_in, speed_mult,
        batch_size, world_size, world_size, world_size, dt, gravity, walk_speed, jump_vel
    );

    // Block interaction (cubic world)
    raycast_interact_kernel<<<num_blocks_batch, block_size>>>(
        voxels, positions, yaws, pitches, do_break, do_place, rewards,
        batch_size, world_size, world_size, world_size, target_block, reward_value
    );

    reset_kernel<<<num_blocks_batch, block_size>>>(
        positions, velocities, yaws, pitches, on_ground,
        do_reset, batch_size, spawn_x, spawn_y, spawn_z
    );

    update_cameras_kernel<<<num_blocks_batch, block_size>>>(
        cameras, positions, yaws, pitches, batch_size
    );

    // 6. Render with ground-accelerated kernel (takes single world_size)
    render_kernel_uint8_fast<<<num_blocks_render, 128>>>(
        voxels, cameras, obs_buffer,
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees,
        ground_height
    );
}

// UINT8 Minimal version - flat colors, no fog/shading for max perf
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
) {
    int block_size = 256;
    int num_blocks_batch = (batch_size + block_size - 1) / block_size;
    int total_pixels = batch_size * height * width;
    int num_blocks_render = (total_pixels + 127) / 128;

    // 1. Decode actions
    decode_actions_kernel<<<num_blocks_batch, block_size>>>(
        buttons, look,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
        jump_in, do_break, do_place, speed_mult,
        batch_size, sprint_mult
    );

    // 2. Physics (cubic world)
    physics_kernel<<<num_blocks_batch, block_size>>>(
        positions, velocities, yaws, pitches, on_ground,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in, jump_in, speed_mult,
        batch_size, world_size, world_size, world_size, dt, gravity, walk_speed, jump_vel
    );

    // 3. Block interaction (cubic world)
    raycast_interact_kernel<<<num_blocks_batch, block_size>>>(
        voxels, positions, yaws, pitches, do_break, do_place, rewards,
        batch_size, world_size, world_size, world_size, target_block, reward_value
    );

    // 4. Reset
    reset_kernel<<<num_blocks_batch, block_size>>>(
        positions, velocities, yaws, pitches, on_ground,
        do_reset, batch_size, spawn_x, spawn_y, spawn_z
    );

    // 5. Update cameras
    update_cameras_kernel<<<num_blocks_batch, block_size>>>(
        cameras, positions, yaws, pitches, batch_size
    );

    // 6. Render with minimal kernel (takes single world_size)
    render_kernel_uint8_minimal<<<num_blocks_render, 128>>>(
        voxels, cameras, obs_buffer,
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

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
) {
    int block_size = 256;
    int num_blocks_batch = (batch_size + block_size - 1) / block_size;
    int total_pixels = batch_size * height * width;
    int num_blocks_render = (total_pixels + 127) / 128;

    // 1. Decode actions
    decode_actions_kernel<<<num_blocks_batch, block_size>>>(
        buttons, look,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
        jump_in, do_break, do_place, speed_mult,
        batch_size, sprint_mult
    );

    // 2. Physics (cubic world: world_size for all dimensions)
    physics_kernel<<<num_blocks_batch, block_size>>>(
        positions, velocities, yaws, pitches, on_ground,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in, jump_in, speed_mult,
        batch_size, world_size, world_size, world_size, dt, gravity, walk_speed, jump_vel
    );

    // 3. Block interaction (cubic world)
    raycast_interact_kernel<<<num_blocks_batch, block_size>>>(
        voxels, positions, yaws, pitches, do_break, do_place, rewards,
        batch_size, world_size, world_size, world_size, target_block, reward_value
    );

    // 4. Reset
    reset_kernel<<<num_blocks_batch, block_size>>>(
        positions, velocities, yaws, pitches, on_ground,
        do_reset, batch_size, spawn_x, spawn_y, spawn_z
    );

    // 5. Update cameras
    update_cameras_kernel<<<num_blocks_batch, block_size>>>(
        cameras, positions, yaws, pitches, batch_size
    );

    // 6. Render with FP16 DDA kernel (takes single world_size)
    render_kernel_uint8_fp16<<<num_blocks_render, 128>>>(
        voxels, cameras, obs_buffer,
        batch_size, world_size, width, height,
        max_steps, view_distance, fov_degrees
    );
}

// Helper to launch prebasis kernels on a specific stream
static void launch_step_uint8_prebasis_kernels(
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
    float spawn_x, float spawn_y, float spawn_z,
    cudaStream_t stream
) {
    int block_size = 256;
    int num_blocks_batch = (batch_size + block_size - 1) / block_size;

    // 2D grid for render: y = batch element, x = pixel blocks within batch
    int pixels_per_batch = height * width;
    int blocks_per_batch = (pixels_per_batch + 127) / 128;
    dim3 render_grid(blocks_per_batch, batch_size);
    dim3 render_block(128);

    // 1. Decode actions
    decode_actions_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        buttons, look,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
        jump_in, do_break, do_place, speed_mult,
        batch_size, sprint_mult
    );

    // 2. Physics
    physics_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        positions, velocities, yaws, pitches, on_ground,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in, jump_in, speed_mult,
        batch_size, world_x, world_y, world_z, dt, gravity, walk_speed, jump_vel
    );

    // 3. Block interaction
    raycast_interact_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        voxels, positions, yaws, pitches, do_break, do_place, rewards,
        batch_size, world_x, world_y, world_z, target_block, reward_value
    );

    // 4. Reset
    reset_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        positions, velocities, yaws, pitches, on_ground,
        do_reset, batch_size, spawn_x, spawn_y, spawn_z
    );

    // 5. Update cameras
    update_cameras_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        cameras, positions, yaws, pitches, batch_size
    );

    // 6. Precompute camera basis (trig once per camera)
    precompute_camera_basis_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        cameras, basis, batch_size, width, height, fov_degrees
    );

    // 7. Render with precomputed basis (no per-pixel trig, shared memory for basis)
    render_kernel_uint8_prebasis<<<render_grid, render_block, 0, stream>>>(
        voxels, basis, obs_buffer,
        batch_size, world_x, world_y, world_z, width, height, max_steps
    );
}

// Fast step helper: episode check/update in C++, but world regen handled by Python
// This avoids the 0.5ms cost of launching masked world regen every step
static void launch_step_uint8_prebasis_fast_kernels(
    int8_t* voxels,
    float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
    const int8_t* buttons, const float* look,
    bool* do_reset,  // Writable: set by episode_check_kernel
    int* step_count, // [B] step counter, updated by episode_update_kernel
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
    int episode_length,  // 0 = no limit
    cudaStream_t stream
) {
    int block_size = 256;
    int num_blocks_batch = (batch_size + block_size - 1) / block_size;

    // 2D grid for render: y = batch element, x = pixel blocks within batch
    int pixels_per_batch = height * width;
    int blocks_per_batch = (pixels_per_batch + 127) / 128;
    dim3 render_grid(blocks_per_batch, batch_size);
    dim3 render_block(128);

    // 0. Episode check: set do_reset based on step_count >= episode_length
    episode_check_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        step_count, do_reset, batch_size, episode_length
    );

    // 1. Decode actions
    decode_actions_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        buttons, look,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
        jump_in, do_break, do_place, speed_mult,
        batch_size, sprint_mult
    );

    // 2. Physics
    physics_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        positions, velocities, yaws, pitches, on_ground,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in, jump_in, speed_mult,
        batch_size, world_x, world_y, world_z, dt, gravity, walk_speed, jump_vel
    );

    // 3. Block interaction
    raycast_interact_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        voxels, positions, yaws, pitches, do_break, do_place, rewards,
        batch_size, world_x, world_y, world_z, target_block, reward_value
    );

    // 4. Reset agent state for done environments
    reset_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        positions, velocities, yaws, pitches, on_ground,
        do_reset, batch_size, spawn_x, spawn_y, spawn_z
    );

    // 5. Update step count: increment for active envs, reset to 1 for done envs
    episode_update_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        step_count, do_reset, batch_size
    );

    // NOTE: World regen handled by Python (saves 0.5ms when no resets happen)

    // 6. Update cameras
    update_cameras_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        cameras, positions, yaws, pitches, batch_size
    );

    // 7. Precompute camera basis (trig once per camera)
    precompute_camera_basis_kernel<<<num_blocks_batch, block_size, 0, stream>>>(
        cameras, basis, batch_size, width, height, fov_degrees
    );

    // 8. Render with precomputed basis (no per-pixel trig, shared memory for basis)
    render_kernel_uint8_prebasis<<<render_grid, render_block, 0, stream>>>(
        voxels, basis, obs_buffer,
        batch_size, world_x, world_y, world_z, width, height, max_steps
    );
}

void launch_step_uint8_prebasis(
    int8_t* voxels,
    float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
    const int8_t* buttons, const float* look,
    const bool* do_reset,
    float* forward_in, float* strafe_in, float* delta_yaw_in, float* delta_pitch_in,
    bool* jump_in, bool* do_break, bool* do_place, float* speed_mult,
    float* cameras,
    float* basis,  // [B, 14] precomputed camera basis buffer
    uint8_t* obs_buffer, float* rewards,
    int batch_size, int world_x, int world_y, int world_z, int width, int height,
    int max_steps, float view_distance, float fov_degrees,
    float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
    int8_t target_block, float reward_value,
    float spawn_x, float spawn_y, float spawn_z
) {
    // Initialize stream on first call
    g_step_graph_prebasis.init();
    cudaStream_t stream = g_step_graph_prebasis.stream;

    // Check if we need to invalidate the graph (config or tensor pointers changed)
    bool config_changed = (batch_size != g_step_graph_prebasis.last_batch_size ||
                          width != g_step_graph_prebasis.last_width ||
                          height != g_step_graph_prebasis.last_height);
    bool pointers_changed = (voxels != g_step_graph_prebasis.last_voxels ||
                            positions != g_step_graph_prebasis.last_positions ||
                            obs_buffer != g_step_graph_prebasis.last_obs_buffer);

    if (config_changed || pointers_changed) {
        g_step_graph_prebasis.reset();
    }

    g_step_graph_prebasis.last_batch_size = batch_size;
    g_step_graph_prebasis.last_width = width;
    g_step_graph_prebasis.last_height = height;
    g_step_graph_prebasis.last_voxels = voxels;
    g_step_graph_prebasis.last_positions = positions;
    g_step_graph_prebasis.last_obs_buffer = obs_buffer;

    if (g_step_graph_prebasis.captured) {
        // Fast path: replay captured graph
        cudaGraphLaunch(g_step_graph_prebasis.instance, stream);
    } else if (g_step_graph_prebasis.warmup_count >= StepGraphState::WARMUP_THRESHOLD) {
        // Capture graph
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        launch_step_uint8_prebasis_kernels(
            voxels, positions, velocities, yaws, pitches, on_ground,
            buttons, look, do_reset,
            forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
            jump_in, do_break, do_place, speed_mult,
            cameras, basis, obs_buffer, rewards,
            batch_size, world_x, world_y, world_z, width, height,
            max_steps, view_distance, fov_degrees,
            dt, gravity, walk_speed, sprint_mult, jump_vel,
            target_block, reward_value,
            spawn_x, spawn_y, spawn_z,
            stream
        );

        cudaStreamEndCapture(stream, &g_step_graph_prebasis.graph);
        cudaGraphInstantiate(&g_step_graph_prebasis.instance, g_step_graph_prebasis.graph, nullptr, nullptr, 0);
        g_step_graph_prebasis.captured = true;

        // Launch the newly captured graph
        cudaGraphLaunch(g_step_graph_prebasis.instance, stream);
    } else {
        // Warmup: eager execution
        launch_step_uint8_prebasis_kernels(
            voxels, positions, velocities, yaws, pitches, on_ground,
            buttons, look, do_reset,
            forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
            jump_in, do_break, do_place, speed_mult,
            cameras, basis, obs_buffer, rewards,
            batch_size, world_x, world_y, world_z, width, height,
            max_steps, view_distance, fov_degrees,
            dt, gravity, walk_speed, sprint_mult, jump_vel,
            target_block, reward_value,
            spawn_x, spawn_y, spawn_z,
            stream
        );
        g_step_graph_prebasis.warmup_count++;
    }
}

// Fast step: episode check/update in C++, world regen left to Python
// This eliminates PyTorch kernels for episode management while avoiding
// the 0.5ms cost of launching masked world regen every step
void launch_step_uint8_prebasis_full(
    int8_t* voxels,
    float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
    const int8_t* buttons, const float* look,
    bool* do_reset,  // Writable: set by episode_check_kernel
    int* step_count, // [B] step counter, updated by episode_update_kernel
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
    int episode_length  // 0 = no limit (unused params removed: ground_height, tree_x, tree_z)
) {
    // Initialize stream on first call
    g_step_graph_prebasis_full.init();
    cudaStream_t stream = g_step_graph_prebasis_full.stream;

    // Check if we need to invalidate the graph (config or tensor pointers changed)
    bool config_changed = (batch_size != g_step_graph_prebasis_full.last_batch_size ||
                          width != g_step_graph_prebasis_full.last_width ||
                          height != g_step_graph_prebasis_full.last_height);
    bool pointers_changed = (voxels != g_step_graph_prebasis_full.last_voxels ||
                            positions != g_step_graph_prebasis_full.last_positions ||
                            obs_buffer != g_step_graph_prebasis_full.last_obs_buffer);

    if (config_changed || pointers_changed) {
        g_step_graph_prebasis_full.reset();
    }

    g_step_graph_prebasis_full.last_batch_size = batch_size;
    g_step_graph_prebasis_full.last_width = width;
    g_step_graph_prebasis_full.last_height = height;
    g_step_graph_prebasis_full.last_voxels = voxels;
    g_step_graph_prebasis_full.last_positions = positions;
    g_step_graph_prebasis_full.last_obs_buffer = obs_buffer;

    if (g_step_graph_prebasis_full.captured) {
        // Fast path: replay captured graph
        cudaGraphLaunch(g_step_graph_prebasis_full.instance, stream);
    } else if (g_step_graph_prebasis_full.warmup_count >= StepGraphState::WARMUP_THRESHOLD) {
        // Capture graph
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

        launch_step_uint8_prebasis_fast_kernels(
            voxels, positions, velocities, yaws, pitches, on_ground,
            buttons, look, do_reset, step_count,
            forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
            jump_in, do_break, do_place, speed_mult,
            cameras, basis, obs_buffer, rewards,
            batch_size, world_x, world_y, world_z, width, height,
            max_steps, view_distance, fov_degrees,
            dt, gravity, walk_speed, sprint_mult, jump_vel,
            target_block, reward_value,
            spawn_x, spawn_y, spawn_z,
            episode_length,
            stream
        );

        cudaStreamEndCapture(stream, &g_step_graph_prebasis_full.graph);
        cudaGraphInstantiate(&g_step_graph_prebasis_full.instance, g_step_graph_prebasis_full.graph, nullptr, nullptr, 0);
        g_step_graph_prebasis_full.captured = true;

        // Launch the newly captured graph
        cudaGraphLaunch(g_step_graph_prebasis_full.instance, stream);
    } else {
        // Warmup: eager execution
        launch_step_uint8_prebasis_fast_kernels(
            voxels, positions, velocities, yaws, pitches, on_ground,
            buttons, look, do_reset, step_count,
            forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
            jump_in, do_break, do_place, speed_mult,
            cameras, basis, obs_buffer, rewards,
            batch_size, world_x, world_y, world_z, width, height,
            max_steps, view_distance, fov_degrees,
            dt, gravity, walk_speed, sprint_mult, jump_vel,
            target_block, reward_value,
            spawn_x, spawn_y, spawn_z,
            episode_length,
            stream
        );
        g_step_graph_prebasis_full.warmup_count++;
    }
}

void launch_step_uint8_smem(
    int8_t* voxels,
    float* positions, float* velocities, float* yaws, float* pitches, bool* on_ground,
    const int8_t* buttons, const float* look,
    const bool* do_reset,
    float* forward_in, float* strafe_in, float* delta_yaw_in, float* delta_pitch_in,
    bool* jump_in, bool* do_break, bool* do_place, float* speed_mult,
    float* cameras,
    float* basis,  // [B, 14] precomputed camera basis buffer
    uint8_t* obs_buffer, float* rewards,
    int batch_size, int world_x, int world_y, int world_z, int width, int height,
    int max_steps, float view_distance, float fov_degrees,
    float dt, float gravity, float walk_speed, float sprint_mult, float jump_vel,
    int8_t target_block, float reward_value,
    float spawn_x, float spawn_y, float spawn_z
) {
    int block_size = 256;
    int num_blocks_batch = (batch_size + block_size - 1) / block_size;

    // 2D grid for render: y = batch element, x = pixel blocks within batch
    int pixels_per_batch = height * width;
    int blocks_per_batch = (pixels_per_batch + 127) / 128;
    dim3 render_grid(blocks_per_batch, batch_size);
    dim3 render_block(128);

    // Shared memory: 64 bytes for basis (aligned) + world voxel bytes
    // NOTE: smem kernel only works for cubic worlds currently
    int wxyz = world_x * world_y * world_z;
    int shared_mem_size = 64 + wxyz;

    // 1. Decode actions
    decode_actions_kernel<<<num_blocks_batch, block_size>>>(
        buttons, look,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in,
        jump_in, do_break, do_place, speed_mult,
        batch_size, sprint_mult
    );

    // 2. Physics
    physics_kernel<<<num_blocks_batch, block_size>>>(
        positions, velocities, yaws, pitches, on_ground,
        forward_in, strafe_in, delta_yaw_in, delta_pitch_in, jump_in, speed_mult,
        batch_size, world_x, world_y, world_z, dt, gravity, walk_speed, jump_vel
    );

    // 3. Block interaction
    raycast_interact_kernel<<<num_blocks_batch, block_size>>>(
        voxels, positions, yaws, pitches, do_break, do_place, rewards,
        batch_size, world_x, world_y, world_z, target_block, reward_value
    );

    // 4. Reset
    reset_kernel<<<num_blocks_batch, block_size>>>(
        positions, velocities, yaws, pitches, on_ground,
        do_reset, batch_size, spawn_x, spawn_y, spawn_z
    );

    // 5. Update cameras
    update_cameras_kernel<<<num_blocks_batch, block_size>>>(
        cameras, positions, yaws, pitches, batch_size
    );

    // 6. Precompute camera basis (trig once per camera)
    precompute_camera_basis_kernel<<<num_blocks_batch, block_size>>>(
        cameras, basis, batch_size, width, height, fov_degrees
    );

    // 7. Render with shared memory voxels (entire voxel grid in smem per block)
    // NOTE: render_kernel_uint8_smem still uses world_size, needs update for non-cubic
    render_kernel_uint8_smem<<<render_grid, render_block, shared_mem_size>>>(
        voxels, basis, obs_buffer,
        batch_size, world_x, width, height, max_steps
    );
}

}  // extern "C"
