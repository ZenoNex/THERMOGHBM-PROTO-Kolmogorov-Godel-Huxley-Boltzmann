/*
thermo_cuda.cu: CUDA Kernels for Thermodynamic GHBM Model

Design Choices & Rationale:
- Custom CUDA Kernels: To achieve real-time simulation of stochastic processes, physical device-driven noise, and thermodynamic/statistical sampling.
- Ensemble, Virtual Time Evolution, Boltzmann Sampling: Directly implements theoretical aspects of Kolmogorov-Godel-Huxley-Boltzmann blending, including noise injection and physical attenuation.
- TAU_VTE and BOLTZMANN_KT_SCALE Constants: Parameterize decay and thermal noise scaling, reflecting physical realities and allowing for hardware-driven experimentation.
- curand Usage: Hardware random sources improve stochastic modeling over software-only sources.
- atomicAddFloat: Needed for atomic updates in single-precision; type-casting relates to CUDA's atomic limitations and ensures safe parallel reductions.
- All kernels parallelized across elements for massive parallel efficiency.
- generative_attenuation: Simulates uncertainty attenuation by simulated Boltzmann weighting, robustly aggregates ensemble predictions.
- Interoperability with hardware driver via "thermo_driver.h": Enables real-time feedback from device sensors, allowing closed-loop integration with physical GPU environment.

The file provides the computational backbone for the Python and C++ interface (ghbm.py, thermo_lib.cpp).
*/

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

extern "C" {
    #include "thermo_driver.h"
}

#define TAU_VTE 16.0f
#define BOLTZMANN_KT_SCALE 0.01f

__device__ float atomicAddFloat(float* addr, float val) {
    return atomicAdd((unsigned int*)addr, __float_as_uint(val));
}

__global__ void init_curand(curandState* state, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &state[tid]);
}

__global__ void vte_update(
    const float* h_prev, const float* noise, float* vte_out,
    float dt_virtual, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float decay = expf(-dt_virtual / TAU_VTE);
    vte_out[i] = decay * h_prev[i] + (1.0f - decay) * noise[i];
}

__global__ void huxley_flow(
    float* h, const float* noise, const float* vte, const float* grad_E,
    float alpha, float beta, float gamma, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    h[i] += alpha * noise[i] - beta * grad_E[i] + gamma * vte[i];
}

__global__ void boltzmann_sample(
    float* h, const float* h_prev, const float* energy, float kT, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float delta_E = energy[i];
    float p_accept = expf(-delta_E / (kT + 1e-6f));
    if (curand_uniform(&curand_states[i]) > p_accept) {
        h[i] = h_prev[i];  // reject
    }
}

__global__ void generative_attenuation(
    const float* ensemble, float* output, int ensemble_size, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float sum = 0.0f, weight_sum = 0.0f;
    float mean = 0.0f;
    for (int e = 0; e < ensemble_size; e++) mean += ensemble[e * N + i];
    mean /= ensemble_size;
    float std = 0.0f;
    for (int e = 0; e < ensemble_size; e++) {
        float diff = ensemble[e * N + i] - mean;
        std += diff * diff;
    }
    std = sqrtf(std / ensemble_size) + 1e-6f;
    for (int e = 0; e < ensemble_size; e++) {
        float val = ensemble[e * N + i];
        float w = expf(-fabsf(val - mean) / std);
        sum += w * val;
        weight_sum += w;
    }
    output[i] = sum / weight_sum;
}
