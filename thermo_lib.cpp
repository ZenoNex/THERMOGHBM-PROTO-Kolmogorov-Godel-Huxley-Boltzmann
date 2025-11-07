// thermo_lib.cpp
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <iostream>
#include <vector>
#include "thermo_driver.h"

class ThermoRuntime {
private:
    int fd;
    struct thermo_sample sample;
    curandState* d_curand;
    float* d_vte, *d_noise, *d_h, *d_ensemble;
    int N, ensemble_size;

public:
    ThermoRuntime(int hidden_size, int ens = 8) : N(hidden_size), ensemble_size(ens) {
        fd = open("/dev/thermo0", O_RDONLY);
        if (fd < 0) throw std::runtime_error("Cannot open /dev/thermo0");

        cudaMalloc(&d_curand, N * sizeof(curandState));
        cudaMalloc(&d_vte, N * sizeof(float));
        cudaMalloc(&d_noise, N * sizeof(float));
        cudaMalloc(&d_h, N * ensemble_size * sizeof(float));
        cudaMalloc(&d_ensemble, N * sizeof(float));

        init_curand<<<(N+255)/256, 256>>>(d_curand, time(NULL));
    }

    void poll_hardware() {
        read(fd, &sample, sizeof(sample));
        std::vector<float> noise(N);
        for (int i = 0; i < N; i++) {
            noise[i] = sample.adc_noise * 0.1f + (sample.jitter[i] % 100) * 1e-3f;
        }
        cudaMemcpy(d_noise, noise.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    }

    void step_ghbm(float dt_virtual = 1.0f) {
        poll_hardware();
        vte_update<<<(N+255)/256, 256>>>(d_h, d_noise, d_vte, dt_virtual, N);
        // ... huxley + boltzmann + attenuation (call kernels)
    }

    ~ThermoRuntime() {
        close(fd);
        cudaFree(d_curand); cudaFree(d_vte); cudaFree(d_noise);
        cudaFree(d_h); cudaFree(d_ensemble);
    }
};
