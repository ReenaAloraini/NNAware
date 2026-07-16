#pragma once
#include<math.h>
#include <cstdint>

enum class NNActivationType : uint8_t { RELU = 0, SIGMOID = 1, TANH = 2, LINEAR = 3 };

inline float applyActivation(NNActivationType type, float x) {
    switch (type) {
        case NNActivationType::RELU:    return x > 0.0f ? x : 0.0f;
        case NNActivationType::SIGMOID: return 1.0f / (1.0f + expf(-x));
        case NNActivationType::TANH:    return tanhf(x);
        case NNActivationType::LINEAR:  return x;
        default:                        return x;
    }
}