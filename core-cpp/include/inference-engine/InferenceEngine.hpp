#pragma once

#include "TensorInfo.hpp"

#include <cstdint>
#include <vector>

namespace inference_engine
{
class InferenceEngine
{
public:
    virtual const std::vector<TensorInfo> &get_input_info() const = 0;
    virtual const std::vector<TensorInfo> &get_output_info() const = 0;

    virtual void run(
        const float *const *input_data,
        float **output_data,
        const int64_t *const *input_shapes = nullptr,
        const int64_t *const *output_shapes = nullptr
    ) = 0;
};
} // namespace inference_engine
