#pragma once

#include <cstdint>
#include <vector>

namespace inference_engine
{
using isize_t = int64_t;

class InferenceEngine
{
public:
    virtual const std::vector<const std::vector<isize_t> &> &get_input_shapes() const = 0;
    virtual const std::vector<const std::vector<isize_t> &> &get_output_shapes() const = 0;

    virtual void run(
        const float *const *input_data,
        const isize_t *const *input_shapes,
        float **output_data,
        const isize_t *const *output_shapes
    ) = 0;
};
} // namespace inference_engine
