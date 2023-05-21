#pragma once

#include <cstdint>
#include <vector>

namespace inference_engine
{
class InferenceEngine
{
public:
    virtual const std::vector<std::vector<int64_t>> &get_input_shapes() const = 0;
    virtual void set_input_shapes(const std::vector<std::vector<int64_t>> &shapes) = 0;
    virtual void set_input_data(const float *const *data) = 0;

    virtual const std::vector<std::vector<int64_t>> &get_output_shapes() const = 0;
    virtual void set_output_shapes(const std::vector<std::vector<int64_t>> &shapes) = 0;
    virtual void set_output_data(float **data) = 0;

    virtual void run() = 0;
};
} // namespace inference_engine
