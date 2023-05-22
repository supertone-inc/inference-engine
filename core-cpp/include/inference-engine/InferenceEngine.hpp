#pragma once

#include <cstdint>
#include <vector>

namespace inference_engine
{
class InferenceEngine
{
public:
    virtual size_t get_input_count() const = 0;
    virtual const std::vector<int64_t> &get_input_shape(size_t index) const = 0;
    virtual void set_input_shape(size_t index, const std::vector<int64_t> &shape) = 0;
    virtual void set_input_data(size_t index, const float *data) = 0;

    virtual size_t get_output_count() const = 0;
    virtual const std::vector<int64_t> &get_output_shape(size_t index) const = 0;
    virtual void set_output_shape(size_t index, const std::vector<int64_t> &shape) = 0;
    virtual void set_output_data(size_t index, float *data) = 0;

    virtual void run() = 0;
};
} // namespace inference_engine
