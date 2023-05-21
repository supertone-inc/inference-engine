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

    virtual const std::vector<std::vector<int64_t>> &get_output_shapes() const = 0;
    virtual void set_output_shapes(const std::vector<std::vector<int64_t>> &shapes) = 0;

    virtual void run(const float *const *input_values, float **output_values) = 0;
};

inline int64_t get_element_count(const int64_t *shape, size_t size)
{
    if (!shape || !size)
    {
        return 0;
    }

    auto element_count = 1;

    for (auto i = 0; i < size; i++)
    {
        if (shape[i] < 0)
        {
            return -1;
        }

        element_count *= shape[i];
    }

    return element_count;
}

inline int64_t get_element_count(const std::vector<int64_t> &shape)
{
    return get_element_count(shape.data(), shape.size());
}
} // namespace inference_engine
