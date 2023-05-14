#pragma once

#include <vector>

namespace inference_engine
{
class TensorInfo
{
public:
    TensorInfo(const std::vector<int64_t> &shape)
        : shape(shape)
        , element_count(get_element_count(shape))
    {
    }

    bool operator==(const TensorInfo &other) const
    {
        return other.shape == shape && other.element_count == element_count;
    }

    const std::vector<int64_t> shape;
    const int64_t element_count;

private:
    inline static int64_t get_element_count(const std::vector<int64_t> &shape)
    {
        if (shape.empty())
        {
            return 0;
        }

        auto element_count = 1;

        for (auto v : shape)
        {
            if (v < 0)
            {
                return -1;
            }

            element_count *= v;
        }

        return element_count;
    }
};
} // namespace inference_engine
