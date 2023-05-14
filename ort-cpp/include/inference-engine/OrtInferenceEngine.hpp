#pragma once

#include "inference-engine/InferenceEngine.hpp"

#include <memory>
#include <vector>

namespace inference_engine
{
class OrtInferenceEngine : public InferenceEngine
{
public:
    OrtInferenceEngine(const std::byte *model_data, size_t model_data_size);

    const std::vector<TensorInfo> &get_input_info() const override;

    const std::vector<TensorInfo> &get_output_info() const override;

    void run(
        const float *const *input_data,
        float **output_data,
        const int64_t *const *input_shapes = nullptr,
        const int64_t *const *output_shapes = nullptr
    ) override;

private:
    class Impl;
    std::shared_ptr<Impl> impl;
};
} // namespace inference_engine
