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

    const std::vector<std::vector<int64_t>> &get_input_shapes() const override;
    void set_input_shapes(const std::vector<std::vector<int64_t>> &shapes) override;

    const std::vector<std::vector<int64_t>> &get_output_shapes() const override;
    void set_output_shapes(const std::vector<std::vector<int64_t>> &shapes) override;

    void set_input_data(const float *const *data) override;
    void set_output_data(float **data) override;

    void run() override;

private:
    class Impl;
    std::shared_ptr<Impl> impl;
};
} // namespace inference_engine
