#pragma once

#include "inference-engine/InferenceEngine.hpp"

#include <memory>
#include <vector>

namespace inference_engine
{
class OrtInferenceEngine : public InferenceEngine
{
public:
    OrtInferenceEngine(const void *model_data, size_t model_data_size_bytes);

    size_t get_input_count() const override;
    const std::vector<int64_t> &get_input_shape(size_t index) const override;
    void set_input_shape(size_t index, const std::vector<int64_t> &shape) override;
    void set_input_data(size_t index, const float *data) override;

    size_t get_output_count() const override;
    const std::vector<int64_t> &get_output_shape(size_t index) const override;
    void set_output_shape(size_t index, const std::vector<int64_t> &shape) override;
    void set_output_data(size_t index, float *data) override;

    void run() override;

private:
    class Impl;
    std::shared_ptr<Impl> impl;
};
} // namespace inference_engine
