#pragma once

#include "inference_engine/InferenceEngine.hpp"

#include <memory>
#include <vector>

namespace inference_engine
{
class TfLiteInferenceEngine : public InferenceEngine
{
public:
    TfLiteInferenceEngine(const void *model_data, size_t model_data_size_bytes);

    size_t get_input_count() const override;
    size_t get_output_count() const override;

    const std::vector<size_t> &get_input_shape(size_t index) const override;
    const std::vector<size_t> &get_output_shape(size_t index) const override;

    void set_input_shape(size_t index, const std::vector<size_t> &shape) override;
    void set_output_shape(size_t index, const std::vector<size_t> &shape) override;

    float *get_input_data(size_t index) override;
    const float *get_output_data(size_t index) const override;

    void set_input_data(size_t index, const float *data) override;
    void set_output_data(size_t index, float *data) override;

    void run() override;

private:
    class Impl;
    std::shared_ptr<Impl> impl;
};
} // namespace inference_engine
