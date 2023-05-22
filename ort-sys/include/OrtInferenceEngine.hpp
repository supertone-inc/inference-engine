#pragma once

#include "Array.hpp"
#include "Result.hpp"

#include <cstddef>

class OrtInferenceEngine
{
public:
    static Result<OrtInferenceEngine> create(const void *model_data, size_t model_data_size_bytes);

    OrtInferenceEngine(OrtInferenceEngine &&);
    OrtInferenceEngine &operator=(OrtInferenceEngine &&);

    OrtInferenceEngine(const OrtInferenceEngine &) = delete;
    OrtInferenceEngine &operator=(const OrtInferenceEngine &) = delete;

    virtual ~OrtInferenceEngine();

    size_t get_input_count() const;
    Array<size_t> get_input_shape(size_t index) const;
    Result<void *> set_input_shape(size_t index, const size_t *data, size_t size);
    Result<void *> set_input_data(size_t index, const float *data);

    size_t get_output_count() const;
    Array<size_t> get_output_shape(size_t index) const;
    Result<void *> set_output_shape(size_t index, const size_t *data, size_t size);
    Result<void *> set_output_data(size_t index, float *data);

    Result<void *> run();

private:
    class Impl;
    Impl *impl = nullptr;

    OrtInferenceEngine() = default;
    OrtInferenceEngine(const void *model_data, size_t model_data_size_bytes);
};
