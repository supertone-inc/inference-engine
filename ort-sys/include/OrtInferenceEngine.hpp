#pragma once

#include "Result.hpp"

#include <cstddef>

class OrtInferenceEngine
{
public:
    static Result<OrtInferenceEngine> create(const void *model_data, size_t model_data_size_bytes);

    OrtInferenceEngine();

    OrtInferenceEngine(OrtInferenceEngine &&);
    OrtInferenceEngine &operator=(OrtInferenceEngine &&);

    OrtInferenceEngine(const OrtInferenceEngine &) = delete;
    OrtInferenceEngine &operator=(const OrtInferenceEngine &) = delete;

    virtual ~OrtInferenceEngine();

private:
    class Impl;
    Impl *impl = nullptr;

    OrtInferenceEngine(const void *model_data, size_t model_data_size_bytes);
};
