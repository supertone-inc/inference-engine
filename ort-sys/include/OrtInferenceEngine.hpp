#pragma once

#include "Result.hpp"

#include <inference-engine/OrtInferenceEngine.hpp>
#include <memory>

struct OrtInferenceEngine
{
    std::shared_ptr<inference_engine::OrtInferenceEngine> source;

    static Result<OrtInferenceEngine> create(const void *model_data, size_t model_data_size_bytes);
};
