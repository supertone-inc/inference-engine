#include "OrtInferenceEngine.hpp"

namespace base = inference_engine;

template <>
Result<OrtInferenceEngine> ok<OrtInferenceEngine>(OrtInferenceEngine &&value)
{
    return {ResultCode::Ok, value, {}};
}

template <>
Result<OrtInferenceEngine> err<OrtInferenceEngine>(Error &&value)
{
    return {ResultCode::Error, {}, std::move(value)};
}

Result<OrtInferenceEngine> OrtInferenceEngine::create(const void *model_data, size_t model_data_size_bytes)
{
    try
    {
        return ok(
            OrtInferenceEngine{std::shared_ptr<base::OrtInferenceEngine>{
                new base::OrtInferenceEngine{
                    model_data,
                    model_data_size_bytes,
                },
            }}
        );
    }
    catch (const std::exception &e)
    {
        return err<OrtInferenceEngine>({e.what()});
    }
}
