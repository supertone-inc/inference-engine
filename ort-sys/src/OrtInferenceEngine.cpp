#include "OrtInferenceEngine.hpp"

#include <inference-engine/OrtInferenceEngine.hpp>
#include <memory>

namespace base = inference_engine;

class OrtInferenceEngine::Impl
{
public:
    Impl(const void *model_data, size_t model_data_size_bytes)
        : ptr(new base::OrtInferenceEngine(model_data, model_data_size_bytes))
    {
    }

    std::shared_ptr<base::OrtInferenceEngine> ptr;
};

template <>
Result<OrtInferenceEngine> ok<OrtInferenceEngine>(OrtInferenceEngine &&value)
{
    return {ResultCode::Ok, std::move(value), {}};
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
        return ok<OrtInferenceEngine>({model_data, model_data_size_bytes});
    }
    catch (const std::exception &e)
    {
        return err<OrtInferenceEngine>({e.what()});
    }
}

OrtInferenceEngine::OrtInferenceEngine()
    : impl(nullptr)
{
}

OrtInferenceEngine::OrtInferenceEngine(const void *model_data, size_t model_data_size_bytes)
    : impl(new Impl(model_data, model_data_size_bytes))
{
}

OrtInferenceEngine::OrtInferenceEngine(OrtInferenceEngine &&other)
{
    *this = std::move(other);
}

OrtInferenceEngine &OrtInferenceEngine::operator=(OrtInferenceEngine &&other)
{
    if (&other != this)
    {
        if (impl)
        {
            delete impl;
        }

        impl = other.impl;
        other.impl = nullptr;
    }

    return *this;
}

OrtInferenceEngine::~OrtInferenceEngine()
{
    if (impl)
    {
        delete impl;
    }
}
