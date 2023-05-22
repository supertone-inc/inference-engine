#include "OrtInferenceEngine.hpp"

#include <inference-engine/OrtInferenceEngine.hpp>
#include <memory>

namespace base = inference_engine;

class OrtInferenceEngine::Impl
{
public:
    Impl(const void *model_data, size_t model_data_size_bytes)
        : ptr(new base::OrtInferenceEngine{model_data, model_data_size_bytes})
    {
    }

    const base::OrtInferenceEngine &operator*() const
    {
        return *ptr;
    }

    base::OrtInferenceEngine &operator*()
    {
        return *ptr;
    }

private:
    std::shared_ptr<base::OrtInferenceEngine> ptr;
};

Result<OrtInferenceEngine> OrtInferenceEngine::create(const void *model_data, size_t model_data_size_bytes)
{
    try
    {
        return {ResultCode::Ok, {model_data, model_data_size_bytes}};
    }
    catch (const std::exception &e)
    {
        return {ResultCode::Error, {}, e.what()};
    }
}

OrtInferenceEngine::OrtInferenceEngine(const void *model_data, size_t model_data_size_bytes)
    : impl(new Impl{model_data, model_data_size_bytes})
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

size_t OrtInferenceEngine::get_input_count() const
{
    return (**impl).get_input_count();
}

Array<size_t> OrtInferenceEngine::get_input_shape(size_t index) const
{
    auto &shape = (**impl).get_input_shape(index);
    return {shape.data(), shape.size()};
}

size_t OrtInferenceEngine::get_output_count() const
{
    return (**impl).get_output_count();
}

Array<size_t> OrtInferenceEngine::get_output_shape(size_t index) const
{
    auto &shape = (**impl).get_output_shape(index);
    return {shape.data(), shape.size()};
}
