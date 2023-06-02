#include "lib.h"

#include "inference_engine/TfliteInferenceEngine.hpp"

#include <string>

using namespace inference_engine;

thread_local std::string last_error_message;

void update_last_error_message(const std::exception &e)
{
    last_error_message = e.what();
}

const char *inference_engine__get_last_error_message()
{
    return last_error_message.c_str();
}

InferenceEngineResultCode inference_engine__create_inference_engine(const void *model_data, size_t model_data_size_bytes, void **engine)
{
    try
    {
        *engine = new TfliteInferenceEngine(model_data, model_data_size_bytes);
        return InferenceEngineResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        update_last_error_message(e);
        return InferenceEngineResultCode::Error;
    }
}

InferenceEngineResultCode inference_engine__destroy_inference_engine(void *engine)
{
    try
    {
        delete static_cast<InferenceEngine *>(engine);
        return InferenceEngineResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        update_last_error_message(e);
        return InferenceEngineResultCode::Error;
    }
}

size_t inference_engine__get_input_count(const void *engine)
{
    return static_cast<const InferenceEngine *>(engine)->get_input_count();
}

size_t inference_engine__get_output_count(const void *engine)
{
    return static_cast<const InferenceEngine *>(engine)->get_output_count();
}

void inference_engine__get_input_shape(const void *engine, size_t index, const size_t **shape_data, size_t *shape_size)
{
    const auto &shape = static_cast<const InferenceEngine *>(engine)->get_input_shape(index);
    *shape_data = shape.data();
    *shape_size = shape.size();
}

void inference_engine__get_output_shape(const void *engine, size_t index, const size_t **shape_data, size_t *shape_size)
{
    const auto &shape = static_cast<const InferenceEngine *>(engine)->get_output_shape(index);
    *shape_data = shape.data();
    *shape_size = shape.size();
}

InferenceEngineResultCode inference_engine__set_input_shape(void *engine, size_t index, const size_t *shape_data, size_t shape_size)
{
    try
    {
        static_cast<InferenceEngine *>(engine)->set_input_shape(index, {shape_data, shape_data + shape_size});
        return InferenceEngineResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        update_last_error_message(e);
        return InferenceEngineResultCode::Error;
    }
}

InferenceEngineResultCode inference_engine__set_output_shape(void *engine, size_t index, const size_t *shape_data, size_t shape_size)
{
    try
    {
        static_cast<InferenceEngine *>(engine)->set_output_shape(index, {shape_data, shape_data + shape_size});
        return InferenceEngineResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        update_last_error_message(e);
        return InferenceEngineResultCode::Error;
    }
}

float *inference_engine__get_input_data(void *engine, size_t index)
{
    return static_cast<InferenceEngine *>(engine)->get_input_data(index);
}

const float *inference_engine__get_output_data(const void *engine, size_t index)
{
    return static_cast<const InferenceEngine *>(engine)->get_output_data(index);
}

InferenceEngineResultCode inference_engine__set_input_data(void *engine, size_t index, const float *data)
{
    try
    {
        static_cast<InferenceEngine *>(engine)->set_input_data(index, data);
        return InferenceEngineResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        update_last_error_message(e);
        return InferenceEngineResultCode::Error;
    }
}

InferenceEngineResultCode inference_engine__set_output_data(void *engine, size_t index, float *data)
{
    try
    {
        static_cast<InferenceEngine *>(engine)->set_output_data(index, data);
        return InferenceEngineResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        update_last_error_message(e);
        return InferenceEngineResultCode::Error;
    }
}

InferenceEngineResultCode inference_engine__run(void *engine)
{
    try
    {
        static_cast<InferenceEngine *>(engine)->run();
        return InferenceEngineResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        update_last_error_message(e);
        return InferenceEngineResultCode::Error;
    }
}
