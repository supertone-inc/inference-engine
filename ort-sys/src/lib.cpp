#include "lib.hpp"

#include "inference-engine/OrtInferenceEngine.hpp"

#include <string>

namespace inference_engine
{
namespace sys
{
thread_local std::string last_error_message;

void update_last_error_message(const std::exception &e)
{
    last_error_message = e.what();
}

const char *get_last_error_message()
{
    return last_error_message.c_str();
}

ResultCode create_inference_engine(const void *model_data, size_t model_data_size_bytes, void **engine)
{
    try
    {
        *engine = new OrtInferenceEngine(model_data, model_data_size_bytes);
        return ResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        update_last_error_message(e);
        return ResultCode::Error;
    }
}

ResultCode destroy_inference_engine(void *engine)
{
    try
    {
        delete static_cast<InferenceEngine *>(engine);
        return ResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        update_last_error_message(e);
        return ResultCode::Error;
    }
}

size_t get_input_count(const void *engine)
{
    return static_cast<const InferenceEngine *>(engine)->get_input_count();
}

size_t get_output_count(const void *engine)
{
    return static_cast<const InferenceEngine *>(engine)->get_output_count();
}

void get_input_shape(const void *engine, size_t index, const size_t **shape_data, size_t *shape_size)
{
    const auto &shape = static_cast<const InferenceEngine *>(engine)->get_input_shape(index);
    *shape_data = shape.data();
    *shape_size = shape.size();
}

void get_output_shape(const void *engine, size_t index, const size_t **shape_data, size_t *shape_size)
{
    const auto &shape = static_cast<const InferenceEngine *>(engine)->get_output_shape(index);
    *shape_data = shape.data();
    *shape_size = shape.size();
}

ResultCode set_input_shape(void *engine, size_t index, const size_t *shape_data, size_t shape_size)
{
    try
    {
        static_cast<InferenceEngine *>(engine)->set_input_shape(index, {shape_data, shape_data + shape_size});
        return ResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        update_last_error_message(e);
        return ResultCode::Error;
    }
}

ResultCode set_output_shape(void *engine, size_t index, const size_t *shape_data, size_t shape_size)
{
    try
    {
        static_cast<InferenceEngine *>(engine)->set_output_shape(index, {shape_data, shape_data + shape_size});
        return ResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        update_last_error_message(e);
        return ResultCode::Error;
    }
}

ResultCode set_input_data(void *engine, size_t index, const float *data)
{
    try
    {
        static_cast<InferenceEngine *>(engine)->set_input_data(index, data);
        return ResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        update_last_error_message(e);
        return ResultCode::Error;
    }
}

ResultCode set_output_data(void *engine, size_t index, float *data)
{
    try
    {
        static_cast<InferenceEngine *>(engine)->set_output_data(index, data);
        return ResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        update_last_error_message(e);
        return ResultCode::Error;
    }
}

ResultCode run(void *engine)
{
    try
    {
        static_cast<InferenceEngine *>(engine)->run();
        return ResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        update_last_error_message(e);
        return ResultCode::Error;
    }
}

} // namespace sys
} // namespace inference_engine