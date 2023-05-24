#pragma onece

#include <cstddef>

namespace inference_engine
{
namespace sys
{
enum class ResultCode
{
    Ok = 0,
    Error = -1,
};

const char *get_last_error_message();

ResultCode create_inference_engine(const void *model_data, size_t model_data_size_bytes, void **engine);
ResultCode destroy_inference_engine(void *engine);

size_t get_input_count(const void *engine);
void get_input_shape(const void *engine, size_t index, const size_t **shape_data, size_t *shape_size);
ResultCode set_input_shape(void *engine, size_t index, const size_t *shape_data, size_t shape_size);
ResultCode set_input_data(void *engine, size_t index, const float *data);

size_t get_output_count(const void *engine);
void get_output_shape(const void *engine, size_t index, const size_t **shape_data, size_t *shape_size);
ResultCode set_output_shape(void *engine, size_t index, const size_t *shape_data, size_t shape_size);
ResultCode set_output_data(void *engine, size_t index, float *data);

ResultCode run(void *engine);
} // namespace sys
} // namespace inference_engine
