#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif
    typedef enum
    {
        Ok = 0,
        Error = -1,
    } InferenceEngineResultCode;

    const char *inference_engine__get_last_error_message();

    InferenceEngineResultCode inference_engine_tflite__create_inference_engine(const void *model_data, size_t model_data_size_bytes, void **engine);
    InferenceEngineResultCode inference_engine__destroy_inference_engine(void *engine);

    size_t inference_engine__get_input_count(const void *engine);
    size_t inference_engine__get_output_count(const void *engine);

    void inference_engine__get_input_shape(const void *engine, size_t index, const size_t **shape_data, size_t *shape_size);
    void inference_engine__get_output_shape(const void *engine, size_t index, const size_t **shape_data, size_t *shape_size);

    InferenceEngineResultCode inference_engine__set_input_shape(void *engine, size_t index, const size_t *shape_data, size_t shape_size);
    InferenceEngineResultCode inference_engine__set_output_shape(void *engine, size_t index, const size_t *shape_data, size_t shape_size);

    float *inference_engine__get_input_data(void *engine, size_t index);
    const float *inference_engine__get_output_data(const void *engine, size_t index);

    InferenceEngineResultCode inference_engine__set_input_data(void *engine, size_t index, const float *data);
    InferenceEngineResultCode inference_engine__set_output_data(void *engine, size_t index, float *data);

    InferenceEngineResultCode inference_engine__run(void *engine);
#ifdef __cplusplus
}
#endif
