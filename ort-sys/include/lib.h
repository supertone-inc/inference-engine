#pragma once

#include <lib_core.h>

#ifdef __cplusplus
extern "C"
{
#endif
    InferenceEngineResultCode inference_engine_ort__create_inference_engine(const void *model_data, size_t model_data_size_bytes, void **engine);
#ifdef __cplusplus
}
#endif
