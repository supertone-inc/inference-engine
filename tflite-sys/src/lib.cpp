#include "lib.h"

#include <inference_engine/TfliteInferenceEngine.hpp>

InferenceEngineResultCode inference_engine_tflite__create_inference_engine(const void *model_data, size_t model_data_size_bytes, void **engine)
{
    try
    {
        *engine = new inference_engine::TfliteInferenceEngine(model_data, model_data_size_bytes);
        return InferenceEngineResultCode::Ok;
    }
    catch (const std::exception &e)
    {
        inference_engine__update_last_error_message(e.what());
        return InferenceEngineResultCode::Error;
    }
}
