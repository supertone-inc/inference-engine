#include "lib.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <filesystem>
#include <fstream>

std::vector<std::byte> read_file(const std::filesystem::path &file_path)
{
    auto file_size = std::filesystem::file_size(file_path);
    std::vector<std::byte> file(file_size);
    std::ifstream ifs(file_path, std::ios::binary);
    ifs.read(reinterpret_cast<char *>(file.data()), file.size());

    return std::move(file);
}

void unwrap(InferenceEngineResultCode code)
{
    if (code == InferenceEngineResultCode::Error)
    {
        throw std::runtime_error(inference_engine__get_last_error_message());
    }
}

std::vector<std::vector<size_t>> get_input_shapes(const void *engine)
{
    std::vector<std::vector<size_t>> shapes;

    for (auto i = 0; i < inference_engine__get_input_count(engine); i++)
    {
        const size_t *data;
        size_t size;
        inference_engine__get_input_shape(engine, i, &data, &size);
        shapes.push_back({data, data + size});
    }

    return std::move(shapes);
}

std::vector<std::vector<size_t>> get_output_shapes(const void *engine)
{
    std::vector<std::vector<size_t>> shapes;

    for (auto i = 0; i < inference_engine__get_output_count(engine); ++i)
    {
        const size_t *data;
        size_t size;
        inference_engine__get_output_shape(engine, i, &data, &size);
        shapes.push_back({data, data + size});
    }

    return std::move(shapes);
}

void set_input_shapes(void *engine, const std::vector<std::vector<size_t>> &shapes)
{
    for (auto i = 0; i < inference_engine__get_input_count(engine); i++)
    {
        unwrap(inference_engine__set_input_shape(engine, i, shapes[i].data(), shapes[i].size()));
    }
}

struct Engine
{
    void *ptr = nullptr;

    void destroy()
    {
        if (ptr)
        {
            unwrap(inference_engine__destroy_inference_engine(ptr));
            ptr = nullptr;
        }
    }

    ~Engine()
    {
        destroy();
    }
};

TEST_CASE("TfLiteInferenceEngine with invalid model data")
{
    REQUIRE_THROWS_WITH(
        unwrap(inference_engine_tflite__create_inference_engine(nullptr, 0, nullptr)),
        "failed to load model"
    );
}

TEST_CASE("TfLiteInferenceEngine with reshaping inputs")
{
    auto model = read_file("../tflite-cpp/test-models/matmul.tflite");

    Engine engine;
    unwrap(inference_engine_tflite__create_inference_engine(model.data(), model.size(), &engine.ptr));
    REQUIRE(engine.ptr != nullptr);

    REQUIRE(inference_engine__get_input_count(engine.ptr) == 2);
    REQUIRE(inference_engine__get_output_count(engine.ptr) == 1);

    REQUIRE(get_input_shapes(engine.ptr) == std::vector<std::vector<size_t>>{{2, 2}, {2, 2}});
    REQUIRE(get_output_shapes(engine.ptr) == std::vector<std::vector<size_t>>{{2, 2}});

    set_input_shapes(engine.ptr, {{2, 1}, {1, 2}});
    REQUIRE(get_input_shapes(engine.ptr) == std::vector<std::vector<size_t>>{{2, 1}, {1, 2}});
    REQUIRE(get_output_shapes(engine.ptr) == std::vector<std::vector<size_t>>{{2, 2}});

    std::vector<std::vector<float>> inputs{{1, 2}, {3, 4}};
    for (auto i = 0; i < inference_engine__get_input_count(engine.ptr); i++)
    {
        unwrap(inference_engine__set_input_data(engine.ptr, i, inputs[i].data()));
        REQUIRE(inference_engine__get_input_data(engine.ptr, i) == inputs[i].data());
    }

    std::vector<std::vector<float>> outputs{{0, 0, 0, 0}};
    for (auto i = 0; i < inference_engine__get_output_count(engine.ptr); i++)
    {
        unwrap(inference_engine__set_output_data(engine.ptr, i, outputs[i].data()));
        REQUIRE(inference_engine__get_output_data(engine.ptr, i) == outputs[i].data());
    }

    unwrap(inference_engine__run(engine.ptr));
    REQUIRE(outputs == std::vector<std::vector<float>>{{{3, 4, 6, 8}}});

    engine.destroy();
}
