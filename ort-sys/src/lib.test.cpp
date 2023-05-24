#include "lib.hpp"

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

using namespace inference_engine::sys;

void unwrap(ResultCode code)
{
    if (code == ResultCode::Error)
    {
        throw std::runtime_error(get_last_error_message());
    }
}

std::vector<std::vector<size_t>> get_input_shapes(const void *engine)
{
    std::vector<std::vector<size_t>> shapes;

    for (auto i = 0; i < get_input_count(engine); i++)
    {
        const size_t *data;
        size_t size;
        get_input_shape(engine, i, &data, &size);
        shapes.push_back({data, data + size});
    }

    return std::move(shapes);
}

void set_input_shapes(void *engine, const std::vector<std::vector<size_t>> &shapes)
{
    for (auto i = 0; i < get_input_count(engine); i++)
    {
        unwrap(set_input_shape(engine, i, shapes[i].data(), shapes[i].size()));
    }
}

void set_input_data(void *engine, const std::vector<std::vector<float>> &data)
{
    for (auto i = 0; i < get_input_count(engine); i++)
    {
        unwrap(set_input_data(engine, i, data[i].data()));
    }
}

std::vector<std::vector<size_t>> get_output_shapes(const void *engine)
{
    std::vector<std::vector<size_t>> shapes;

    for (auto i = 0; i < get_output_count(engine); ++i)
    {
        const size_t *data;
        size_t size;
        get_output_shape(engine, i, &data, &size);
        shapes.push_back({data, data + size});
    }

    return std::move(shapes);
}

void set_output_shapes(void *engine, const std::vector<std::vector<size_t>> &shapes)
{
    for (auto i = 0; i < get_output_count(engine); ++i)
    {
        unwrap(set_output_shape(engine, i, shapes[i].data(), shapes[i].size()));
    }
}

void set_output_data(void *engine, std::vector<std::vector<float>> &data)
{
    for (auto i = 0; i < get_output_count(engine); ++i)
    {
        unwrap(set_output_data(engine, i, data[i].data()));
    }
}

struct Engine
{
    void *ptr = nullptr;

    void destroy()
    {
        if (ptr)
        {
            unwrap(destroy_inference_engine(ptr));
            ptr = nullptr;
        }
    }

    ~Engine()
    {
        destroy();
    }
};

TEST_CASE("OrtInferenceEngine with invalid model data")
{
    REQUIRE_THROWS_WITH(
        unwrap(create_inference_engine(nullptr, 0, nullptr)),
        "No graph was found in the protobuf."
    );
}

TEST_CASE("OrtInferenceEngine with dynamic-shape model")
{
    auto model = read_file("../ort-cpp/test-models/matmul_dynamic.onnx");

    Engine engine;
    unwrap(create_inference_engine(model.data(), model.size(), &engine.ptr));
    REQUIRE(engine.ptr != nullptr);

    REQUIRE(get_input_count(engine.ptr) == 2);
    REQUIRE(get_output_count(engine.ptr) == 1);
    REQUIRE(get_input_shapes(engine.ptr) == std::vector<std::vector<size_t>>{{0, 0}, {0, 0}});
    REQUIRE(get_output_shapes(engine.ptr) == std::vector<std::vector<size_t>>{{0, 0}});

    set_input_shapes(engine.ptr, {{2, 1}, {1, 2}});
    REQUIRE(get_input_shapes(engine.ptr) == std::vector<std::vector<size_t>>{{2, 1}, {1, 2}});

    set_output_shapes(engine.ptr, {{2, 2}});
    REQUIRE(get_output_shapes(engine.ptr) == std::vector<std::vector<size_t>>{{2, 2}});

    std::vector<std::vector<float>> inputs{{1, 2}, {3, 4}};
    set_input_data(engine.ptr, inputs);

    std::vector<std::vector<float>> outputs{{0, 0, 0, 0}};
    set_output_data(engine.ptr, outputs);

    unwrap(run(engine.ptr));
    REQUIRE(outputs == std::vector<std::vector<float>>{{{3, 4, 6, 8}}});

    engine.destroy();
}
