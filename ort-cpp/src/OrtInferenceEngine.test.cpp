#include "inference_engine/OrtInferenceEngine.hpp"

#include <catch2/catch_test_macros.hpp>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

std::vector<std::byte> read_file(const std::filesystem::path &file_path)
{
    auto file_size = std::filesystem::file_size(file_path);
    std::vector<std::byte> file(file_size);
    std::ifstream ifs(file_path, std::ios::binary);
    ifs.read(reinterpret_cast<char *>(file.data()), file.size());

    return std::move(file);
}

using namespace inference_engine;

TEST_CASE("OrtInferenceEngine with fixed-shape model")
{
    auto model = read_file("test-models/matmul.onnx");
    auto engine = OrtInferenceEngine(model.data(), model.size());

    REQUIRE(engine.get_input_count() == 2);
    REQUIRE(engine.get_output_count() == 1);

    REQUIRE(engine.get_input_shape(0) == std::vector<size_t>{2, 2});
    REQUIRE(engine.get_input_shape(1) == std::vector<size_t>{2, 2});
    REQUIRE(engine.get_output_shape(0) == std::vector<size_t>{2, 2});

    std::vector<std::vector<float>> inputs{{1, 2, 3, 4}, {5, 6, 7, 8}};
    for (auto i = 0; i < engine.get_input_count(); i++)
    {
        engine.set_input_data(i, inputs[i].data());
        REQUIRE(engine.get_input_data(i) == inputs[i].data());
    }

    std::vector<std::vector<float>> outputs{{0, 0, 0, 0}};
    for (auto i = 0; i < engine.get_output_count(); i++)
    {
        engine.set_output_data(i, outputs[i].data());
        REQUIRE(engine.get_output_data(i) == outputs[i].data());
    }

    engine.run();

    REQUIRE(outputs == std::vector<std::vector<float>>{{{19, 22, 43, 50}}});
}

TEST_CASE("OrtInferenceEngine with dynamic-shape model")
{
    auto model = read_file("test-models/matmul_dynamic.onnx");
    auto engine = OrtInferenceEngine(model.data(), model.size());

    REQUIRE(engine.get_input_count() == 2);
    REQUIRE(engine.get_output_count() == 1);

    REQUIRE(engine.get_input_shape(0) == std::vector<size_t>{0, 0});
    REQUIRE(engine.get_input_shape(1) == std::vector<size_t>{0, 0});
    REQUIRE(engine.get_output_shape(0) == std::vector<size_t>{0, 0});

    engine.set_input_shape(0, {2, 1});
    engine.set_input_shape(1, {1, 2});
    REQUIRE(engine.get_input_shape(0) == std::vector<size_t>{2, 1});
    REQUIRE(engine.get_input_shape(1) == std::vector<size_t>{1, 2});

    engine.set_output_shape(0, {2, 2});
    REQUIRE(engine.get_output_shape(0) == std::vector<size_t>{2, 2});

    std::vector<std::vector<float>> inputs{{1, 2}, {3, 4}};
    for (auto i = 0; i < engine.get_input_count(); i++)
    {
        engine.set_input_data(i, inputs[i].data());
        REQUIRE(engine.get_input_data(i) == inputs[i].data());
    }

    std::vector<std::vector<float>> outputs{{0, 0, 0, 0}};
    for (auto i = 0; i < engine.get_output_count(); i++)
    {
        engine.set_output_data(i, outputs[i].data());
        REQUIRE(engine.get_output_data(i) == outputs[i].data());
    }

    engine.run();
    REQUIRE(outputs == std::vector<std::vector<float>>{{{3, 4, 6, 8}}});
}
