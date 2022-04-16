#pragma once

#include <cstdint>
#include <cstdlib>
#include <random>
#include <nlohmann/json.hpp>

namespace soft_npu {

using SizeType = std::size_t;
using ValueType = double;
using TimeType = double;
using RandomEngineType = std::default_random_engine;
using ParamsType = nlohmann::json;

}
