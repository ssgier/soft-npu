include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

macro(add_test _test_name _source_file)
    add_executable(
            ${_test_name}
            ${_source_file})
    target_include_directories(
            ${_test_name}
            PRIVATE
            ../src json_INCLUDE_DIR
            ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(
            ${_test_name}
            gtest_main
            soft_npu
            nlohmann_json::nlohmann_json
    )


    target_include_directories(${_test_name} PRIVATE )

    gtest_discover_tests(${_test_name})
endmacro()

add_test(batched_ring_buffer_test BatchedRingBufferTest.cpp)
add_test(env_events_test EnvEventsTest.cpp)
add_test(basic_integration_tests integration_tests/BasicIntegrationTests.cpp)
add_test(stdp_integration_tests integration_tests/STDPIntegrationTests.cpp)
add_test(short_term_plasticity_test integration_tests/ShortTermPlasticityTest.cpp)
add_test(da_modulation_integration_tests integration_tests/DAModulationIntegrationTests.cpp)
add_test(continuous_inhibition_test integration_tests/ContinuousInhibitionTest.cpp)
add_test(population_generator_tests PopulationGeneratorTests.cpp)
add_test(population_generator_evo_test PopulationGeneratorEvoTest.cpp)
add_test(population_test PopulationTest.cpp)
add_test(topographic_channel_projector_test TopographicChannelProjectorTest.cpp)
add_test(gene_operation_utils_test GeneOperationUtilsTest.cpp)
add_test(selection_utils_test SelectionUtilsTest.cpp)
add_test(gene_test GeneTest.cpp)
add_test(evolution_test EvolutionTest.cpp)
