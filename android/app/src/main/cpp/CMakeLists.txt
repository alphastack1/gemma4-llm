cmake_minimum_required(VERSION 3.22.1)
project(gemma4-llm)

# Stock llama.cpp from ggml-org (no special fork needed for Gemma 4)
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp")

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "llama.cpp not found at ${LLAMA_CPP_DIR}.\n"
        "Run: git clone --depth 1 --branch b8683 https://github.com/ggml-org/llama.cpp.git ${LLAMA_CPP_DIR}"
    )
endif()

# Build llama.cpp as a subdirectory
add_subdirectory(${LLAMA_CPP_DIR} llama.cpp)

# Android trick: rename executable to libllama_server.so
# so the package manager extracts it to the native lib directory
if(TARGET llama-server)
    set_target_properties(llama-server PROPERTIES
        OUTPUT_NAME "llama_server"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}"
    )
    set_target_properties(llama-server PROPERTIES SUFFIX ".so")
    set_target_properties(llama-server PROPERTIES PREFIX "lib")
endif()
