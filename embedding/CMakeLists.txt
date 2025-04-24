# 设置 CMAKE_BUILD_TYPE 为 Debug
# set(CMAKE_BUILD_TYPE Debug)

# 显式关闭优化
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

add_library(embedding STATIC embedding.cc)

target_link_libraries(embedding PUBLIC llama common)

target_include_directories(embedding PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
