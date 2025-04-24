# 设置 CMAKE_BUILD_TYPE 为 Debug
# set(CMAKE_BUILD_TYPE Debug)

# 显式关闭优化
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

add_executable(Embedding_Test Embedding_Test.cpp)

add_executable(E2E_test E2E_test.cpp ../test.h ../utils.h ../kvstore_api.h ../MurmurHash3.h
        ../kvstore.h ../kvstore.cc ../skiplist.cpp ../skiplist.h ../sstable.cpp
        ../sstable.h ../bloom.cpp ../bloom.h ../sstablehead.cpp ../sstablehead.h
        ../hnsw.cpp ../hnsw.h)

target_link_libraries(Embedding_Test PUBLIC embedding)

target_link_libraries(E2E_test PUBLIC embedding)
