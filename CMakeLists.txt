cmake_minimum_required(VERSION 3.10)
project(LSM_TREE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置 CMAKE_BUILD_TYPE 为 Debug
# set(CMAKE_BUILD_TYPE Debug)

# 显式关闭优化
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

add_executable(correctness correctness.cc kvstore_api.h kvstore.h
        kvstore.cc skiplist.cpp skiplist.h sstable.cpp sstable.h
        bloom.cpp bloom.h MurmurHash3.h utils.h test.h
        sstablehead.cpp sstablehead.h hnsw.cpp hnsw.h)

add_executable(persistence persistence.cc kvstore_api.h kvstore.h kvstore.cc
        skiplist.cpp skiplist.h sstable.cpp sstable.h
        bloom.cpp bloom.h MurmurHash3.h utils.h test.h
        sstablehead.cpp sstablehead.h hnsw.cpp hnsw.h)

add_executable(Vector_Persistent_Test_Phase1 Vector_Persistent_Test_Phase1.cpp
        kvstore_api.h kvstore.h kvstore.cc
        skiplist.cpp skiplist.h sstable.cpp sstable.h
        bloom.cpp bloom.h MurmurHash3.h utils.h test.h
        sstablehead.cpp sstablehead.h hnsw.cpp hnsw.h)

add_executable(Vector_Persistent_Test_Phase2 Vector_Persistent_Test_Phase2.cpp
        kvstore_api.h kvstore.h kvstore.cc
        skiplist.cpp skiplist.h sstable.cpp sstable.h
        bloom.cpp bloom.h MurmurHash3.h utils.h test.h
        sstablehead.cpp sstablehead.h hnsw.cpp hnsw.h)

add_executable(HNSW_Delete_Test HNSW_Delete_Test.cpp
        kvstore_api.h kvstore.h kvstore.cc
        skiplist.cpp skiplist.h sstable.cpp sstable.h
        bloom.cpp bloom.h MurmurHash3.h utils.h test.h
        sstablehead.cpp sstablehead.h hnsw.cpp hnsw.h)

add_executable(HNSW_Persistent_Test_Phase1 HNSW_Persistent_Test_Phase1.cpp
        kvstore_api.h kvstore.h kvstore.cc
        skiplist.cpp skiplist.h sstable.cpp sstable.h
        bloom.cpp bloom.h MurmurHash3.h utils.h test.h
        sstablehead.cpp sstablehead.h hnsw.cpp hnsw.h)

add_executable(HNSW_Persistent_Test_Phase2 HNSW_Persistent_Test_Phase2.cpp
        kvstore_api.h kvstore.h kvstore.cc
        skiplist.cpp skiplist.h sstable.cpp sstable.h
        bloom.cpp bloom.h MurmurHash3.h utils.h test.h
        sstablehead.cpp sstablehead.h hnsw.cpp hnsw.h)


target_link_libraries(correctness PUBLIC embedding)
target_link_libraries(persistence PUBLIC embedding)
target_link_libraries(Vector_Persistent_Test_Phase1 PUBLIC embedding)
target_link_libraries(Vector_Persistent_Test_Phase2 PUBLIC embedding)
target_link_libraries(HNSW_Delete_Test PUBLIC embedding)
target_link_libraries(HNSW_Persistent_Test_Phase1 PUBLIC embedding)
target_link_libraries(HNSW_Persistent_Test_Phase2 PUBLIC embedding)

add_subdirectory(third_party/llama.cpp)

add_subdirectory(embedding)

add_subdirectory(test)
