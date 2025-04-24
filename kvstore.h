#pragma once

#include "kvstore_api.h"
#include "skiplist.h"
#include "sstable.h"
#include "sstablehead.h"
#include "hnsw.h"

#include <map>
#include <set>

class KVStore : public KVStoreAPI {
    // You can add your implementation here
private:
    skiplist *s = new skiplist(0.5); // memtable
    // std::vector<sstablehead> sstableIndex;  // sstable的表头缓存

    std::vector<sstablehead> sstableIndex[15]; // the sshead for each level

    std::vector<std::vector<std::vector<float>>> vecs[15]; // embedding for each value

    int totalLevel = -1; // 层数

    HNSW hnsw;
public:
    KVStore(const std::string &dir);

    ~KVStore();

    void put(uint64_t key, const std::string &s) override;

    std::string get(uint64_t key) override;

    bool del(uint64_t key) override;

    void reset() override;

    void scan(uint64_t key1, uint64_t key2, std::list<std::pair<uint64_t, std::string>> &list) override;
    
    std::vector<std::pair<uint64_t, std::string>> search_knn(std::string query, int k) override;

    void build_hnsw(int M, int M_max, int efConstruction, int m_L);

    std::vector<std::pair<std::uint64_t, std::string>> search_knn_hnsw(std::string query, int k) override;

    void compaction();

    void delsstable(std::string filename);  // 从缓存中删除filename.sst， 并物理删除
    void addsstable(sstable ss, int level); // 将ss加入缓存

    std::string fetchString(std::string file, int startOffset, uint32_t len);
};
