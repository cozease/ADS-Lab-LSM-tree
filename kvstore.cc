#include "kvstore.h"

#include "skiplist.h"
#include "sstable.h"
#include "utils.h"
#include "embedding.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <chrono>

static const std::string DEL = "~DELETED~";
const uint32_t MAXSIZE       = 2 * 1024 * 1024;

struct poi {
    int sstableId; // vector中第几个sstable
    int pos;       // 该sstable的第几个key-offset
    uint64_t time;
    Index index;
};

struct cmpPoi {
    bool operator()(const poi &a, const poi &b) {
        if (a.index.key == b.index.key)
            return a.time < b.time;
        return a.index.key > b.index.key;
    }
};

KVStore::KVStore(const std::string &dir) :
    KVStoreAPI(dir) // read from sstables
{
    for (totalLevel = 0;; ++totalLevel) {
        std::string path = dir + "/level-" + std::to_string(totalLevel) + "/";
        std::vector<std::string> files;
        if (!utils::dirExists(path)) {
            totalLevel--;
            break; // stop read
        }
        int nums = utils::scanDir(path, files);
        sstable cur;
        for (int i = 0; i < nums; ++i) {       // 读每一个文件头
            std::string url = path + files[i]; // url, 每一个文件名
            cur.loadFile(url.data());
            sstableIndex[totalLevel].push_back(cur.getHead());
            TIME = std::max(TIME, cur.getTime()); // 更新时间戳

            // 读取每个文件里的所有value并存入embeddings中
            std::vector<std::vector<float>> file_vecs;
            for (int j = 0; j < cur.getCnt(); ++j) {
                std::string val = cur.getData(j);
                file_vecs.push_back(embedding_single(val));
            }
            vecs[totalLevel].push_back(file_vecs);
        }
    }
}

KVStore::~KVStore()
{
    sstable ss(s);
    if (!ss.getCnt())
        return; // empty sstable
    std::string path = std::string("./data/level-0/");
    if (!utils::dirExists(path)) {
        utils::_mkdir(path.data());
        totalLevel = 0;
    }
    ss.putFile(ss.getFilename().data());
    compaction(); // 从0层开始尝试合并
}

/**
 * Insert/Update the key-value pair.
 * No return values for simplicity.
 */
void KVStore::put(uint64_t key, const std::string &val) {
    uint32_t nxtsize = s->getBytes();
    std::string res  = s->search(key);
    if (!res.length()) { // new add
        nxtsize += 12 + val.length();
    } else
        nxtsize = nxtsize - res.length() + val.length(); // change string
    if (nxtsize + 10240 + 32 <= MAXSIZE)
        s->insert(key, val); // 小于等于（不超过） 2MB
    else {
        sstable ss(s);
        s->reset();
        std::string url  = ss.getFilename();
        std::string path = "./data/level-0";
        if (!utils::dirExists(path)) {
            utils::mkdir(path.data());
            totalLevel = 0;
        }
        addsstable(ss, 0);      // 加入缓存
        ss.putFile(url.data()); // 加入磁盘
        compaction();
        s->insert(key, val);
    }
}

/**
 * Returns the (string) value of the given key.
 * An empty string indicates not found.
 */
std::string KVStore::get(uint64_t key) //
{
    uint64_t time = 0;
    int goalOffset;
    uint32_t goalLen;
    std::string goalUrl;
    std::string res = s->search(key);
    if (res.length()) { // 在memtable中找到, 或者是deleted，说明最近被删除过，
                        // 不用查sstable
        if (res == DEL)
            return "";
        return res;
    }
    for (int level = 0; level <= totalLevel; ++level) {
        for (sstablehead it : sstableIndex[level]) {
            if (key < it.getMinV() || key > it.getMaxV())
                continue;
            uint32_t len;
            int offset = it.searchOffset(key, len);
            if (offset == -1) {
                if (!level)
                    continue;
                else
                    break;
            }
            // sstable ss;
            // ss.loadFile(it.getFilename().data());
            if (it.getTime() > time) { // find the latest head
                time       = it.getTime();
                goalUrl    = it.getFilename();
                goalOffset = offset + 32 + 10240 + 12 * it.getCnt();
                goalLen    = len;
            }
        }
        if (time)
            break; // only a test for found
    }
    if (!goalUrl.length())
        return ""; // not found a sstable
    res = fetchString(goalUrl, goalOffset, goalLen);
    if (res == DEL)
        return "";
    return res;
}

/**
 * Delete the given key-value pair if it exists.
 * Returns false iff the key is not found.
 */
bool KVStore::del(uint64_t key) {
    std::string res = get(key);
    if (!res.length())
        return false; // not exist
    put(key, DEL);    // put a del marker
    return true;
}

/**
 * This resets the kvstore. All key-value pairs should be removed,
 * including memtable and all sstables files.
 */
void KVStore::reset() {
    s->reset(); // 先清空memtable
    std::vector<std::string> files;
    for (int level = 0; level <= totalLevel; ++level) { // 依层清空每一层的sstables
        std::string path = std::string("./data/level-") + std::to_string(level);
        int size         = utils::scanDir(path, files);
        for (int i = 0; i < size; ++i) {
            std::string file = path + "/" + files[i];
            utils::rmfile(file.data());
        }
        utils::rmdir(path.data());
        sstableIndex[level].clear();
    }
    totalLevel = -1;
}

/**
 * Return a list including all the key-value pair between key1 and key2.
 * keys in the list should be in an ascending order.
 * An empty string indicates not found.
 */

struct myPair {
    uint64_t key, time;
    int id, index;
    std::string filename;

    myPair(uint64_t key, uint64_t time, int index, int id,
           std::string file) { // construct function
        this->time     = time;
        this->key      = key;
        this->id       = id;
        this->index    = index;
        this->filename = file;
    }
};

struct cmp {
    bool operator()(myPair &a, myPair &b) {
        if (a.key == b.key)
            return a.time < b.time;
        return a.key > b.key;
    }
};


void KVStore::scan(uint64_t key1, uint64_t key2, std::list<std::pair<uint64_t, std::string>> &list) {
    std::vector<std::pair<uint64_t, std::string>> mem;
    // std::set<myPair> heap; // 维护一个指针最小堆
    std::priority_queue<myPair, std::vector<myPair>, cmp> heap;
    // std::vector<sstable> ssts;
    std::vector<sstablehead> sshs;
    s->scan(key1, key2, mem);   // add in mem
    std::vector<int> head, end; // [head, end)
    int cnt = 0;
    if (mem.size())
        heap.push(myPair(mem[0].first, INF, 0, -1, "qwq"));
    for (int level = 0; level <= totalLevel; ++level) {
        for (sstablehead it : sstableIndex[level]) {
            if (key1 > it.getMaxV() || key2 < it.getMinV())
                continue; // 无交集
            int hIndex = it.lowerBound(key1);
            int tIndex = it.lowerBound(key2);
            if (hIndex < it.getCnt()) { // 此sstable可用
                // sstable ss; // 读sstable
                std::string url = it.getFilename();
                // ss.loadFile(url.data());

                heap.push(myPair(it.getKey(hIndex), it.getTime(), hIndex, cnt++, url));
                head.push_back(hIndex);
                if (it.search(key2) == tIndex)
                    tIndex++; // tIndex为第一个不可的
                end.push_back(tIndex);
                // ssts.push_back(ss); // 加入ss
                sshs.push_back(it);
            }
        }
    }
    uint64_t lastKey = INF; // only choose the latest key
    while (!heap.empty()) { // 维护堆
        myPair cur = heap.top();
        heap.pop();
        if (cur.id >= 0) { // from sst
            if (cur.key != lastKey) {
                lastKey         = cur.key;
                uint32_t start  = sshs[cur.id].getOffset(cur.index - 1);
                uint32_t len    = sshs[cur.id].getOffset(cur.index) - start;
                uint32_t scnt   = sshs[cur.id].getCnt();
                std::string res = fetchString(cur.filename, 10240 + 32 + scnt * 12 + start, len);
                if (res.length() && res != DEL)
                    list.emplace_back(cur.key, res);
            }
            if (cur.index + 1 < end[cur.id]) { // add next one to heap
                heap.push(myPair(sshs[cur.id].getKey(cur.index + 1), cur.time, cur.index + 1, cur.id, cur.filename));
            }
        } else { // from mem
            if (cur.key != lastKey) {
                lastKey         = cur.key;
                std::string res = mem[cur.index].second;
                if (res.length() && res != DEL)
                    list.emplace_back(cur.key, mem[cur.index].second);
            }
            if (cur.index < mem.size() - 1) {
                heap.push(myPair(mem[cur.index + 1].first, cur.time, cur.index + 1, -1, cur.filename));
            }
        }
    }
}

struct simPair {
    int level;
    int table;
    int index;
    float sim;
    simPair(int level, int table, int index, float sim) {
        this->level = level;
        this->table = table;
        this->index = index;
        this->sim   = sim;
    }
    bool operator>(const simPair &other) const {
        return sim > other.sim;
    }
};

std::vector<std::pair<uint64_t, std::string>> KVStore::search_knn(std::string query, int k) {
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> query_vec = embedding_single(query);
    auto end = std::chrono::high_resolution_clock::now();
    long long embedding_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "embedding time: " << embedding_duration << "ms" << std::endl;
    static int cnt = 0;
    ++cnt;
    static long long TIME = 0;
    TIME += embedding_duration;
    if (cnt == 120) std::cout << "average embedding time: " << (double)TIME / cnt << "ms" << std::endl;

    // 先从memtable里搜索
    std::vector<std::pair<float, std::pair<uint64_t, std::string>>> res1 = s->search_knn(query_vec, k);

    // 再从所有的sstable里搜索
    std::priority_queue<simPair, std::vector<simPair>, std::greater<simPair>> heap;
    for (int curLevel = 0; curLevel <= totalLevel; ++curLevel) {
        for (int curTable = 0; curTable < vecs[curLevel].size(); ++curTable) {
            for (int index = 0; index < vecs[curLevel][curTable].size(); ++index) {
                float sim = common_embd_similarity_cos(query_vec.data(), vecs[curLevel][curTable][index].data(), query_vec.size());
                if (heap.size() < k) {
                    heap.push(simPair(curLevel, curTable, index, sim));
                } else {
                    if (sim > heap.top().sim) {
                        heap.pop();
                        heap.push(simPair(curLevel, curTable, index, sim));
                    }
                }
            }
        }
    }
    std::vector<std::pair<float, std::pair<uint64_t, std::string>>> res2;
    while (!heap.empty()) {
        simPair cur = heap.top();
        heap.pop();
        uint64_t key = sstableIndex[cur.level][cur.table].getKey(cur.index);
        std::string filename = sstableIndex[cur.level][cur.table].getFilename();
        uint32_t len;
        int offset = sstableIndex[cur.level][cur.table].searchOffset(key, len);
        std::string val = fetchString(filename, offset + 32 + 10240 + 12 * sstableIndex[cur.level][cur.table].getCnt(), len);
        res2.push_back(std::make_pair(cur.sim, std::make_pair(key, val)));
    }
    
    // 合并两个结果
    std::vector<std::pair<float, std::pair<uint64_t, std::string>>> res(res1.begin(), res1.end());
    res.insert(res.end(), res2.begin(), res2.end());
    std::sort(res.begin(), res.end(), [](const auto &a, const auto &b) {
        return a.first > b.first;
    });
    std::vector<std::pair<uint64_t, std::string>> ans;
    for (int i = 0; i < k; ++i) {
        ans.push_back(std::make_pair(res[i].second.first, res[i].second.second));
    }
    return ans;
}

void KVStore::build_hnsw(int M, int M_max, int efConstruction, int m_L) {
    hnsw = HNSW(M, M_max, efConstruction, m_L);

    // 把 memtable 中的数据插入 HNSW
    s->hnsw_insert_all(hnsw);

    // 把所有 sstable 中的数据插入 HNSW
    for (int curLevel = 0; curLevel <= totalLevel; ++curLevel) {
        for (int curTable = 0; curTable < vecs[curLevel].size(); ++curTable) {
            for (int index = 0; index < vecs[curLevel][curTable].size(); ++index) {
                uint64_t key = sstableIndex[curLevel][curTable].getKey(index);
                std::string filename = sstableIndex[curLevel][curTable].getFilename();
                uint32_t len;
                int offset = sstableIndex[curLevel][curTable].searchOffset(key, len);
                std::string val = fetchString(filename, offset + 32 + 10240 + 12 * sstableIndex[curLevel][curTable].getCnt(), len);        
                hnsw.insert(key, val, vecs[curLevel][curTable][index]);
            }
        }
    }
}

std::vector<std::pair<std::uint64_t, std::string>> KVStore::search_knn_hnsw(std::string query, int k) {
    return hnsw.search(query, k);
}

void KVStore::compaction() {
    int curLevel = 0;
    // TODO here
    while (sstableIndex[curLevel].size() > 1 << curLevel + 1) {
        std::vector<sstablehead> targets;  // 需要合并的所有sstable
        uint64_t start = INF, end = 0;  // 当前层要合并的sstable覆盖的区间
        int compactionNum;  // 当前层要合并的sstable数

        // 处理当前层要合并的sstable
        if (curLevel == 0) compactionNum = sstableIndex[curLevel].size();
        else compactionNum = sstableIndex[curLevel].size() - (1 << curLevel + 1);
        for (int i = 0; i < compactionNum; ++i) {
            sstablehead ssh = sstableIndex[curLevel][i];
            targets.push_back(ssh);
            start = std::min(start, ssh.getMinV());
            end   = std::max(end, ssh.getMaxV());
        }
        vecs[curLevel].erase(vecs[curLevel].begin(), vecs[curLevel].begin() + compactionNum);

        // 处理下一层要合并的sstable
        ++curLevel;
        if (curLevel > totalLevel) {
            std::string path = "./data/level-" + std::to_string(curLevel);
            if (!utils::dirExists(path)) {
                utils::mkdir(path.data());
                totalLevel++;
            }
        } else {
            for (int i = sstableIndex[curLevel].size() - 1; i >= 0; --i) {
                if (!(sstableIndex[curLevel][i].getMinV() > end || sstableIndex[curLevel][i].getMaxV() < start))
                    targets.push_back(sstableIndex[curLevel][i]);
                vecs[curLevel].erase(vecs[curLevel].begin() + i);
            }
        }

        // 存储需要合并的所有数据，第一个参数是key，第二个是val，第三个是time
        std::map<uint64_t, std::pair<std::string, uint64_t>> datas;
        for (auto it : targets) {
            sstable ss;
            ss.loadFile(it.getFilename().data());
            for (uint64_t i = 0; i < it.getCnt(); ++i) {
                std::pair<uint64_t, std::pair<std::string, uint64_t>> data = {it.getKey(i), {ss.getData(i), it.getTime()}};
                auto itr = datas.find(data.first);
                if (itr == datas.end()) {
                    datas.insert(data);
                } else {
                    if (itr->second.second < data.second.second)
                        itr->second = data.second;
                }
            }
        }

        // 删除所有要合并的sstable
        for (auto it : targets) {
            delsstable(it.getFilename());
        }

        // 在下一层创建新的sstable
        sstable ss;
        ss.setTime(++TIME);
        ss.setFilename("./data/level-" + std::to_string(curLevel) + "/" + std::to_string(TIME) + ".sst");
        for (auto data : datas) {
            if (ss.getBytes() + 12 + data.second.first.length() > MAXSIZE) {
                ss.putFile(ss.getFilename().data());
                addsstable(ss, curLevel);
                ss.reset();
                ss.setTime(++TIME);
                ss.setFilename("./data/level-" + std::to_string(curLevel) + "/" + std::to_string(TIME) + ".sst");
            }
            ss.insert(data.first, data.second.first);
        }
        if (ss.getCnt() > 0) {
            ss.putFile(ss.getFilename().data());
            addsstable(ss, curLevel);
        }
    }
}

void KVStore::delsstable(std::string filename) {
    for (int level = 0; level <= totalLevel; ++level) {
        int size = sstableIndex[level].size(), flag = 0;
        for (int i = 0; i < size; ++i) {
            if (sstableIndex[level][i].getFilename() == filename) {
                sstableIndex[level].erase(sstableIndex[level].begin() + i);
                flag = 1;
                break;
            }
        }
        if (flag)
            break;
    }
    int flag = utils::rmfile(filename.data());
    if (flag != 0) {
        std::cout << "delete fail!" << std::endl;
        std::cout << strerror(errno) << std::endl;
    }
}

void KVStore::addsstable(sstable ss, int level) {
    sstableIndex[level].push_back(ss.getHead());

    // embedding for each value
    std::vector<std::vector<float>> file_vecs;
    for (int i = 0; i < ss.getCnt(); ++i) {
        std::string val = ss.getData(i);
        file_vecs.push_back(embedding_single(val));
    }
    vecs[level].push_back(file_vecs);
}

char strBuf[2097152];

/**
 * @brief Fetches a substring from a file starting at a given offset.
 *
 * This function opens a file in binary read mode, seeks to the specified start offset,
 * reads a specified number of bytes into a buffer, and returns the buffer as a string.
 *
 * @param file The path to the file from which to read the substring.
 * @param startOffset The offset in the file from which to start reading.
 * @param len The number of bytes to read from the file.
 * @return A string containing the read bytes.
 */
std::string KVStore::fetchString(std::string file, int startOffset, uint32_t len) {
    // TODO here
    FILE *fp = fopen(file.c_str(), "rb");
    fseek(fp, startOffset, SEEK_SET);
    fread(strBuf, 1, len, fp);
    fclose(fp);
    return std::string(strBuf, len);
}
