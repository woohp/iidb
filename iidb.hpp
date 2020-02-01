#include <cstddef>
#include <future>
#include <lmdb.h>
#include <lz4hc.h>
#include <math.h>
#include <optional>
#include <string_view>
#include <thread>
#include <vector>
#include <zstd.h>

namespace iidb
{
enum class openflags
{
    none = 0,
    fixedmap = MDB_FIXEDMAP,
    nosubdir = MDB_NOSUBDIR,
    rdonly = MDB_RDONLY,
    writemap = MDB_WRITEMAP,
    nometasync = MDB_NOMETASYNC,
    nosync = MDB_NOSYNC,
    mapasync = MDB_MAPASYNC,
    notls = MDB_NOTLS,
    nolock = MDB_NOLOCK,
    nordahead = MDB_NORDAHEAD,
    nomeminit = MDB_NOMEMINIT,
};

constexpr openflags operator|(const openflags& a, const openflags& b) noexcept
{
    return static_cast<openflags>(static_cast<unsigned int>(a) | static_cast<unsigned int>(b));
}

class lmdb;

template <typename T = std::byte>
struct blob : MDB_val
{
    const T* data() const
    {
        return reinterpret_cast<const T*>(this->mv_data);
    }

    std::size_t size() const
    {
        return this->mv_size;
    }

    std::vector<T> copy() const
    {
        return std::vector<T>(this->data(), this->data + this->size());
    }
};

class txn
{
private:
    MDB_txn* _handle = nullptr;
    friend class lmdb;

    txn(MDB_env* const env, bool writeable)
    {
        if (::mdb_txn_begin(env, nullptr, writeable ? 0 : MDB_RDONLY, &this->_handle) != MDB_SUCCESS)
            throw std::runtime_error { "mdb: failed to begin transaction" };
    }

public:
    txn(const txn& other) = delete;

    txn(txn&& other)
    {
        std::swap(this->_handle, other._handle);
    }

    ~txn()
    {
        this->abort();
    }

    void commit()
    {
        if (this->_handle)
        {
            ::mdb_txn_commit(this->_handle);
            this->_handle = nullptr;
        }
    }

    void abort()
    {
        if (this->_handle)
        {
            ::mdb_txn_abort(this->_handle);
            this->_handle = nullptr;
        }
    }

    template <typename T = std::byte>
    std::optional<blob<T>> get(std::string_view key)
    {
        MDB_dbi dbi_handle = 0;
        if (::mdb_dbi_open(this->_handle, nullptr, 0, &dbi_handle) != MDB_SUCCESS)
            throw std::runtime_error { "mdb: failed to open dbi" };

        MDB_val key_;
        blob<T> out;
        key_.mv_data = const_cast<char*>(key.data());
        key_.mv_size = key.size();

        auto rc = ::mdb_get(this->_handle, dbi_handle, &key_, &out);
        if (rc == MDB_NOTFOUND)
            return std::nullopt;
        else if (rc != MDB_SUCCESS)
            throw std::runtime_error { "mdb: failed to get value" };

        return out;
    }

    template <typename T = std::byte>
    void put(std::string_view key, blob<T>& value)
    {
        MDB_dbi dbi_handle = 0;
        if (::mdb_dbi_open(this->_handle, nullptr, 0, &dbi_handle) != MDB_SUCCESS)
            throw std::runtime_error { "mdb: failed to open dbi" };

        MDB_val key_;
        key_.mv_data = const_cast<char*>(key.data());
        key_.mv_size = key.size();

        auto rc = ::mdb_put(this->_handle, dbi_handle, &key_, &value, 0);
        if (rc != MDB_SUCCESS)
            throw std::runtime_error { "mdb: failed to get value" };
    }

    template <typename T = std::byte>
    std::optional<blob<T>> get(std::int64_t key)
    {
        return this->get(std::to_string(key));
    }
};

class lmdb
{
protected:
    MDB_env* _handle = nullptr;

public:
    lmdb(std::string_view path, openflags flags = openflags::nosubdir | openflags::rdonly | openflags::nolock)
    {
        auto rc = ::mdb_env_create(&this->_handle);
        if (rc != MDB_SUCCESS)
            throw std::runtime_error { "mdb: failed to create environment" };

        rc = ::mdb_env_open(this->_handle, path.data(), static_cast<unsigned int>(flags), 0644);
        if (rc != MDB_SUCCESS)
            throw std::runtime_error { "mdb: failed to open environment" };
    }

    lmdb(const lmdb& other) = delete;

    constexpr lmdb(lmdb&& other)
    {
        std::swap(this->_handle, other._handle);
    }

    ~lmdb()
    {
        this->close();
    }

    bool closed() const
    {
        return !static_cast<bool>(this->_handle);
    }

    void close()
    {
        if (this->_handle)
        {
            ::mdb_env_close(this->_handle);
            this->_handle = nullptr;
        }
    }

    size_t size() const
    {
        MDB_stat stat;
        if (::mdb_env_stat(this->_handle, &stat) != MDB_SUCCESS)
            throw std::runtime_error { "mdb: failed to get env info stat" };

        return stat.ms_entries;
    }

    lmdb& set_mapsize(std::size_t size)
    {
        if (::mdb_env_set_mapsize(this->_handle, size) != MDB_SUCCESS)
            throw std::runtime_error { "mdb: failed to set map_size" };
        return *this;
    }

    txn begin(bool writeable = false)
    {
        return txn(this->_handle, writeable);
    }
};

template <typename F>
void parallel_for_(const size_t start, const size_t end, size_t grain_size, F f)
{
    const auto total = end - start;
    const auto hardware_concurrency = 4u;
    if (total <= grain_size || hardware_concurrency == 1)
    {
        f(0, start, end);
        return;
    }

    const auto n_threads
        = std::min(hardware_concurrency, static_cast<unsigned int>(ceil(static_cast<double>(total) / grain_size)));
    const auto thread_stride = total / n_threads;
    auto leftover = total - n_threads * thread_stride;

    std::vector<std::future<void>> futures(n_threads);
    for (size_t i = 0, thread_start = start; i < n_threads; i++)
    {
        auto this_thread_stride = thread_stride;
        if (leftover > 0)
        {
            this_thread_stride++;
            leftover--;
        }
        auto thread_end = std::min(end, thread_start + this_thread_stride);
        futures[i] = std::async(std::launch::async, f, i, thread_start, thread_end);
        thread_start += this_thread_stride;
    }

    for (const auto& future : futures)
        future.wait();
}

struct image
{
    std::vector<std::byte> data;
    std::uint16_t height;
    std::uint16_t width;
    std::uint16_t channels;
};

struct image_dim
{
    std::uint16_t height;
    std::uint16_t width;
    std::uint16_t channels;
};

class iidb : public lmdb
{
public:
    iidb(std::string_view path, bool writeable = false)
        : lmdb(path, openflags::nosubdir | openflags::nolock | (writeable ? openflags::none : openflags::rdonly))
    {
        this->set_mapsize(1024L * 1024 * 1024 * 1024);  // 1 Tebibyte
    }

    std::optional<image_dim> get_image_dimension(std::string_view key)
    {
        auto txn = this->begin();
        auto value = txn.get(key);
        if (!value)
            return std::nullopt;

        const std::uint16_t* header = reinterpret_cast<const uint16_t*>(value->data());
        auto height = header[1];
        auto width = header[2];
        auto channels = header[3];
        return image_dim { height, width, channels };
    }

    std::optional<image_dim> get_image_dimension(std::int64_t key)
    {
        auto key_ = std::to_string(key);
        return this->get_image_dimension(key_);
    }

    std::optional<image> get(std::string_view key, std::byte* out = nullptr)
    {
        auto txn = this->begin();
        auto value = txn.get(key);
        if (!value)
            return std::nullopt;

        const std::uint16_t* header = reinterpret_cast<const uint16_t*>(value->data());
        auto mode = header[0];
        auto height = header[1];
        auto width = header[2];
        auto channels = header[3];
        auto total_size = width * height * channels;  // in bytes
        std::vector<std::byte> uncompressed;

        if (!out)
        {
            uncompressed.resize(total_size);
            out = uncompressed.data();
        }

        if (mode == 0)
            ZSTD_decompress(out, total_size, value->data() + 8, value->size() - 8);
        else if (mode == 1)
            LZ4_decompress_safe(
                reinterpret_cast<const char*>(value->data() + 8),
                reinterpret_cast<char*>(out),
                value->size() - 8,
                total_size);

        return image { std::move(uncompressed), height, width, channels };
    }

    std::optional<image> get(std::int64_t key, std::byte* const out = nullptr)
    {
        auto key_ = std::to_string(key);
        return this->get(key_, out);
    }

    void getmulti(
        const std::vector<std::int64_t>& keys, std::byte* out, std::optional<std::size_t> chunk_size = std::nullopt)
    {
        std::vector<blob<std::byte>> blobs(keys.size());
        std::vector<image_dim> image_dims(keys.size());

        std::uint16_t mode = 0;

        parallel_for_(0, keys.size(), 1, [&](size_t, size_t start, size_t end) {
            auto thread_txn = this->begin();

            for (size_t i = start; i < end; i++)
            {
                auto key = keys[i];
                auto key_ = std::to_string(key);
                auto value = thread_txn.get(key_);
                if (!value)
                    throw std::out_of_range { "key not found: " + key_ };
                blobs[i] = *value;

                const std::uint16_t* header = reinterpret_cast<const uint16_t*>(value->data());
                image_dims[i].height = header[1];
                image_dims[i].width = header[2];
                image_dims[i].channels = header[3];
                if (i == 0)
                    mode = header[0];
            }
        });

        // in serial, calculate the destination pointer addresses
        std::vector<std::pair<std::byte*, std::size_t>> dests;
        dests.reserve(keys.size());
        for (size_t i = 0; i < keys.size(); i++)
        {
            const auto& dim = image_dims[i];
            std::size_t total_size = chunk_size.value_or(dim.width * dim.height * dim.channels);  // in bytes
            dests.emplace_back(out, total_size);
            out += total_size;
        }

        parallel_for_(0, blobs.size(), 1, [&](size_t, size_t start, size_t end) {
            ZSTD_DCtx* zstd_dctx = nullptr;
            if (mode == 0)
                zstd_dctx = ZSTD_createDCtx();

            for (size_t i = start; i < end; i++)
            {
                const auto& blob = blobs[i];
                auto [out_ptr, out_size] = dests[i];
                if (mode == 0)
                    ZSTD_decompressDCtx(zstd_dctx, out_ptr, out_size, blob.data() + 8, blob.size() - 8);
                else if (mode == 1)
                {
                    LZ4_decompress_safe(
                        reinterpret_cast<const char*>(blob.data() + 8),
                        reinterpret_cast<char*>(out_ptr),
                        blob.size() - 8,
                        out_size);
                }
            }

            if (zstd_dctx)
                ZSTD_freeDCtx(zstd_dctx);
        });
    }
};

static_assert(std::is_move_constructible_v<iidb>);

}
