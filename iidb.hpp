#include <cstddef>
#include <lmdb.h>
#include <omp.h>
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
        return std::vector(this->data(), this->data + this->size());
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
            throw std::runtime_error{"mdb: failed to begin transaction"};
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
            throw std::runtime_error{"mdb: failed to open dbi"};

        MDB_val key_;
        blob<T> out;
        key_.mv_data = const_cast<char*>(key.data());
        key_.mv_size = key.size();

        auto rc = ::mdb_get(this->_handle, dbi_handle, &key_, &out);
        if (rc == MDB_NOTFOUND)
            return std::nullopt;
        else if (rc != MDB_SUCCESS)
            throw std::runtime_error{"mdb: failed to get value"};

        return out;
    }

    template <typename T = std::byte>
    std::optional<blob<T>> get(std::int64_t key)
    {
        return this->get(std::to_string(key));
    }
};

class lmdb
{
private:
    MDB_env* _handle = nullptr;

public:
    lmdb(std::string_view path, openflags flags = openflags::nosubdir | openflags::rdonly | openflags::nolock)
    {
        auto rc = ::mdb_env_create(&this->_handle);
        if (rc != MDB_SUCCESS)
            throw std::runtime_error{"mdb: failed to create environment"};

        rc = ::mdb_env_open(this->_handle, path.data(), static_cast<unsigned int>(flags), 0644);
        if (rc != MDB_SUCCESS)
            throw std::runtime_error{"mdb: failed to open environment"};
    }

    lmdb(const lmdb& other) = delete;

    constexpr lmdb(lmdb&& other)
    {
        std::swap(this->_handle, other._handle);
    }

    ~lmdb()
    {
        if (this->_handle)
        {
            ::mdb_env_close(this->_handle);
            this->_handle = nullptr;
        }
    }

    lmdb& set_mapsize(std::size_t size)
    {
        if (::mdb_env_set_mapsize(this->_handle, size) != MDB_SUCCESS)
            throw std::runtime_error{"mdb: failed to set map_size"};
        return *this;
    }

    txn begin(bool writeable = false)
    {
        return txn(this->_handle, writeable);
    }
};

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

class iidb : protected lmdb
{
public:
    iidb(std::string_view path, bool writeable = true)
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
        return image_dim{ height, width, channels };
    }

    std::optional<image_dim> get_image_dimension(std::int64_t key)
    {
        auto key_ = std::to_string(key);
        return this->get_image_dimension(key_);
    }

    std::optional<image> get(std::string_view key, std::byte* const out = nullptr)
    {
        auto txn = this->begin();
        auto value = txn.get(key);
        if (!value)
            return std::nullopt;

        const std::uint16_t* header = reinterpret_cast<const uint16_t*>(value->data());
        auto height = header[1];
        auto width = header[2];
        auto channels = header[3];
        auto total_size = width * height * channels;  // in bytes
        std::vector<std::byte> uncompressed;

        if (out)
            ZSTD_decompress(out, total_size, value->data() + 8, value->size() - 8);
        else
        {
            uncompressed.resize(total_size);
            ZSTD_decompress(uncompressed.data(), total_size, value->data() + 8, value->size() - 8);
        }

        return image{ std::move(uncompressed), height, width, channels };
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

        auto num_threads = std::min<unsigned int>(std::thread::hardware_concurrency(), keys.size());
        std::vector<txn> transactions;
        transactions.reserve(num_threads);
        for (unsigned int i = 0; i < num_threads; i++)
            transactions.emplace_back(this->begin());

        #pragma omp parallel for
        for (size_t i = 0; i < keys.size(); i++)
        {
            auto& thread_txn = transactions[omp_get_thread_num()];
            auto key = keys[i];
            auto key_ = std::to_string(key);
            auto value = thread_txn.get(key_);
            if (!value)
                throw std::out_of_range{ "key not found: " + key_ };
            blobs[i] = *value;

            const std::uint16_t* header = reinterpret_cast<const uint16_t*>(value->data());
            image_dims[i].height = header[1];
            image_dims[i].width = header[2];
            image_dims[i].channels = header[3];
        }

        // in serial, calculate the destination pointer addresses
        std::vector<std::pair<std::byte*, std::size_t>> dests;
        dests.reserve(keys.size());
        const auto original = out;
        for (size_t i = 0; i < keys.size(); i++)
        {
            const auto& dim = image_dims[i];
            std::size_t total_size = chunk_size.value_or(dim.width * dim.height * dim.channels);  // in bytes
            dests.push_back({ out, total_size });
            out += total_size;
        }

        #pragma omp parallel for
        for (size_t i = 0; i < blobs.size(); i++)
        {
            const auto& blob = blobs[i];
            auto [out_ptr, out_size] = dests[i];
            ZSTD_decompress(out_ptr, out_size, blob.data() + 8, blob.size() - 8);
        }
    }
};
}
