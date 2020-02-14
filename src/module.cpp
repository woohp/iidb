#include "../iidb.hpp"
#include <condition_variable>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>
namespace py = pybind11;
using namespace iidb;

typedef py::array_t<uint8_t, py::array::c_style | py::array::forcecast> array_type;

class py_iidb : public iidb
{
public:
    py_iidb(std::string_view path, bool readonly, int mode)
        : iidb(path, !readonly)
        , path(path)
        , readonly(readonly)
        , mode(mode)
    {}

    py_iidb(py_iidb&&) = default;

private:
    void _set_header(void* bytes, uint16_t height, uint16_t width, uint16_t channels)
    {
        auto header = reinterpret_cast<uint16_t*>(bytes);
        header[0] = this->mode;
        header[1] = height;
        header[2] = width;
        header[3] = channels;
    }

public:
    py_iidb& __enter__()
    {
        return *this;
    }

    void __exit__(py::object exc_type, py::object exc_value, py::object exc_traceback)
    {
        this->close();
    }

    bool contains(std::int64_t key)
    {
        auto txn = this->begin();
        auto key_ = std::to_string(key);
        auto value = txn.get(key_);
        return static_cast<bool>(value);
    }

    std::variant<std::pair<int, int>, std::tuple<int, int, int>> get_image_dimension(std::int64_t key)
    {
        auto key_ = std::to_string(key);
        auto txn = this->begin();
        auto value = txn.get(key_);
        if (!value)
            throw std::out_of_range { "key not found: " + key_ };

        const std::uint16_t* header = reinterpret_cast<const uint16_t*>(value->data());
        int height = header[1];
        int width = header[2];
        int channels = header[3];

        if (channels == 1)
            return std::pair(height, width);
        else
            return std::tuple(height, width, channels);
    }

    array_type get(std::int64_t key)
    {
        auto txn = this->begin();
        auto key_ = std::to_string(key);
        auto value = txn.get(key_);
        if (!value)
            throw std::out_of_range { "key not found: " + key_ };

        const std::uint16_t* header = reinterpret_cast<const uint16_t*>(value->data());
        int mode = header[0];
        int height = header[1];
        int width = header[2];
        int channels = header[3];

        array_type out;
        if (channels == 1)
            out.resize({ height, width });
        else
            out.resize({ height, width, channels });
        auto out_ptr = reinterpret_cast<char*>(out.request().ptr);

        if (mode == 0)
        {
            this->_init_zstd_contexts();
            ZSTD_decompressDCtx(this->zstd_dcontexts[0], out_ptr, out.nbytes(), value->data() + 8, value->size() - 8);
        }
        else
        {
            LZ4_decompress_safe(
                reinterpret_cast<const char*>(value->data() + 8),
                reinterpret_cast<char*>(out_ptr),
                value->size() - 8,
                out.nbytes());
        }

        return out;
    }

    void put(std::int64_t key, array_type value)
    {
        auto key_ = std::to_string(key);

        auto src_nbytes = value.nbytes();
        auto buffer_info = value.request();
        auto src_ptr = buffer_info.ptr;
        uint16_t height = buffer_info.shape[0];
        uint16_t width = buffer_info.shape[1];
        uint16_t channels = buffer_info.ndim == 2 ? 1 : buffer_info.shape[2];

        if (this->mode == 0)
        {
            this->_init_zstd_contexts();
            auto compress_bound_size = ZSTD_compressBound(src_nbytes);
            auto temp_buffer = new std::byte[compress_bound_size + 8];
            this->_set_header(temp_buffer, height, width, channels);
            auto compressed_nbytes = ZSTD_compressCCtx(
                this->zstd_ccontexts[0], temp_buffer + 8, compress_bound_size, src_ptr, src_nbytes, 7);

            auto txn = this->begin(true);
            blob<std::byte> value_blob { compressed_nbytes + 8, temp_buffer };
            txn.put(key_, value_blob);
            txn.commit();

            delete temp_buffer;
        }

        else if (this->mode == 1)
        {
            auto compress_bound_size = LZ4_compressBound(src_nbytes);
            auto temp_buffer = new char[compress_bound_size + 8];
            this->_set_header(temp_buffer, height, width, channels);
            auto compressed_nbytes = LZ4_compress_HC(
                reinterpret_cast<const char*>(src_ptr), temp_buffer + 8, src_nbytes, compress_bound_size, 7);

            auto txn = this->begin(true);
            blob<std::byte> value_blob { static_cast<size_t>(compressed_nbytes) + 8, temp_buffer };
            txn.put(key_, value_blob);
            txn.commit();

            delete temp_buffer;
        }
    }

    array_type getmulti(const std::vector<std::int64_t>& keys)
    {
        std::vector<blob<std::byte>> blobs(keys.size());
        std::vector<image_dim> image_dims(keys.size());

        std::uint16_t mode = 0;

        auto txn = this->begin();
        for (size_t i = 0; i < keys.size(); i++)
        {
            auto key = keys[i];
            auto key_ = std::to_string(key);
            auto value = txn.get(key_);
            if (!value)
                throw std::out_of_range { "key not found: " + key_ };
            blobs[i] = *value;

            auto header = reinterpret_cast<const uint16_t*>(value->data());
            mode = header[0];
            image_dims[i].height = header[1];
            image_dims[i].width = header[2];
            image_dims[i].channels = header[3];
        }

        // make sure all the images are of the same shape
        for (size_t i = 1; i < keys.size(); i++)
        {
            if (image_dims[i].width != image_dims[0].width || image_dims[i].height != image_dims[0].height
                || image_dims[i].channels != image_dims[0].channels)
                throw std::runtime_error { "images not all the same shape" };
        }
        auto image_dim = image_dims[0];
        auto image_nbytes = image_dim.width * image_dim.height * image_dim.channels;

        // create output array
        array_type out;
        if (image_dim.channels == 1)
            out.resize({ int(keys.size()), int(image_dim.height), int(image_dim.width) });
        else
            out.resize({ int(keys.size()), int(image_dim.height), int(image_dim.width), int(image_dim.channels) });
        auto out_ptr = reinterpret_cast<std::byte*>(out.request().ptr);

        // create zstd contexts
        if (mode == 0)
            this->_init_zstd_contexts();

        this->pool->parallel_for(0, blobs.size(), [&](size_t i, size_t thread_idx) {
            const auto& blob = blobs[i];
            auto this_out_ptr = out_ptr + i * image_nbytes;
            if (mode == 0)
                ZSTD_decompressDCtx(
                    this->zstd_dcontexts[thread_idx], this_out_ptr, image_nbytes, blob.data() + 8, blob.size() - 8);
            else if (mode == 1)
            {
                LZ4_decompress_safe(
                    reinterpret_cast<const char*>(blob.data() + 8),
                    reinterpret_cast<char*>(this_out_ptr),
                    blob.size() - 8,
                    image_nbytes);
            }
        });

        return out;
    }

    void putmulti(const std::vector<std::pair<std::int64_t, array_type>>& items)
    {
        if (items.size() == 0)
            return;

        std::vector<std::string> to_insert_keys(items.size());
        std::vector<std::vector<std::byte>> to_insert_values(items.size());
        if (this->mode == 0)
            this->_init_zstd_contexts();

        for (size_t i = 0; i < items.size(); i++)
        {
            auto& [key, value] = items[i];
            to_insert_keys[i] = std::to_string(key);

            auto src_nbytes = value.nbytes();
            auto buffer_info = value.request();
            auto src_ptr = buffer_info.ptr;
            uint16_t height = buffer_info.shape[0];
            uint16_t width = buffer_info.shape[1];
            uint16_t channels = buffer_info.ndim == 2 ? 1 : buffer_info.shape[2];

            if (this->mode == 0)
            {
                auto compress_bound_size = ZSTD_compressBound(src_nbytes);
                to_insert_values[i].resize(compress_bound_size + 8);
                this->_set_header(to_insert_values[i].data(), height, width, channels);
                auto compressed_nbytes = ZSTD_compressCCtx(
                    this->zstd_ccontexts[0],
                    to_insert_values[i].data() + 8,
                    compress_bound_size,
                    src_ptr,
                    src_nbytes,
                    7);
                to_insert_values[i].resize(compressed_nbytes + 8);
                ZSTD_CCtx_reset(this->zstd_ccontexts[0], ZSTD_reset_session_only);
            }

            else if (this->mode == 1)
            {
                auto compress_bound_size = LZ4_compressBound(src_nbytes);
                to_insert_values[i].resize(compress_bound_size + 8);
                this->_set_header(to_insert_values[i].data(), height, width, channels);
                auto compressed_nbytes = LZ4_compress_default(
                    reinterpret_cast<const char*>(src_ptr),
                    reinterpret_cast<char*>(to_insert_values[i].data() + 8),
                    src_nbytes,
                    compress_bound_size);
                to_insert_values[i].resize(compressed_nbytes + 8);
            }
        };

        auto txn = this->begin(true);
        for (size_t i = 0; i < items.size(); i++)
        {
            blob<std::byte> value_blob { to_insert_values[i].size(), to_insert_values[i].data() };
            txn.put(to_insert_keys[i], value_blob);
        }
        txn.commit();
    }

    const std::string path;
    const bool readonly;
    const int mode;
};

static_assert(std::is_move_constructible_v<py_iidb>);

PYBIND11_MODULE(iidb, m)
{
    using namespace pybind11::literals;

    py::class_<py_iidb>(m, "IIDB")
        .def(py::init<std::string_view, bool, int>(), "", "path"_a, "readonly"_a = true, "mode"_a = 0)
        .def_property_readonly("closed", &py_iidb::closed, "")
        .def("close", &py_iidb::close, "")
        .def("__enter__", &py_iidb::__enter__, "")
        .def("__exit__", &py_iidb::__exit__, "")
        .def("__contains__", &py_iidb::contains, "", "key"_a)
        .def("__len__", &py_iidb::size, "")
        .def("get_image_dimension", &py_iidb::get_image_dimension, "", "key"_a)
        .def("get", &py_iidb::get, "", "key"_a)
        .def("__getitem__", &py_iidb::get, "", "key"_a)
        .def("put", &py_iidb::put, "", "key"_a, "value"_a)
        .def("__setitem__", &py_iidb::put, "", "key"_a, "value"_a)
        .def("getmulti", (array_type(py_iidb::*)(const std::vector<std::int64_t>& keys)) & py_iidb::getmulti)
        .def("putmulti", &py_iidb::putmulti, "", "items"_a);

    m.def(
        "open",
        [](std::string_view path, bool readonly, int mode) { return py_iidb(path, readonly, mode); },
        "",
        "path"_a,
        "readonly"_a = true,
        "mode"_a = 0);

    m.def("__zstd_version__", []() {
        return std::to_string(ZSTD_VERSION_MAJOR) + '.' + std::to_string(ZSTD_VERSION_MINOR) + '.'
            + std::to_string(ZSTD_VERSION_RELEASE);
    });

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#endif
}
