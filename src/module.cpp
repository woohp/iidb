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
using std::pair;
using std::string;
using std::string_view;
using std::tuple;
using std::vector;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

typedef py::array_t<uint8_t, py::array::c_style | py::array::forcecast> array_type;
typedef std::variant<int64_t, string_view> generic_key_type;

string preprocess_key(const generic_key_type& key)
{
    return std::visit(
        [](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, int64_t>)
                return std::to_string(arg);
            else  // string_view
                return std::string { arg };
        },
        key);
}

class py_iidb : public ::iidb::iidb
{
public:
    py_iidb(string_view path, bool readonly, int mode)
        : ::iidb::iidb(path, !readonly)
        , path(path)
        , readonly(readonly)
        , mode(mode)
    { }

    py_iidb(py_iidb&&) = default;

public:
    py_iidb& __enter__()
    {
        return *this;
    }

    void __exit__(py::object exc_type, py::object exc_value, py::object exc_traceback)
    {
        this->close();
    }

    bool contains(int64_t key)
    {
        auto txn = this->begin();
        auto key_ = std::to_string(key);
        auto value = txn.get(key_);
        return static_cast<bool>(value);
    }

    std::variant<pair<int, int>, tuple<int, int, int>> get_image_dimension(int64_t key)
    {
        auto key_ = std::to_string(key);
        auto txn = this->begin();
        auto value = txn.get(key_);
        if (!value)
            throw std::out_of_range { "key not found: " + key_ };

        const uint16_t* header = reinterpret_cast<const uint16_t*>(value->data());
        int height = header[1];
        int width = header[2];
        int channels = header[3];

        if (channels == 1)
            return pair(height, width);
        else
            return tuple { height, width, channels };
    }

    array_type get(string_view key)
    {
        auto txn = this->begin();
        auto value = txn.get(key);
        if (!value)
            throw std::out_of_range { "key not found: " + std::string(key) };

        const uint16_t* header = reinterpret_cast<const uint16_t*>(value->data());
        int mode = header[0];
        int height = header[1];
        int width = header[2];
        int channels = header[3];

        array_type out;
        if (channels == 1)
            out.resize({ height, width });
        else
            out.resize({ height, width, channels });
        auto out_ptr = reinterpret_cast<std::byte*>(out.request().ptr);

        if (mode == 0)
            this->_init_zstd_contexts();

        this->_decompress(mode, out_ptr, out.nbytes(), value->data(), value->size(), 0);

        return out;
    }

    array_type get(int64_t key)
    {
        return this->get(std::to_string(key));
    }

    void put(string_view key, array_type value)
    {
        auto src_nbytes = value.nbytes();
        auto buffer_info = value.request();
        auto src_ptr = buffer_info.ptr;
        uint16_t height = buffer_info.shape[0];
        uint16_t width = buffer_info.shape[1];
        uint16_t channels = buffer_info.ndim == 2 ? 1 : buffer_info.shape[2];

        auto buffer = this->_compress(this->mode, height, width, channels, src_ptr, src_nbytes);
        auto txn = this->begin(true);
        txn.put(key, buffer);
        txn.commit();
    }

    void put(int64_t key, array_type value)
    {
        this->put(std::to_string(key), value);
    }

    array_type getmulti(const vector<generic_key_type>& keys)
    {
        vector<::iidb::blob<std::byte>> blobs(keys.size());
        vector<::iidb::image_dim> image_dims(keys.size());

        uint16_t mode = 0;

        auto txn = this->begin();
        for (size_t i = 0; i < keys.size(); i++)
        {
            auto key = preprocess_key(keys[i]);

            auto value = txn.get(key);
            if (!value)
                throw std::out_of_range { "key not found: " + key };
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

            this->_decompress(mode, this_out_ptr, image_nbytes, blob.data(), blob.size(), thread_idx);
        });

        return out;
    }

    void putmulti(const vector<pair<generic_key_type, array_type>>& items)
    {
        if (items.size() == 0)
            return;

        vector<string> to_insert_keys(items.size());
        vector<vector<std::byte>> to_insert_values(items.size());
        if (this->mode == 0)
            this->_init_zstd_contexts();

        for (size_t i = 0; i < items.size(); i++)
        {
            auto& [key, value] = items[i];
            to_insert_keys[i] = preprocess_key(key);

            auto src_nbytes = value.nbytes();
            auto buffer_info = value.request();
            auto src_ptr = buffer_info.ptr;
            uint16_t height = buffer_info.shape[0];
            uint16_t width = buffer_info.shape[1];
            uint16_t channels = buffer_info.ndim == 2 ? 1 : buffer_info.shape[2];
            to_insert_values[i] = this->_compress(this->mode, height, width, channels, src_ptr, src_nbytes);
        };

        auto txn = this->begin(true);
        for (size_t i = 0; i < items.size(); i++)
        {
            txn.put(to_insert_keys[i], to_insert_values[i]);
        }
        txn.commit();
    }

    const string path;
    const bool readonly;
    const int mode;
};

static_assert(std::is_move_constructible_v<py_iidb>);

PYBIND11_MODULE(iidb, m)
{
    using namespace pybind11::literals;

    py::class_<py_iidb>(m, "IIDB")
        .def(py::init<string_view, bool, int>(), "", "path"_a, "readonly"_a = true, "mode"_a = 0)
        .def_property_readonly("closed", &py_iidb::closed, "")
        .def("close", &py_iidb::close, "")
        .def("__enter__", &py_iidb::__enter__, "")
        .def("__exit__", &py_iidb::__exit__, "")
        .def("__contains__", &py_iidb::contains, "", "key"_a)
        .def("__len__", &py_iidb::size, "")
        .def("get_image_dimension", &py_iidb::get_image_dimension, "", "key"_a)
        .def("get", py::overload_cast<string_view>(&py_iidb::get), "", "key"_a)
        .def("get", py::overload_cast<int64_t>(&py_iidb::get), "", "key"_a)
        .def("__getitem__", py::overload_cast<std::string_view>(&py_iidb::get), "", "key"_a)
        .def("__getitem__", py::overload_cast<int64_t>(&py_iidb::get), "", "key"_a)
        .def("__setitem__", py::overload_cast<string_view, array_type>(&py_iidb::put), "", "key"_a, "value"_a)
        .def("__setitem__", py::overload_cast<int64_t, array_type>(&py_iidb::put), "", "key"_a, "value"_a)
        .def("getmulti", py::overload_cast<const vector<generic_key_type>&>(&py_iidb::getmulti), "", "keys"_a)
        .def("putmulti", &py_iidb::putmulti, "", "items"_a);

    m.def(
        "open",
        [](string_view path, bool readonly, int mode) { return py_iidb(path, readonly, mode); },
        "",
        "path"_a,
        "readonly"_a = true,
        "mode"_a = 0);

    m.def("__zstd_version__", []() {
        return std::to_string(ZSTD_VERSION_MAJOR) + '.' + std::to_string(ZSTD_VERSION_MINOR) + '.'
            + std::to_string(ZSTD_VERSION_RELEASE);
    });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#endif
}
