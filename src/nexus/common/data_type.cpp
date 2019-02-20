#include "nexus/common/data_type.h"
#include <glog/logging.h>

namespace nexus {

Array::Array() : buffer_(nullptr), tag_(-1) {}

Array::Array(DataType type, size_t num_elements, Device* device) :
    data_type_(type),
    num_elements_(num_elements),
    tag_(-1) {
  size_t nbytes = num_elements * type_size(type);
  buffer_ = std::make_shared<Buffer>(nbytes, device);
}

Array::Array(DataType type, size_t num_elements, std::shared_ptr<Buffer> buf) :
    data_type_(type),
    num_elements_(num_elements),
    buffer_(buf),
    tag_(-1) {
  CHECK(buf != nullptr) << "buf must not be nullptr";
  size_t nbytes = num_elements * type_size(type);
  CHECK_LE(nbytes, buf->nbytes()) << "Buffer size is not large enough (" <<
      nbytes << " vs " << buf->nbytes() << ")";
}

std::shared_ptr<Array> Array::Slice(size_t offset, size_t num_elements) {
  size_t offset_bytes = offset * type_size(data_type_);
  size_t nbytes = num_elements * type_size(data_type_);
  auto slice_buf = buffer_->Slice(offset_bytes, nbytes);
  return std::make_shared<Array>(data_type_, num_elements, slice_buf);
}

Shape::Shape() {}

Shape::Shape(const std::vector<int>& dims) :
    dims_(dims) {}

Shape::Shape(std::initializer_list<int> list) :
    dims_(list) {}

Shape::Shape(const Shape& other) :
    dims_(other.dims_) {}

int Shape::dim(int axis) const {
  CHECK_LT(axis, dims_.size());
  return dims_[axis];
}

size_t Shape::ndims() const {
  return dims_.size();
}

const std::vector<int>& Shape::dims() const {
  return dims_;
}

void Shape::set_dims(const std::vector<int>& dims) {
  dims_.clear();
  dims_ = dims;
}

void Shape::set_dims(const std::vector<long int>& dims) {
  dims_.clear();
  dims_.resize(dims.size());
  for (uint i = 0; i < dims.size(); ++i) {
    dims_[i] = dims[i];
  }
}

void Shape::set_dims(std::initializer_list<int> list) {
  dims_.clear();
  dims_ = list;
}

size_t Shape::NumElements(int axis) const {
  CHECK_LT(axis, dims_.size());
  size_t size = 1;
  for (uint i = axis; i < dims_.size(); ++i) {
    size *= dims_[i];
  }
  return size;
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  if (shape.dims_.empty()) {
    return out;
  }
  out << shape.dims_[0];
  for (uint i = 1; i < shape.dims_.size(); ++i) {
    out << "x" << shape.dims_[i];
  }
  return out;
}

Value::Value(const ValueProto& value) :
    data_type_(value.data_type()) {
  switch (data_type_) {
    case DT_BOOL: { b_ = value.b(); break; }
    case DT_INT8:
    case DT_UINT8:
    case DT_INT32:
    case DT_UINT32: { i_ = value.i(); break; }
    case DT_FLOAT: { f_ = value.f(); break; }
    case DT_DOUBLE: { d_ = value.d(); break; }
    case DT_STRING: { s_ = value.s(); break; }
    case DT_TENSOR: {
      tensor_.CopyFrom(value.tensor());
      break;
    }
    case DT_IMAGE: {
      image_.CopyFrom(value.image());
      break;
    }
    case DT_RECT: {
      rect_.CopyFrom(value.rect());
      break;
    }
    default:
      LOG(FATAL) << "Unknown data type: " << data_type_;
      break;
  }
}

template<class T>
const T& Value::as() const {
  LOG(FATAL) << "Unsupported data type: " << typeid(T).name();
}

template<>
const bool& Value::as<bool>() const {
  CHECK_EQ(data_type_, DT_BOOL) << "Data type mismatch";
  return b_;
}

template<>
const int& Value::as<int>() const {
  if (data_type_ != DT_INT8 && data_type_ != DT_UINT8 &&
      data_type_ != DT_INT32 && data_type_ != DT_UINT32) {
    LOG(FATAL) << "Data type mismatch";
  }
  return i_;
}

template<>
const float& Value::as<float>() const {
  CHECK_EQ(data_type_, DT_FLOAT) << "Data type mismatch";
  return f_;
}

template<>
const double& Value::as<double>() const {
  CHECK_EQ(data_type_, DT_DOUBLE) << "Data type mismatch";
  return d_;
}

template<>
const std::string& Value::as<std::string>() const {
  CHECK_EQ(data_type_, DT_STRING) << "Data type mismatch";
  return s_;
}

template<>
const TensorProto& Value::as<TensorProto>() const {
  CHECK_EQ(data_type_, DT_TENSOR) << "Data type mismatch";
  return tensor_;
}

template<>
const ImageProto& Value::as<ImageProto>() const {
  CHECK_EQ(data_type_, DT_IMAGE) << "Data type mismatch";
  return image_;
}

template<>
const RectProto& Value::as<RectProto>() const {
  CHECK_EQ(data_type_, DT_RECT) << "Data type mismatch";
  return rect_;
}

void Value::ToProto(ValueProto* proto) const {
  proto->set_data_type(data_type_);
  switch (data_type_) {
    case DT_BOOL: { proto->set_b(b_); break; }
    case DT_INT8:
    case DT_UINT8:
    case DT_INT32:
    case DT_UINT32: { proto->set_i(i_); break; }
    case DT_FLOAT: { proto->set_f(f_); break; }
    case DT_DOUBLE: { proto->set_d(d_); break; }
    case DT_STRING: { proto->set_s(s_); break; }
    case DT_TENSOR: {
      proto->mutable_tensor()->CopyFrom(tensor_);
      break;
    }
    case DT_IMAGE: {
      proto->mutable_image()->CopyFrom(image_);
      break;
    }
    case DT_RECT: {
      proto->mutable_rect()->CopyFrom(rect_);
      break;
    }
    default:
      break;
  }
}

Record::Record(const RecordProto& record) {
  for (auto named_value : record.named_value()) {
    values_.emplace(named_value.name(), Value(named_value));
  }
}

void Record::ToProto(RecordProto* proto) const {
  for (auto iter : values_) {
    auto value_p = proto->add_named_value();
    value_p->set_name(iter.first);
    iter.second.ToProto(value_p);
  }
}

const Value& Record::operator[](const std::string&& key) const {
  return values_.at(key);
}

} // namespace nexus
