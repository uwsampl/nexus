#ifndef NEXUS_COMMON_DATA_TYPE_H_
#define NEXUS_COMMON_DATA_TYPE_H_

#include <cstdlib>
#include <type_traits>
#include <unordered_map>

#include "nexus/common/buffer.h"
#include "nexus/proto/nnquery.pb.h"

namespace nexus {

template<DataType> struct TypeMap;

template<> struct TypeMap<DT_BOOL> {
  using type = bool;
  static constexpr size_t size = 1;
};

template<> struct TypeMap<DT_INT> {
  using type = int32_t;
  static constexpr size_t size = 4;
};

template<> struct TypeMap<DT_FLOAT> {
  using type = float;
  static constexpr size_t size = 4;
};

template<> struct TypeMap<DT_DOUBLE> {
  using type = double;
  static constexpr size_t size = 8;
};

inline size_t type_size(DataType type) {
  switch (type) {
    case DT_INT:
      return TypeMap<DT_INT>::size;
    case DT_FLOAT:
      return TypeMap<DT_FLOAT>::size;
    case DT_DOUBLE:
      return TypeMap<DT_DOUBLE>::size;
    case DT_BOOL:
      return TypeMap<DT_BOOL>::size;
    default:
      return 0;
  };
}

class Array {
 public:
  Array();

  Array(DataType type, size_t num_elements, Device* device);

  Array(DataType type, size_t num_elements, std::shared_ptr<Buffer> buf);

  DataType data_type() const { return data_type_; }

  size_t num_elements() const { return num_elements_; }

  Device* device() const { return buffer_->device(); }

  DeviceType device_type() const { return buffer_->device()->type(); }

  void set_tag(int tag) { tag_ = tag; }

  int tag() const { return tag_; }

  template <typename T>
  T* Data() { return (T*) buffer_->data(); }

  template <typename T>
  const T* Data() const { return (const T*) buffer_->data(); }

  std::shared_ptr<Array> Slice(size_t offset, size_t num_elements);

  std::shared_ptr<Buffer> buffer() const { return buffer_; }

 private:
  DataType data_type_;
  size_t num_elements_;
  std::shared_ptr<Buffer> buffer_;
  int tag_;
};

using ArrayPtr = std::shared_ptr<Array>;

class Shape {
 public:
  Shape();

  Shape(const std::vector<int>& dims);

  Shape(std::initializer_list<int> list);

  Shape(const Shape& other);

  int dim(int axis) const;

  const std::vector<int>& dims() const;

  void set_dims(const std::vector<int>& dims);

  void set_dims(const std::vector<long int>& dims);

  void set_dims(std::initializer_list<int> list);

  size_t ndims() const;

  size_t NumElements(int axis = 0) const;

  friend std::ostream& operator<<(std::ostream& out, const Shape& shape);

 private:
  std::vector<int> dims_;
};

class Value {
 public:
  Value(const ValueProto& value);

  template<class T>
  const T& as() const;

  void ToProto(ValueProto* proto) const;
  
 private:
  DataType data_type_;
  bool b_;
  int i_;
  float f_;
  double d_;
  std::string s_;
  TensorProto tensor_;
  ImageProto image_;
  RectProto rect_;
};

class Record {
 public:
  Record(const RecordProto& record);

  void ToProto(RecordProto* proto) const;

  const Value& operator[](const std::string&& key) const;
  
 private:
  std::unordered_map<std::string, Value> values_;
};

} // namespace nexus

#endif // NEXUS_COMMON_DATA_TYPE_H_
