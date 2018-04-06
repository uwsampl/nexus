#include <cstring>
#include <glog/logging.h>

#include "nexus/common/message.h"

namespace nexus {

#if 0
#define htonll(x)                                                       \
  ((1==htonl(1)) ? (x) :                                                \
   ((uint64_t) htonl((x) & 0xFFFFFFFF) << 32) | htonl((uint64_t)(x) >> 32))

#define ntohll(x)                                                       \
  ((1==ntohl(1)) ? (x) :                                                \
   ((uint64_t) ntohl((x) & 0xFFFFFFFF) << 32) | ntohl((uint64_t)(x) >> 32))
#endif

bool DecodeHeader(const char* buffer, MessageHeader* header) {
  header->magic_number = ntohl(*(const uint32_t*) buffer);
  if (header->magic_number != NEXUS_SERVICE_MAGIC_NUMBER) {
    return false;
  }
  header->msg_type = ntohl(*(const uint32_t*) (buffer + 4));
  header->body_length = ntohl(*(const uint32_t*) (buffer + 8));
  return true;
}

Message::Message(const MessageHeader& header) {
  type_ = static_cast<MessageType>(header.msg_type);
  body_length_ = header.body_length;
  data_ = new char[MESSAGE_HEADER_SIZE + body_length_];
  *((uint32_t*) data_) = htonl(NEXUS_SERVICE_MAGIC_NUMBER);
  *((uint32_t*) (data_ + 4)) = htonl((uint32_t) type_);
  *((uint32_t*) (data_ + 8)) = htonl(body_length_);
}

Message::Message(MessageType type, size_t body_length) :
    type_(type),
    body_length_(body_length) {
  data_ = new char[MESSAGE_HEADER_SIZE + body_length];
  *((uint32_t*) data_) = htonl(NEXUS_SERVICE_MAGIC_NUMBER);
  *((uint32_t*) (data_ + 4)) = htonl((uint32_t) type);
  *((uint32_t*) (data_ + 8)) = htonl(body_length_);
}

Message::~Message() {
  delete[] data_;
}

void Message::DecodeBody(google::protobuf::Message* message) const {
  message->ParseFromArray(body(), body_length_);
}

void Message::EncodeBody(const google::protobuf::Message& message) {
  CHECK_GT(body_length_, 0) << "Body length hasn't been initialized";
  message.SerializeToArray(body(), body_length_);
}

} // namespace nexus
