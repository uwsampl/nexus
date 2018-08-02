#ifndef NEXUS_COMMON_MESSAGE_H_
#define NEXUS_COMMON_MESSAGE_H_

#include <arpa/inet.h>
#include <google/protobuf/message.h>
#include <string>

namespace nexus {

/*! \brief Message types */
enum MessageType {
  /*! \brief register user to frontend */
  kUserRegister = 1,
  /*! \brief request from user to fronend */
  kUserRequest = 2,
  /*! \brief reply from frontend to user */
  kUserReply = 3,

  // Internal message type
  /*! \brief request from frontend to backend */
  kBackendRequest = 100,
  /*! \brief reply from backend to frontend */
  kBackendReply = 101,
  /*! \brief relay request from backend to backup */
  kBackendRelay = 102,
  /*! \brief relay reply from backup */
  kBackendRelayReply = 103,
};

/*! \brief Message header format */
struct MessageHeader {
  /*! \brief magic number field */
  uint32_t magic_number;
  /*! \brief message type */
  uint32_t msg_type;
  /*! \brief length of payload */
  uint32_t body_length;
};

/*! \brief Magic number for Nexus service */
#define NEXUS_SERVICE_MAGIC_NUMBER  0xDEADBEEF
/*! \brief Header length in bytes */
#define MESSAGE_HEADER_SIZE         sizeof(MessageHeader)

bool DecodeHeader(const char* buffer, MessageHeader* header);

/*!
 * \brief Message is used to hold the packets that are communicated between
 * client and frontend server, and between frontend server and backend server.
 */
class Message {
 public:
  /*!
   * \brief Construct a nessage.
   *
   * It allocates the data buffer with maximal size. This constructor is mainly
   * used to hold an inbound packet when the message size is unknown.
   */
  //Message();
  Message(const MessageHeader& header);
  /*!
   * \brief Construct a nessage with explicit body length.
   * 
   * It allocates the data buffer with body length plus header size. This
   * constructor is mainly used to hold an outbound packet when the message
   * size is known
   *
   * \param body_length Length of payload in bytes
   */
  Message(MessageType type, size_t body_length);
  /*! \brief Destruct a message. */
  ~Message();
  /*! \brief Get the data pointer */
  char* data() { return data_; }
  /*! \brief Get the read-only data pointer */
  const char* data() const { return data_; }
  /*! \brief Get the body pointer */
  char* body() { return data_ + MESSAGE_HEADER_SIZE; }
  /*! \brief Get the read-only body pointer */
  const char* body() const { return data_ + MESSAGE_HEADER_SIZE; }
  /*! \brief Get the length of entire message in bytes */
  size_t length() const { return MESSAGE_HEADER_SIZE + body_length_; }
  /*! \brief Get the length of body in bytes */
  size_t body_length() const { return body_length_; }
  /*! \brief Get the type of message */
  MessageType type() const { return type_; }
  /*!
   * \brief Set the message type
   * \param type Message type
   */
  void set_type(MessageType type);
  /*!
   * \brief Decode the message from the body
   * \param message Protobuf message for the decoding result
   */
  void DecodeBody(google::protobuf::Message* message) const;
  /*!
   * \brief Encode the protobuf message and store in the body
   * \param message Protobuf message to encode
   */
  void EncodeBody(const google::protobuf::Message& message);

 private:
  /*! \brief Data buffer */
  char* data_;
  /*! \brief Message type */
  MessageType type_;
  /*! \brief Length of message body in bytes */
  size_t body_length_;
};

} // namespace nexus

#endif // NEXUS_COMMON_MESSAGE_H_
