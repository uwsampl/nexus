#ifndef NEXUS_COMMON_CONNECTION_H_
#define NEXUS_COMMON_CONNECTION_H_

#include <boost/asio.hpp>
#include <deque>
#include <memory>
#include <mutex>

#include "common/message.h"

namespace nexus {

class Connection; // forward declare

class MessageHandler {
 public:
  /*!
   * \brief handles new message
   * \param conn Shared pointer of Connection
   * \param message Received message
   */
  virtual void HandleMessage(std::shared_ptr<Connection> conn,
                             std::shared_ptr<Message> message) = 0;
  /*!
   * \brief handles error
   * \param conn Shared pointer of Connection
   * \param ec Boost error code
   */
  virtual void HandleError(std::shared_ptr<Connection> conn,
                           boost::system::error_code ec) = 0;
};

class Connection : public std::enable_shared_from_this<Connection> {
 public:
  // disable copy
  Connection(const Connection&) = delete;
  Connection& operator=(const Connection&) = delete;
  // constructor
  explicit Connection(boost::asio::ip::tcp::socket socket,
                      MessageHandler* handler);
  /*! \brief starts processing packets received from socket */
  virtual void Start();
  /*! \brief stops the socket */
  virtual void Stop();
  /*!
   * \brief sends a message through socket
   * \param msg Shared pointer of message, yield the ownership to the function
   */
  virtual void Write(std::shared_ptr<Message> msg);

 protected:
  Connection(boost::asio::io_service& io_service, MessageHandler* handler);
  /*! \brief reads the header from the connection */
  void DoReadHeader();
  /*! \brief reads the body of message and invoke the handler */
  void DoReadBody(std::shared_ptr<Message> msg);
  /*! \brief sends the message to the peer */
  void DoWrite();

 protected:
  /*! \brief Socket */
  boost::asio::ip::tcp::socket socket_;
  /*! \brief Message handler */
  MessageHandler* handler_;
  /*! \brief Wrong header indicator */
  bool wrong_header_;
  /*! \brief Receiving message */
  //std::shared_ptr<Message> recv_message_;
  char msg_header_buffer_[MESSAGE_HEADER_SIZE];
  /*! \brief Queue for outbound messages */
  std::deque<std::shared_ptr<Message> > write_queue_;
  /*! \brief Mutex for write_queue_ */
  std::mutex write_queue_mutex_;
};

} // namespace nexus

#endif // NEXUS_COMMON_CONNECTION_H_
