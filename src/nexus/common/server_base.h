#ifndef NEXUS_COMMON_SERVER_BASE_H_
#define NEXUS_COMMON_SERVER_BASE_H_

#include <boost/asio.hpp>
#include <memory>
#include <string>

namespace nexus {

class ServerBase {
 public:
  // Disable copy
  ServerBase(const ServerBase&) = delete;
  ServerBase& operator=(const ServerBase&) = delete;
  // Construct the server given port.
  ServerBase(std::string port);
  // Construct the server given the IP address and port.
  ServerBase(std::string ip, std::string port);
  // Get the server address
  std::string address() const { return ip_ + ":" + port_; }
  // Get listening port
  std::string port() const { return port_; }
  // Start the server.
  virtual void Run();
  // Hanlde a stop operation.
  virtual void Stop();
 protected:
  // Asynchronously wait an accept request.
  void DoAccept();
  // Asynchronously wait a stop request.
  void DoAwaitStop();
  // Handle an accept operation.
  virtual void HandleAccept() = 0;
  // data fields
  std::string ip_;
  std::string port_;
  boost::asio::io_service io_service_;
  boost::asio::signal_set signals_;
  boost::asio::ip::tcp::acceptor acceptor_;
  boost::asio::ip::tcp::socket socket_;
};

} // namespace nexus

#endif // NEXUS_COMMON_SERVER_BASE_H_
