#include "common/server_base.h"

#include <glog/logging.h>
#include <signal.h>

namespace nexus {

ServerBase::ServerBase(std::string port) :
    ServerBase("0.0.0.0", port) {
}

ServerBase::ServerBase(std::string ip, std::string port)
    : ip_(ip),
      port_(port),
      io_service_(),
      signals_(io_service_),
      acceptor_(io_service_),
      socket_(io_service_) {
  // handle stop signal
  signals_.add(SIGINT);
  signals_.add(SIGTERM);

  DoAwaitStop();

  boost::asio::ip::tcp::resolver resolver(io_service_);
  boost::asio::ip::tcp::endpoint endpoint = *resolver.resolve({ip, port});
  acceptor_.open(endpoint.protocol());
  acceptor_.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
  acceptor_.bind(endpoint);
  acceptor_.listen();

  DoAccept();
}

void ServerBase::Run() {
  io_service_.run();
}

void ServerBase::Stop() {
  acceptor_.close();
}

void ServerBase::DoAccept() {
  acceptor_.async_accept(
      socket_,
      [this](boost::system::error_code ec){
        if (!acceptor_.is_open()) {
          return;
        }
        if (!ec) {
          HandleAccept();
        }
        DoAccept();
      });
}

void ServerBase::DoAwaitStop() {
  signals_.async_wait(
      [this](boost::system::error_code /*ec*/, int /*signo*/) {
        Stop();
      });
}

} // namespace nexus
