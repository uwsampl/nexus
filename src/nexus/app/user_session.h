#ifndef NEXUS_APP_USER_SESSION_H_
#define NEXUS_APP_USER_SESSION_H_

#include "common/connection.h"

namespace nexus {
namespace app {

class UserSession : public Connection {
 public:
  UserSession(boost::asio::ip::tcp::socket socket, MessageHandler* handler) :
      Connection(std::move(socket), handler), user_id_(0) {}

  uint32_t user_id() const { return user_id_; }

  void set_user_id(uint32_t user_id) { user_id_ = user_id; }

 private:
  uint32_t user_id_;
};

} // namespace app
} // namespace nexus

#endif // NEXUS_APP_USER_SESSION_H_
