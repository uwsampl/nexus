#ifndef NEXUS_COMMON_RPC_CALL_H_
#define NEXUS_COMMON_RPC_CALL_H_

#include <grpc++/grpc++.h>

namespace nexus {

enum RpcCallStatus {
  RPC_CALL_CREATE,
  RPC_CALL_PROCESS,
  RPC_CALL_FINISH,
};

class RpcCallBase {
 public:
  RpcCallBase(grpc::ServerCompletionQueue* cq) :
      cq_(cq),
      status_(RPC_CALL_CREATE) {
  }

  virtual ~RpcCallBase() {}

  virtual void Proceed() = 0;

 protected:
  grpc::ServerCompletionQueue* cq_;
  grpc::ServerContext ctx_;
  RpcCallStatus status_;
};

#define INSTANTIATE_RPC_CALL(SERVICE, RPCCALL, REQUEST, REPLY)          \
  class RPCCALL##_Call : public RpcCallBase {                           \
   public:                                                              \
    using Handler = std::function<void(const grpc::ServerContext&,      \
                                       const REQUEST&, REPLY*)>;        \
    RPCCALL##_Call(SERVICE* service, grpc::ServerCompletionQueue* cq,   \
                   Handler handle) :                                    \
        RpcCallBase(cq),                                                \
        service_(service),                                              \
        handle_(std::move(handle)),                                     \
        responder_(&ctx_) {                                             \
      Proceed();                                                        \
    }                                                                   \
    void Proceed() {                                                    \
      if (status_ == RPC_CALL_CREATE) {                                 \
        status_ = RPC_CALL_PROCESS;                                     \
        service_->Request##RPCCALL(                                     \
            &ctx_, &request_, &responder_, cq_, cq_, this);             \
      } else if (status_ == RPC_CALL_PROCESS) {                         \
        new RPCCALL##_Call(service_, cq_, handle_);                     \
        handle_(ctx_, request_, &reply_);                               \
        status_ = RPC_CALL_FINISH;                                      \
        responder_.Finish(reply_, grpc::Status::OK, this);              \
      } else {                                                          \
        CHECK_EQ(status_, RPC_CALL_FINISH);                             \
        delete this;                                                    \
      }                                                                 \
    }                                                                   \
   private:                                                             \
    SERVICE* service_;                                                  \
    Handler handle_;                                                    \
    grpc::ServerAsyncResponseWriter<REPLY> responder_;                  \
    REQUEST request_;                                                   \
    REPLY reply_;                                                       \
  }

} // namespace nexus

#endif // NEXUS_COMMON_RPC_CALL_H_
