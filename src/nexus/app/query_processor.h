#ifndef NEXUS_APP_QUERY_PROCESSOR_H_
#define NEXUS_APP_QUERY_PROCESSOR_H_

#include "nexus/app/exec_block.h"
#include "nexus/app/request_context.h"

namespace nexus {
namespace app {

class QueryProcessor {
 public:
  QueryProcessor(std::vector<ExecBlock*> blocks) :
      blocks_(blocks) {
    std::unordered_set<int> block_ids;
    for (auto block : blocks) {
      if (block_ids.count(block->id()) > 0) {
        LOG(FATAL) << "Block id " << block->id() << " already exists";
      }
      block_ids.insert(block->id());
    }
  }

  void Process(std::shared_ptr<RequestContext> ctx) {
    if (ctx->state() == kUninitialized) {
      // LOG(INFO) << "Init req " << ctx->const_request().user_id() << ":" <<
      //     ctx->const_request().req_id();
      ctx->SetExecBlocks(blocks_);
    }
    while (!ctx->finished()) {
      auto block = ctx->NextReadyBlock();
      if (block == nullptr) {
        ctx->SetState(kBlocking);
        return;
      }
      // LOG(INFO) << "Exec req " << ctx->const_request().user_id() << ":" <<
      //     ctx->const_request().req_id() << ", block " << block->id();
      auto ret = block->Run(ctx);
      if (ctx->state() == kError) {
        break;
      }
      ctx->AddBlockReturn(ret);
    }
    // LOG(INFO) << "Reply req " << ctx->const_request().user_id() << ":" <<
    //     ctx->const_request().req_id();
    ctx->SendReply();
  }

 private:
  std::vector<ExecBlock*> blocks_;
};

} // namespace app
} // namespace nexus

#endif // NEXUS_APP_QUERY_PROCESSOR_H_
