/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "nccl.h"
#include "json.h"
#include "collectives.h"
#include <fstream>

NCCL_API(ncclResult_t, ncclReduceScatter, const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  globalCounter_++;
  std::ifstream inputFile("data.json");
  std::string jsonData((std::istreambuf_iterator<char>(inputFile)),
                         std::istreambuf_iterator<char>());
  Json::Reader reader;
  Json::Value root;
  std::string errs;
  reader.parse(jsonData, root);
  int sleep_time = root[std::to_string(globalCounter_)].asInt();

  sleep(sleep_time); 
  ncclResult_t ret = ncclSuccess;
  return ret;
}
