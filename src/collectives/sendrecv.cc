/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "nccl.h"
#include "json.h"
#include "collectives.h"
#include <fstream>
#include "argcheck.h" // Need some checks here since we access comm

struct NvtxParamsSendRecv {
    size_t bytes;
    int peer;
};
constexpr const nvtxPayloadSchemaEntry_t SendRecvSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Bytes"},
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Peer rank", nullptr, 0, offsetof(NvtxParamsSendRecv, peer)}
};

NCCL_API(ncclResult_t, ncclSend, const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  globalCounter_++;
  std::ifstream inputFile("data.json");
  std::string jsonData((std::istreambuf_iterator<char>(inputFile)),
                         std::istreambuf_iterator<char>());
  Json::Reader reader;
  Json::Value jsonroot;
  std::string errs;
  reader.parse(jsonData, jsonroot);
  int sleep_time = jsonroot[std::to_string(globalCounter_)].asInt();

  sleep(sleep_time); 
  ncclResult_t ret = ncclSuccess;
  return ret;
}

NCCL_API(ncclResult_t, ncclRecv, void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer,
    ncclComm_t comm, cudaStream_t stream) {
  globalCounter_++;
  std::ifstream inputFile("data.json");
  std::string jsonData((std::istreambuf_iterator<char>(inputFile)),
                         std::istreambuf_iterator<char>());
  Json::Reader reader;
  Json::Value jsonroot;
  std::string errs;
  reader.parse(jsonData, jsonroot);
  int sleep_time = jsonroot[std::to_string(globalCounter_)].asInt();

  sleep(sleep_time); 
  ncclResult_t ret = ncclSuccess;
  return ret;
}
