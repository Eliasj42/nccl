/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "nccl.h"
#include "json.h"
#include <iostream>
#include <fstream>

#ifndef COUNTER_INIT_H
#define COUNTER_INIT_H

int globalCounter_ = 0;

#endif

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  globalCounter_++;
  std::ifstream inputFile("data.json");
  std::string jsonData((std::istreambuf_iterator<char>(inputFile)),
                         std::istreambuf_iterator<char>());
  Json::Reader reader;
  Json::Value root;
  std::string errs;
  reader.parse(jsonData, root);
  std::cout << "The value of myVariable is: " << root[std::to_string(globalCounter_)].asInt() << std::endl;
  int sleep_time = root[std::to_string(globalCounter_)].asInt();
  //int sleep_time = globalCounter_;

  sleep(sleep_time); 
  ncclResult_t ret = ncclSuccess;
  return ret;
}
