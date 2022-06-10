/*
All modification made by Cambricon Corporation: © 2018 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2018, the respective contributors
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_COMMAND_OPTION_HPP_
#define CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_COMMAND_OPTION_HPP_
#include <gflags/gflags.h>

DECLARE_string(offlinemodel);
DECLARE_string(meanfile);
DECLARE_string(meanvalue);
DECLARE_int32(core_number);
DECLARE_int32(batchsize);
DECLARE_int32(threads);
DECLARE_int32(channel_dup);
DECLARE_int32(simple_compile);
DECLARE_string(images);
DECLARE_string(dataset_path);
DECLARE_string(labels);
DECLARE_int32(fix8);
DECLARE_int32(int8);
DECLARE_int32(yuv);
DECLARE_int32(rgb);
DECLARE_int32(interpolation);
DECLARE_double(scale);
DECLARE_string(model);
DECLARE_string(weights);
DECLARE_int32(dump);
DECLARE_string(mmode);
DECLARE_string(mcore);
DECLARE_string(datastrategy);
DECLARE_int32(fifosize);
DECLARE_double(confidence);
DECLARE_string(outputdir);
DECLARE_string(logdir);
DECLARE_string(mludevice);
DECLARE_string(resize);
DECLARE_int32(apiversion);
DECLARE_string(functype);
DECLARE_int32(async_copy);
DECLARE_int32(input_format);
DECLARE_int32(dim_mutable);
DECLARE_int32(dim_mutable_max_batch);
DECLARE_int32(preprocess_method);
DECLARE_int32(timestamp);
DECLARE_int32(perf_mode);
DECLARE_int32(perf_mode_img_num);


#endif  // CATCH_EXAMPLES_OFFLINE_COMMON_INCLUDE_COMMAND_OPTION_HPP_
