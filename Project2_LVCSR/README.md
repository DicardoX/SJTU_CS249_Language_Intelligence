# Experiments on LVCSR System 

#### Course Project of SJTU Intelligence Language Technology

--------





## 1. `Kaldi` 的安装

[【语音识别工具kaldi在linux环境下的安装步骤】](https://blog.csdn.net/qq_36511757/article/details/77920849)

- 环境：`Linux-20.04`

- 新建 `Kaldi` 文件夹，并使用 `git clone https://github.com/kaldi-asr/kaldi.git` 指令下载 `Kaldi` 相关文件

- 进入 `./kaldi/tools/extras`，使用 `./check_dependencies.sh` 工具检查依赖，并安装缺少的软件工具

- 退回到 `./kaldi/tools` 目录下，使用 `make -j 6` 进行编译，数字表示CPU数量，可以加速编译

- 进入 `./kaldi/src` 目录下，依次调用如下命令：

    ```shell
    ./configure
    make depend
    make
    ```

---------



## 2. 简单运行

- 进入 `./kaldi/egs/aishell/s5`，进行相关的适配：

    - 修改 `cmd.sh`：将 `queue.pl` 修改为 `run.pl`，以在本机运行脚本；同时根据主机配置修改相应的内存限制，以最大化运行性能

        ```shell
        export train_cmd="run.pl --mem 12G"
        export decode_cmd="run.pl --mem 12G"
        export mkgraph_cmd="run.pl --mem 12G"
        ```

    - 修改 `run.sh`：修改脚本的并行进程数 `--nj`，设置为 `$(nproc)` 字段，以获取本机的CPU数目

        ```shell
        --nj $(nproc)
        ```

        同时，修改 `run.sh` 中的数据集来源，并注释掉在线下载数据集的指令，使用预下载的本地数据集

        ```shell
        data=/home/dicardo/Kaldi/kaldi/dataset
        data_url=www.openslr.org/resources/33
        
        . ./cmd.sh
        
        # local/download_and_untar.sh $data $data_url data_aishell || exit 1;
        # local/download_and_untar.sh $data $data_url resource_aishell || exit 1;
        ```

    - 替换 `run_tdnn.sh` 脚本：原因是官方 recipe 中 DNN-HMM 模型的训练 在 CPU 上运行时间过于漫长且占用内存过大，脚本位于 `./kaldi/egs/aishell/s5/local/nnet3/run_tdnn.sh`；同时，需要在 `run_ivector_common.sh` 脚本执行到阶段四前添加 `exit 0`，原因是修改后的 `run_tdnn.sh` 脚本在阶段四后是无用的。

- 运行 `./run.sh`：

    尝试运行 `./run.sh` 报错：

    ```
    local/aishell_train_lms.sh: train_lm.sh is not found. That might mean it's not installed
    ```

    进入 `./kaldi/tools/extras`，分别运行：

    ```shell
    ./install_kaldi_lm.sh
    source env.sh
    ```









```
run.pl: job failed, log is in exp/mono/decode_dev/log/analyze_alignments.log
```

[~~kaldi训练aishell模型时遇到run.pl:job failed, log is in exp/nnet3/tdnn_sp/log/train.13.2.log](https://blog.csdn.net/qq_40212975/article/details/105485313)~~

~~把local/nnet3/run_tdnn.sh 的num_jobs_final和num_jobs_initial的值都改为1~~

查看`./kaldi/egs/aishell/s5/exp/mono/decode_dev/log/analyze_alignments.log`

```shell
# gunzip -c exp/mono/decode_dev/phone_stats.*.gz | steps/diagnostic/analyze_phone_length_stats.py exp/mono/graph 
# Started at Fri Jun 25 13:39:51 CST 2021
#
/usr/bin/env: 'python': No such file or directory
# Accounting: time=0 threads=1
# Ended (code 127) at Fri Jun 25 13:39:51 CST 2021, elapsed time 0 seconds
```

`apt install python`



```
Exception: Command exited with status 1: steps/nnet3/get_egs.sh                  --cmd "run.pl --mem 12G"
```

参考[指导](https://docs.qq.com/doc/DR2JVeXlXZ3BLQXhP)





## 3. Kaldi脚本解释

[【Kaldi中每个脚本的简单解释】](http://blog.360converter.com/archives/11621)

- `steps/train_mono.sh`：单音素模型训练
- `utils/mkgraph.sh`：用于解码
- `steps/make_mfcc_pitch.sh`：提取MFCC和pitch特征
- `steps/compute_cmvn_stats.sh`：提取倒谱特征，语音识别时用
- `utils/fix_data_dir.sh`：数据规整

#### 3.1 特征提取

`make_mfcc_pitch.sh`

配置文件：`./kaldi/egs/aishell/s5/conf/mfcc.conf` & `./kaldi/egs/aishell/s5/conf/pitch.conf`



