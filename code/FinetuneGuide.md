本项目使用[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/README_zh.md)进行模型微调，下面是搭建Llama-factory环境的步骤。

### 过程记录
#### 环境
首先按Llama-Factory 的要求安装
```shell
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
installed bitsandbytes through pip
```shell
pip install bitsandbytes
```
installed flash-attention through pip
```shell
pip install flash-attention
```
and encountered an error, so download .whl from [here](https://github.com/bdashore3/flash-attention/releases)
```shell
pip install dependency\flash_attn-2.7.1.post1+cu124torch2.5.1cxx11abiFALSE-cp312-cp312-win_amd64.whl
```
>doc中提到的可选的额外依赖项：torch、torch-npu、metrics、deepspeed、liger-kernel、bitsandbytes、hqq、eetq、gptq、awq、aqlm、vllm、galore、badam、adam-mini、qwen、modelscope、openmind、swanlab、quality 目前没有格外安装

deepspeed for windows仅支持推理并需要安装VS2019，所以以后再说 #TODO


#### 数据
| 数据集 | 数据量 | 数据来源 | 
| --- | --- | --- |
|网文写作|50k|[link](https://huggingface.co/datasets/zxbsmk/webnovel_cn)|

clone了仓库并把json文件复制到`dependency\LLaMA-Factory\data`文件夹下
由于该数据集就是alpaca格式所以不用转换
在`dependency\LLaMA-Factory\data\dataset_info.json`中添加了数据集信息

#### FineTune
1. 首先尝试使用`qwen2.5-1.5b-instruct`+qwen[文档]([qwen2.5-1.5b-instruct](https://qwen.readthedocs.io/zh-cn/latest/training/SFT/llama_factory.html))中推荐的使用方法进行lora微调
没有更改torchrun的参数，~~直接运行~~用gitbash运行
```bash
bash code/finetune.sh
```
参数缺少必要值，调整参数后重新运行
再次报错，重新安装了torch
```shell
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```
遇到"DLL load failed while importing flash_attn_2_cuda: 找不到指定的程序。"
1. 搜到可能是cuda&torch 不匹配: 重新安装了CUDA~~12.1~~12.4, 顺便更新了torch2.4.0
torch 安装后找不到指定的模块
按官网安装torch2.5.1, print(torch.version.cuda)==12.4
由于flashattention这个版本不支持python3.12,所以降级到3.11(看错了，其实是支持的)
出问题，换回3.12
error:
`use_libuv was requested but PyTorch was build without libuv support`
转换思路，使用powershell 成功启动Llama-Factory cli, 可以进行推理
**理论上应该可以训练！**
但是爆显存了，下次试试量化的qwen

##### 量化模型
下载了qwen2.5-1.5b-instruct-AWQ
要安装auto-awq及其依赖triton
```shell
pip install triton
pip install autoawq
```
BOT环境目前不太行，换成train环境，BOT有时间改一下
triton去[huggingface](https://hf-mirror.com/madbuda/triton-windows-builds)找到了windows版本的whl文件
运行，bitsandbytes有问题：
```log
RuntimeError: Failed to import transformers.integrations.bitsandbytes because of the following error (look up to see its traceback):
DLL load failed while importing libtriton: 动态链接库(DLL)初始化例程失败。
```
尝试安装bitsandbytes-windows
`pip install bitsandbytes-windows`
同时卸载bitsandbytes
运行到`model.generate()`出现问题：autoawq```UnboundLocalError: cannot access local variable 'user_has_been_warned' where it is not associated with a value```
首先尝试重装autoawq,没有效果
更改[文件](BOTtrain\Lib\site-packages\awq\modules\linear\gemm.py", line 71),添加
```python
global user_has_been_warned
```
可以运行，爆warning`UserWarning: Using naive (slow) implementation.No module named 'awq_ext'`
成功生成
进行完整安装`pip install autoawq[kernels]`
成功生成，没有报错