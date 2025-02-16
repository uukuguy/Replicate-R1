# Replicate DeepSeek-R1 on Small Language Model

```bash
# torch==2.5.1
# transformers==4.48.1
# trl==0.14.0
# vllm==0.7.2
# flashinfer-python==0.2.1.post1
git clone https://github.com/flashinfer-ai/flashinfer.git
pip install -v .
# Disable FlashInfer (Temporary)
# from vllm import _custom_ops as ops
# ops.set_attn_backend(ops.AttnBackend.XFORMERS)  # Or another supported backend

# bitsandbytes==0.45.3.dev0
# pip install git+https://github.com/bitsandbytes-foundation/bitsandbytes.git
# Build bitsandbytes from source
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
cmake -DCOMPUTE_BACKEND=cuda -DCUDA_VERSION=123 -S .
make
pip install -v .
export BNB_CUDA_VERSION=123

# xformers==0.0.28.post2
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

# Check
pip -m bitsandtypes
pip -m xformers.info

```

```bash
# Enable Software Collections (SCL)
sudo yum install centos-release-scl

# Install DevToolset (e.g., GCC 11)
sudo yum install devtoolset-11-gcc devtoolset-11-gcc-c++
scl enable devtoolset-11 bash
```

## Best Practice

### Mini-R1

[Mini-R1: Reproduce Deepseek R1 „aha moment“ a RL tutorial](https://www.philschmid.de/mini-deepseek-r1)

### Unsloth-GRPO

[Run DeepSeek R1 Dynamic 1.58-bit](https://unsloth.ai/blog/deepseekr1-dynamic?continueFlag=5ddc281ae3bb0b39401e562a1112ccc5)

### oat-zero

[There May Not be Aha Moment in R1-Zero-like Training — A Pilot Study](https://oatllm.notion.site/oat-zero) [Code](https://github.com/sail-sg/oat-zero)

### Logic_RL

[Deepseek R1 Zero成功复现, 三阶段RL，Response长度涨幅超50%，涌现语言混杂，double-check](https://mp.weixin.qq.com/s/2nQ08yLafXp19qTLWcBqNA) [Code](https://github.com/Unakar/Logic-RL)
