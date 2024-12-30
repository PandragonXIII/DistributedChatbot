# conda create -n BOTtrain python=3.12
conda activate BOTtrain
pip install torch==2.5.1+cu124
pip install torchaudio==2.5.1+cu124
pip install torchvision==0.20.1+cu124
# check if CUDA version==12.4, if not, exit
$CUDA_ver = python -c "import torch; print(torch.version.cuda)"
if ($CUDA_ver -ne "12.4") {
    echo "CUDA version is not 12.4, please check your CUDA version"
    pause
    exit
}
# Relevant link is included in requirements.txt
# download special windows package whl
# curl -o dependency/flash_attn-2.7.1.post1+cu124torch2.5.1cxx11abiFALSE-cp312-cp312-win_amd64.whl https://github.com/bdashore3/flash-attention/releases/download/v2.7.1.post1/flash_attn-2.7.1.post1+cu124torch2.5.1cxx11abiFALSE-cp312-cp312-win_amd64.whl
# wget -o dependency/triton-windows-builds/resolve/main/triton-3.0.0-cp312-cp312-win_amd64.whl https://hf-mirror.com/madbuda/triton-windows-builds/resolve/main/triton-3.0.0-cp312-cp312-win_amd64.whl
# install all requirements
pip install -r BOTtrain_requirements.txt
# install requirements for llama-factory
cd libs/LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ../..
# and this should work

# download dataset
# webnovel
wget -c -o libs/LLaMA-Factory/data/novel_cn_token512_50k.json https://hf-mirror.com/datasets/zxbsmk/webnovel_cn/resolve/main/novel_cn_token512_50k.json?download=true