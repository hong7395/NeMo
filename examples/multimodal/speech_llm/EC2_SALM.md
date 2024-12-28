# EC2 - SALM 추론 가이드

이 가이드는 AWS EC2 인스턴스를 세팅하고, SALM(Speech-Augmented Language Model)을 사용하여 데이터를 준비하고, 시스템을 설정하며, 추론을 실행하는 과정을 단계별로 설명합니다.

---

## 0. EC2 설정

### 0.1 인스턴스 생성
Ubuntu 22.04 인스턴스 생성

인스턴스 : g4dn.xlarge - T4 (p2.xlarge - K80, p4d.24xlarge - A100)

볼륨 : 128 + 125 GB

(루트 디스크 : gp3, 3000 IOPS, 처리량 125MB/s)

(인스턴스 스토어 볼륨 125GB 는 사용 x, 휘발성 및 백업 힘듦)

탄력적 IP 설정

### 0.2 패키지 업데이트, NVIDIA Driver 설치, 디스크 마운트 및 프로젝트 준비
#### 단계:

참고: ~/ = /home/ubuntu

1. EC2 인스턴스에 접속 후 패키지 목록을 업데이트합니다:

    ```bash
    sudo apt update && sudo apt upgrade -y
    ```

2. 기본 유틸리티 및 Python 관련 패키지를 설치합니다:

    ```bash
    sudo apt install -y build-essential wget curl git python3 python3-pip python3-venv sox libsndfile1 ffmpeg mpich
    ```

3. NVIDIA 드라이버 및 CUDA 설치:

    3-1. GPU 확인:

    ```bash
    lspci | grep -i nvidia
    ```

    3-2. gcc 설치:

    ```bash
    sudo apt install gcc -y
    ```

    3-3. Linux Kernel Header 설치:

    ```bash
    sudo apt install linux-headers-$(uname -r)
    ```

    3-4. ubuntu-drivers 설치:

    ```bash
    sudo apt install ubuntu-drivers-common -y
    ```

    3-5. ubuntu-driver를 통해 nvidia-driver 추천 버전을 확인:

    ```bash
    ubuntu-drivers devices
    ```

    3-6. 설치할 Driver 버전을 위한 Repository를 추가:

    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
    wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb

    # Repository 추가된 것을 확인
    cat /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list
    ```

    3-7. nvidia-driver 설치:

    ```bash
    sudo apt install nvidia-driver-???
    ```

    3-8. 서버 재 시작 후 Nvidia-driver 설치 확인:

    ```bash
    nvidia-smi
    ```

    3-9. CUDA 11.8 설치:

    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run
    ```

    그래픽 드라이버가 이미 설치되어 있어 오류 발생:

    `Continue 이동 후 엔터 -> accept 입력 후 엔터 -> Driver에서 스페이스바를 눌러 체크 해제 후 Install 이동 후 엔터`

    경로 설정:
    
    ```bash
    sudo nano ~/.bashrc

    # 다음 2 줄을 복사한 후, 화살표키로 .bashrc 파일 끝으로 이동한 후, 붙여넣기
    export PATH="/usr/local/cuda-11.8/bin:$PATH"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64/

    # Ctrl + O를 누른 후, 엔터를 입력하여 저장한 후, Ctrl + X를 눌려 nano 편집기를 종료

    # 환경 적용을 위해 다음 명령 실행
    source ~/.bashrc
    ```

    설치 확인:
    
    ```bash
    nvcc -V

    /usr/local/cuda-11.8/extras/demo_suite/deviceQuery
    ```

    3-10. cuDNN 설치:

    ```bash
    sudo apt-get update
    sudo apt-get -y install cudnn-cuda-11
    ```

    설치 확인:
    
    ```bash
    ldconfig -N -v $(sed 's/:/ /' <<< $LD_LIBRARY_PATH) 2>/dev/null | grep libcudnn
    ```

4. Python 가상환경 생성 및 활성화:

    4-1. 가상환경 생성:

    ```bash
    python3 -m venv ~/salm_env
    ```

    4-2. 가상환경 활성화:

    ```bash
    source ~/salm_env/bin/activate
    ```

5. NeMo 설치:

    ```bash
    git clone https://github.com/hong7395/NeMo.git

    cd NeMo

    pip install --upgrade pip setuptools wheel
    pip install Cython packaging
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    pip install nemo_toolkit['all']

    # 설치 확인
    pip show nemo_toolkit

    # 설치가 안된다면,
    pip install nemo_toolkit['asr']
    ```

    참고: Pytorch 최신(2.5.1)은 안됨, mamba-ssm 설치 오류

6. LLM / Multimodal 추가 설치:

    6-1. Apex 설치:

    ```bash
    cd ~
    git clone https://github.com/NVIDIA/apex.git
    cd apex
    pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir \
        --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam"
    ```

    6-2. Transformer Engine 설치:

    (cuda 11.8과 호환되는 0.7.0-770e968 으로 체크아웃하여 설치)

    ```bash
    cd ~
    git clone https://github.com/NVIDIA/TransformerEngine.git
    cd TransformerEngine
    git checkout 770e968
    git submodule init && git submodule update
    NVTE_FRAMEWORK=pytorch NVTE_WITH_USERBUFFERS=1 MPI_HOME=/usr pip install .
    ```

    6-3. Megatron Core (Megatron-LM) 설치:

    ```bash
    cd ~
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    pip install .
    cd megatron/core/datasets
    make
    ```

    6-4. MNeMo Text Processing 설치:

    ```bash
    cd ~
    git clone https://github.com/NVIDIA/NeMo-text-processing
    cd NeMo-text-processing
    ./reinstall.sh
    ```

## 1. 데이터셋 준비

### 1.1 데이터셋 다운로드 및 추출
LibriSpeech의 `test-clean` 데이터셋을 사용합니다.

#### 단계:
1. 데이터셋 다운로드:
    ```bash
    cd ~/NeMo/examples/multimodal/speech_llm/data
    wget http://www.openslr.org/resources/12/test-clean.tar.gz
    ```
2. 데이터셋 압축 해제:
    ```bash
    tar -xvzf test-clean.tar.gz -C ~/NeMo/examples/multimodal/speech_llm/data
    ```
### 1.2 JSONL 매니페스트 파일 생성
오디오 파일과 전사 텍스트를 매핑하는 JSONL 파일을 생성합니다.

(현재 위치 디렉토리 : ~/NeMo/examples/multimodal/speech_llm/data)

`test_manifest.jsonl` 생성 스크립트:

다음 스크립트를 `create_test_manifest.py`로 저장하세요:
```python
    import os
    import json
    from pydub.utils import mediainfo  # 오디오 길이 계산용
    from tqdm import tqdm  # 진행도 표시

    # 데이터 디렉토리 설정
    data_dir = "./LibriSpeech/test-clean"
    output_jsonl = "./test_manifest.jsonl"

    # JSONL 생성
    data = []

    # 디렉토리 내 모든 파일 확인
    all_files = [os.path.join(root, file) for root, _, files in os.walk(data_dir) for file in files if file.endswith(".trans.txt")]

    # 진행도 표시를 위한 tqdm 사용
    for transcript_path in tqdm(all_files, desc="Processing transcripts"):
        with open(transcript_path, "r") as f:
            # 'trans.txt' 파일 내 각 줄 처리
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(" ", 1)
                    audio_id = parts[0]
                    transcript = parts[1]
                    audio_file = os.path.join(os.path.dirname(transcript_path), f"{audio_id}.flac")
                    if os.path.exists(audio_file):
                        # 오디오 길이 계산
                        try:
                            info = mediainfo(audio_file)
                            duration = float(info['duration'])  # 초 단위
                        except Exception as e:
                            print(f"오디오 길이 계산 실패: {audio_file} - {e}")
                            duration = None

                        data.append({
                            "audio_filepath": audio_file,
                            "offset": 0,
                            "duration": duration,
                            "text": transcript
                        })

    # JSONL 저장
    with open(output_jsonl, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    print(f"test_manifest.jsonl 생성 완료: {output_jsonl}")
```
스크립트 실행:
```bash
    python ~/NeMo/examples/multimodal/speech_llm/data/create_test_manifest.py

    # speech_llm 디렉터리로 돌아감
    cd ..
```
## 2. 모델 준비
### 2.1 Fast Conformer 모델 다운로드
사전 학습된 `stt_en_fastconformer_transducer_large.nemo` 모델을 다운로드하여 models 폴더에 저장합니다:
```bash
    mkdir models
    wget https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_transducer_large/versions/1.0.0/files/stt_en_fastconformer_transducer_large.nemo -O ./models/stt_en_fastconformer_transducer_large.nemo
```
## 3. 설정 파일 작성
### 3.1 `salm_config.yaml` 파일
다음과 같이 설정 파일을 작성하고 `conf/salm` 폴더에 저장합니다:
```yaml
    name: salm_fastconformer_gpt_lora_tuning

    trainer:
    devices: 1
    accelerator: gpu
    num_nodes: 1
    precision: 16
    max_steps: 1000000
    gradient_clip_val: 1.0
    accumulate_grad_batches: 1

    model:
    seed: 1234
    tensor_model_parallel_size: 1
    pipeline_model_parallel_size: 1
    pretrained_audio_model: stt_en_fastconformer_transducer_large
    freeze_llm: true
    restore_from_path: "~/NeMo/examples/multimodal/speech_llm/models/stt_en_fastconformer_transducer_large.nemo"
    save_nemo_on_validation_end: false

    data:
    test_ds:
        manifest_filepath: "~/NeMo/examples/multimodal/speech_llm/data/test_manifest.jsonl"
        prompt_template: "Q: {context}\\nA: {answer}"
        tokens_to_generate: 128
        shuffle: false
        num_workers: 0
        pin_memory: true
        max_seq_length: 2048
        min_seq_length: 1
        add_eos: true
        sample_rate: 16000
        max_duration: 24
        min_duration: 0.1

    optim:
    name: fused_adam
    lr: 1e-4
    weight_decay: 0.001
    betas:
        - 0.9
        - 0.98
    sched:
        name: CosineAnnealing
        warmup_steps: 2000
        min_lr: 0.0
        constant_steps: 0
```
## 4. 추론 실행
1. 설정 파일을 이용하여 추론 실행:
    ```bash
    python ~/NeMo/examples/multimodal/speech_llm/modular_audio_gpt_eval.py --config-path=conf/salm --config-name=salm_config.yaml
    ```
2. 결과 확인:

추론 결과는 콘솔에 출력되거나 설정된 파일로 저장됩니다.



#### torch killed 시, 메모리 부족으로 스왑 메모리 추가 설정:

1. 4GB 스왑 파일 생성:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

2. 스왑 활성화 확인:

```bash
free -h
```

3. Torch 설치 재시도:

```bash
pip install torch
```

4. 설치 완료 후 스왑 비활성화(선택):

```bash
sudo swapoff /swapfile
sudo rm /swapfile
```

#### 추가 디스크 마운트 (옵션):

1. 디스크 확인:

```bash
lsblk
```

(추가 디스크 이름 및 `UUID` 확인, 이름 예시 : `nvme0n1`)

(다만 디스크 이름은 부팅 순서에 따라 변할 수 있음. nvme0n1 -> nvme1n1 이 되는 등 ..)

2. 파일 시스템 생성 (`xfs` or `ext4`):

```bash
# 파일 시스템 확인
sudo file -s /dev/nvme0n1
```

출력:

```bash
/dev/nvme0n1: data
```

```bash
# 파일 시스템 생성
sudo mkfs -t xfs /dev/nvme0n1
```

3. 디스크 마운트:

    1. 추가 디스크를 마운트할 디렉토리 생성:

    ```bash
    sudo mkdir /mnt/storage
    ```

    2. 디스크 마운트:

    ```bash
    sudo mount /dev/nvme0n1 /mnt/storage
    ```

    3. 마운트 확인:
    
    ```bash
    df -h
    ```

    `/mnt/storage` 에 추가 디스크 용량이 표시되면 성공

4. 영구 마운트 설정 (재부팅 후에도 유지):

    1. `/etc/fstab` 파일에 추가:

    ```bash
    echo 'UUID={추가 디스크 UUID 기입} /mnt/storage xfs defaults,nofail 0 2' | sudo tee -a /etc/fstab
    ```

    2. 설정 적용:

    ```bash
    sudo mount -a
    ```

5. 디렉토리 권한 변경:

    디렉토리를 생성한 후, 현재 사용자 계정이 해당 디렉토리를 자유롭게 사용할 수 있도록 소유권을 변경합니다:

    1. 디렉토리 소유권 변경:

    ```bash
    sudo chown -R $USER:$USER /mnt/storage
    ```

    2. 권한 확인:

    ```bash
    ls -ld /mnt/storage
    ```