# EC2 - SALM 추론 가이드

이 가이드는 AWS EC2 인스턴스를 세팅하고, Docker를 사용하여 NeMo 환경을 세팅하고, SALM(Speech-Augmented Language Model)을 사용하여 데이터를 준비하고, 시스템을 설정하며, 추론을 실행하는 과정을 단계별로 설명합니다.

---

## 0. EC2 설정

### 0.1 인스턴스 생성
Ubuntu 22.04 인스턴스 생성

인스턴스 : g4dn.xlarge - T4 (p2.xlarge - K80, p4d.24xlarge - A100)

볼륨 : 128 + 125 GB

(루트 디스크 : gp3, 3000 IOPS, 처리량 125MB/s)

(인스턴스 스토어 볼륨 125GB 는 사용 x, 휘발성 및 백업 힘듦)

탄력적 IP 설정

ssh : ssh-keygen -R ec2-???.ap-northeast-2.compute.amazonaws.com

### 0.2 패키지 업데이트, NVIDIA Driver 설치, 프로젝트 준비
#### 단계:

참고: ~/ = /home/ubuntu

1. EC2 인스턴스에 접속 후 패키지 목록을 업데이트합니다:

    ```bash
    sudo apt update && sudo apt upgrade -y
    ```

2. 기본 유틸리티 및 Python 관련 패키지를 설치합니다:

    ```bash
    sudo apt install -y build-essential dkms
    ```

3. NVIDIA 드라이버 설치:

    3-1. GPU 확인:

    ```bash
    lspci | grep -i nvidia
    ```

    3-2. Linux Kernel Header 설치:

    ```bash
    sudo apt install linux-headers-$(uname -r)
    ```

    3-3. ubuntu-drivers 설치:

    ```bash
    sudo apt install ubuntu-drivers-common -y
    ```

    3-4. 설치할 Driver 버전을 위한 Repository를 추가:

    ```bash
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
    wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb

    # Repository 추가된 것을 확인
    cat /etc/apt/sources.list.d/cuda-ubuntu2204-x86_64.list
    ```

    3-5. ubuntu-driver를 통해 nvidia-driver 추천 버전을 확인:

    ```bash
    ubuntu-drivers devices
    ```

    3-6. nvidia-driver 설치:

    ```bash
    sudo apt install nvidia-driver-535
    ```

    3-8. 서버 재 시작 후 Nvidia-driver 설치 확인:

    ```bash
    sudo reboot

    nvidia-smi
    ```

4. Docker 설치 및 NVIDIA Container Toolkit 설정:

    4-1. Docker 설치:

    ```bash
    sudo apt install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker $USER

    sudo reboot
    ```

    4-2. NVIDIA Container Toolkit 설치:

    ```bash
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt install -y nvidia-container-toolkit
    sudo systemctl restart docker
    ```

    4-3. 설치 확인:

    ```bash
    docker run --gpus all --rm nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
    ```

5. NeMo 전용 컨테이너 실행:

    5-0. NeMo 프로젝트 클론:

    ```bash
    cd ~
    git clone https://github.com/hong7395/NeMo.git
    ```

    5-1. NeMo 컨테이너 다운로드:

    ```bash
    docker pull nvcr.io/nvidia/nemo:24.05
    ```

    5-2. 컨테이너 실행:

    ```bash
    docker run --gpus all -it --rm -v ~/NeMo:/NeMo --shm-size=8g \ 
    -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit \ 
    stack=67108864 --device=/dev/snd nvcr.io/nvidia/nemo:24.05
    ```

    5-3. 컨테이너 내에서 NeMo 확인:

    ```bash
    python -c "import nemo; print(nemo.__version__)"
    ```

    참고:

    ```bash
    docker ps -a  # 실행된 적 있는 컨테이너 목록 확인
    docker start -ai <컨테이너_ID 또는 이름>  #중지된 컨테이너 재시작
    docker exec -it <컨테이너 ID 또는 이름> bash  #실행중인 컨테이너에 접속
    docker stop <컨테이너_ID 또는 이름>  # 컨테이너 중지
    docker rm <컨테이너_ID 또는 이름>  # 컨테이너 삭제/종료

    docker images  # 현재 도커 이미지 확인
    docker rmi <IMAGE_ID>  # 도커 이미지 삭제
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