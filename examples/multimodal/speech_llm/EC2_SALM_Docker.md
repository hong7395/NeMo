# EC2 - SALM 추론 가이드

이 가이드는 AWS EC2 인스턴스를 세팅하고, Docker를 사용하여 NeMo 환경을 세팅하고, SALM(Speech-Augmented Language Model)을 사용하여 데이터를 준비하고, 시스템을 설정하며, 추론을 실행하는 과정을 단계별로 설명합니다.

---

## 0. EC2 설정

### 0.1 인스턴스 생성
Ubuntu 22.04 인스턴스 생성

AMI : Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)

-> NVIDIA Driver, Docker, NVIDIA Container Toolkit 등 설치

인스턴스 : g4dn.xlarge - T4 (p2.xlarge - K80, p4d.24xlarge - A100)

볼륨 : 128 + 125 GB

(루트 디스크 : gp3, 3000 IOPS, 처리량 125MB/s)

(인스턴스 스토어 볼륨 125GB 는 사용 x, 휘발성 및 백업 힘듦)

탄력적 IP 설정

보안 그룹 : 인바운드 규칙 편집 - 8888, 6006 포트 추가

ssh : ssh-keygen -R ec2-???.ap-northeast-2.compute.amazonaws.com

### 0.2 패키지 업데이트, 도커 준비, 프로젝트 준비
#### 단계:

참고: ~/ = /home/ubuntu

1. EC2 인스턴스에 접속 후 패키지 목록 업데이트 및 GPU 확인:

    ```bash
    sudo apt update && sudo apt upgrade -y
    ```

    ```bash
    nvidia-smi
    ```

2. Docker 설치 확인:

    ```bash
    docker run --gpus all --rm nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
    ```

3. NeMo 전용 컨테이너 실행:

    3-0. NeMo 프로젝트 클론:

    ```bash
    cd ~
    git clone https://github.com/hong7395/NeMo.git
    ```

    3-1. NeMo 컨테이너 다운로드:

    ```bash
    docker pull nvcr.io/nvidia/nemo:24.05
    ```

    3-2. 컨테이너 실행:

    ```bash
    docker run --gpus all -it --rm -v ~/NeMo:/NeMo --shm-size=8g -p 8888:8888 -p 6006:6006 --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/nemo:24.05
    ```

    3-3. 컨테이너 내에서 NeMo 확인:

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
1. 데이터 디렉토리 생성:
    ```bash
    mkdir -p /NeMo/examples/multimodal/speech_llm/data
    cd /NeMo/examples/multimodal/speech_llm/data
    ```

2. 데이터셋 다운로드:
    ```bash
    wget http://www.openslr.org/resources/12/test-clean.tar.gz
    ```

3. 데이터셋 압축 해제:
    ```bash
    tar -xvzf test-clean.tar.gz
    ```

### 1.2 JSONL 매니페스트 파일 생성
오디오 파일과 전사 텍스트를 매핑하는 JSONL 파일을 생성합니다.

1. 매니페스트 생성 스크립트 작성

2. 스크립트 실행:
    ```bash
    python create_test_manifest.py
    ```

3. 매니페스트 확인:
    ```bash
    head -n 1 test_manifest.jsonl
    ```

4. NeMo 디렉터리로 돌아가기:
    ```bash
    cd /NeMo
    ```

## 2. 모델 준비

### 2.1 모델 디렉토리 생성
```bash
mkdir -p /NeMo/examples/multimodal/speech_llm/models
```

### 2.2 모델 다운로드 (선택 사항)
NGC에서 직접 모델을 사용하거나, 로컬에 다운로드할 수 있습니다:

```bash
cd /NeMo/examples/multimodal/speech_llm/models
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/speechllm_fc_llama2_7b/1.23.1/files?redirect=true&path=speechllm_fc_llama2_7b.nemo' -O speechllm_fc_llama2_7b.nemo
```

## 3. 추론 실행

### 3.1 기본 설정
1. 테스트 데이터셋 매니페스트 경로 설정:
```bash
cd /NeMo
TEST_MANIFESTS="[/NeMo/examples/multimodal/speech_llm/data/test_manifest.jsonl]"
TEST_NAMES="[librispeech-test-clean]"
```

2. 출력 디렉토리 생성:
```bash
mkdir -p /NeMo/examples/multimodal/speech_llm/test_outputs
```

3. 필요한 패키지 설치
```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install Cython packaging
pip install nemo_toolkit['all']

# Lightning 관련 패키지 설치
pip install -r /NeMo/requirements/requirements_lightning.txt

# ASR 관련 패키지 설치 (lhotse 포함)
pip install -r /NeMo/requirements/requirements_asr.txt

# 멀티모달 패키지 설치
pip install -r /NeMo/requirements/requirements_multimodal.txt

# OpenCV 버전 확인 및 설치
pip uninstall -y opencv-python opencv-python-headless
pip install opencv-python==4.8.0.76 opencv-python-headless==4.8.0.76

# Transformers 버전 확인 및 설치
pip uninstall -y transformers
pip install transformers==4.35.0

# HuggingFace Hub 업그레이드
pip install --upgrade huggingface_hub
```

### 3.2 NGC 모델을 사용한 추론
NGC에서 제공하는 사전 학습된 모델을 사용하여 추론을 실행합니다:

```bash
cd /NeMo
CUDA_VISIBLE_DEVICES=0 python /NeMo/examples/multimodal/speech_llm/modular_audio_gpt_eval.py \
  model.from_pretrained="speechllm_fc_llama2_7b" \
  +model.data.test_ds.manifest_filepath=$TEST_MANIFESTS \
  +model.data.test_ds.names=$TEST_NAMES \
  model.data.test_ds.global_batch_size=1 \
  model.data.test_ds.micro_batch_size=1 \
  model.data.test_ds.tokens_to_generate=256 \
  ++inference.greedy=False \
  ++inference.top_k=50 \
  ++inference.top_p=0.95 \
  ++inference.temperature=0.4 \
  ++inference.repetition_penalty=1.2 \
  +model.data.test_ds.output_dir="/NeMo/examples/multimodal/speech_llm/test_outputs"
```

### 3.3 로컬 모델을 사용한 추론
로컬에 다운로드한 .nemo 파일을 사용하여 추론을 실행합니다:

```bash
cd /NeMo
CUDA_VISIBLE_DEVICES=0 python /NeMo/examples/multimodal/speech_llm/modular_audio_gpt_eval.py \
    model.restore_from_path="/NeMo/examples/multimodal/speech_llm/models/speechllm_fc_llama2_7b.nemo" \
    model.data.test_ds.manifest_filepath=$TEST_MANIFESTS \
    model.data.test_ds.names=$TEST_NAMES \
    model.data.test_ds.global_batch_size=1 \
    model.data.test_ds.micro_batch_size=1 \
    model.data.test_ds.tokens_to_generate=256 \
    ++inference.greedy=False \
    ++inference.top_k=50 \
    ++inference.top_p=0.95 \
    ++inference.temperature=0.4 \
    ++inference.repetition_penalty=1.2 \
    ++model.data.test_ds.output_dir="/NeMo/examples/multimodal/speech_llm/test_outputs"
```

### 3.4 추론 매개변수 설명
- `global_batch_size`: 전체 배치 크기 (T4 GPU에서는 1로 설정)
- `micro_batch_size`: GPU당 배치 크기 (T4 GPU에서는 1로 설정)
- `tokens_to_generate`: 생성할 최대 토큰 수
- `greedy`: 탐욕적 디코딩 사용 여부
- `top_k`: 상위 k개의 토큰만 고려
- `top_p`: 누적 확률이 p가 될 때까지의 토큰만 고려
- `temperature`: 출력 분포의 온도 (높을수록 더 다양한 출력)
- `repetition_penalty`: 반복 방지를 위한 페널티

### 3.5 결과 확인
추론이 완료되면 다음과 같이 결과를 확인할 수 있습니다:

1. 콘솔 출력 확인:
   - 각 오디오에 대한 추론 결과가 실시간으로 출력됩니다
   - 전체 데이터셋에 대한 평가 메트릭도 표시됩니다

2. 출력 파일 확인:
```bash
ls -l /NeMo/examples/multimodal/speech_llm/test_outputs/
cat /NeMo/examples/multimodal/speech_llm/test_outputs/predictions_librispeech-test-clean.txt  # 예측 결과 확인
```

### 3.6 문제 해결
- GPU 메모리 부족 시:
  - `micro_batch_size`를 1로 설정하세요 (T4 GPU에서는 필수)
  - `tokens_to_generate` 값을 줄여보세요 (128 또는 64로)
  
- 추론 속도가 느린 경우:
  - `greedy=True`로 설정해보세요
  - 더 작은 데이터셋으로 테스트해보세요 (처음 10개 항목만)

- 결과가 만족스럽지 않은 경우:
  - `temperature` 값을 조정해보세요 (0.1-1.0 사이)
  - `repetition_penalty` 값을 조정해보세요

- 모델 로드 오류 발생 시:
  - NGC에서 직접 모델을 사용하는 방식(`model.from_pretrained`)을 시도해보세요
  - 로컬 모델 경로가 정확한지 확인하세요