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
1. 데이터셋 다운로드:
    ```bash
    cd /NeMo/examples/multimodal/speech_llm/data
    wget http://www.openslr.org/resources/12/test-clean.tar.gz
    ```
2. 데이터셋 압축 해제:
    ```bash
    tar -xvzf test-clean.tar.gz -C /NeMo/examples/multimodal/speech_llm/data
    ```
### 1.2 JSONL 매니페스트 파일 생성
오디오 파일과 전사 텍스트를 매핑하는 JSONL 파일을 생성합니다.

(현재 위치 디렉토리 : /NeMo/examples/multimodal/speech_llm/data)

`test_manifest.jsonl` 생성 스크립트 실행:
```bash
    python /NeMo/examples/multimodal/speech_llm/data/create_test_manifest.py

    # speech_llm 디렉터리로 돌아감
    cd ..
```
## 2. 모델 준비
NGC(클라우드) 에서 사전 학습 모델 사용:
```yaml
model:
  from_pretrained: "speechllm_fc_llama2_7b"  # pretrained model name on NGC or HF
```

혹은 로컬 모델 사용:
```bash
    mkdir models
    wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/speechllm_fc_llama2_7b/1.23.1/files?redirect=true&path=speechllm_fc_llama2_7b.nemo' -O ./models/speechllm_fc_llama2_7b.nemo
```
```yaml
model:
  restore_from_path: "/NeMo/examples/multimodal/speech_llm/models/speechllm_fc_llama2_7b.nemo" # Path to an existing .nemo model you wish to add new tasks to or run inference with
```

## 3. 추론 실행
1. 설정 파일을 이용하여 추론 실행:
    ```bash
    python /NeMo/examples/multimodal/speech_llm/modular_audio_gpt_eval.py --config-path=conf --config-name=modular_audio_gpt_config_eval.yaml
    ```
2. 결과 확인:

추론 결과는 콘솔에 출력되거나 설정된 파일로 저장됩니다.