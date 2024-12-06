# EC2 - SALM 추론 가이드

이 가이드는 AWS EC2 인스턴스를 세팅하고, SALM(Speech-Augmented Language Model)을 사용하여 데이터를 준비하고, 시스템을 설정하며, 추론을 실행하는 과정을 단계별로 설명합니다.

---

## 0. EC2 설정

### 0.1 인스턴스 생성
Ubuntu 22.04 인스턴스 생성 (테스트용으로 프리 티어 생성)
스토리지 용량 16G 설정

### 0.2 패키지 업데이트
#### 단계:
1. EC2 인스턴스에 접속 후 패키지 목록을 업데이트합니다:
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```
2. 기본 유틸리티 및 Python 관련 패키지를 설치합니다:
    ```bash
    sudo apt install -y build-essential wget curl git python3 python3-pip python3-venv sox
    ```
3. Python 가상환경 생성 및 활성화:
    ```bash
    python3 -m venv salm_env
    source salm_env/bin/activate
    ```
4. NeMo GitHub 리포지토리 클론
    ```bash
    git clone https://github.com/hong7395/NeMo.git
    cd NeMo/examples/multimodal/speech_llm
    ```
5. 필수 라이브러리 설치
    ```bash
    pip install numpy
    pip install -r requirements.txt -r requirements_asr.txt -r requirements_common.txt
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
    wget https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_transducer_large/versions/1.0.0/files/stt_en_fastconformer_transducer_large.nemo -O ./models/stt_en_fastconformer_transducer_large.nemo
    ```
## 3. 설정 파일 작성
### 3.1 `salm_config_cpu.yaml` 파일
다음과 같이 설정 파일을 작성하고 `conf` 폴더에 저장합니다:

    ```yaml
    name: salm_fastconformer_gpt_lora_tuning

    trainer:
    devices: 1
    accelerator: cpu
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
    restore_from_path: "./examples/multimodal/speech_llm/models/stt_en_fastconformer_transducer_large.nemo"
    save_nemo_on_validation_end: false

    data:
    test_ds:
        manifest_filepath: "./examples/multimodal/speech_llm/data/test_manifest.jsonl"
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
    python ./modular_audio_gpt_eval.py --config-path=conf --config-name=salm_config_cpu.yaml
    ```
2. 결과 확인:

추론 결과는 콘솔에 출력되거나 설정된 파일로 저장됩니다.

### 추가 참고 사항
필요 라이브러리 설치:
    ```bash
    pip install pydub soundfile tqdm
    JSONL 파일과 모델 경로가 정확히 설정되었는지 확인하세요.
    ```
    
위 가이드는 SALM 모델을 사용한 추론을 실행하기 위한 전체 과정을 포함합니다. 추가 질문이 있으면 언제든 말씀해주세요!