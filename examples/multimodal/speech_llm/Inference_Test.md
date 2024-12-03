# Creating a comprehensive guide for SALM inference in Korean

# SALM 추론 가이드

이 가이드는 SALM(Speech-Augmented Language Model)을 사용하여 데이터를 준비하고, 시스템을 설정하며, 추론을 실행하는 과정을 단계별로 설명합니다.

---

## 1. 데이터셋 준비

### 1.1 데이터셋 다운로드 및 추출
LibriSpeech의 `test-clean` 데이터셋을 사용합니다.

#### 단계:
1. 데이터셋 다운로드:
    ```bash
    wget http://www.openslr.org/resources/12/test-clean.tar.gz
    ```
2. 데이터셋 압축 해제:
    ```bash
    tar -xvzf test-clean.tar.gz -C ./examples/multimodal/speech_llm/data
    ```
### 1.2 JSONL 매니페스트 파일 생성
오디오 파일과 전사 텍스트를 매핑하는 JSONL 파일을 생성합니다.

`test_manifest.jsonl` 생성 스크립트:
다음 스크립트를 `create_test_manifest.py`로 저장하세요:

    ```python
    import os
    import json
    from pydub.utils import mediainfo

    # LibriSpeech test-clean 데이터셋 디렉토리 설정
    data_dir = "./examples/multimodal/speech_llm/data/test-clean"
    output_jsonl = "./examples/multimodal/speech_llm/data/test_manifest.jsonl"

    # JSONL 파일 생성
    data = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".trans.txt"):
                transcript_path = os.path.join(root, file)
                with open(transcript_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split(" ", 1)
                            audio_id = parts[0]
                            transcript = parts[1]
                            audio_file = os.path.join(root, f"{audio_id}.flac")
                            if os.path.exists(audio_file):
                                info = mediainfo(audio_file)
                                duration = float(info.get('duration', 0))
                                data.append({
                                    "audio_filepath": audio_file,
                                    "offset": 0,
                                    "duration": duration,
                                    "text": transcript
                                })

    # JSONL 파일 저장
    with open(output_jsonl, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\\n")

    print(f"test_manifest.jsonl 생성 완료: {output_jsonl}")
    ```
스크립트 실행:
    ```bash
    python examples/multimodal/speech_llm/data/create_test_manifest.py
    ```
## 2. 모델 준비
### 2.1 Fast Conformer 모델 다운로드
사전 학습된 `stt_en_fastconformer_transducer_large.nemo` 모델을 다운로드하여 models 폴더에 저장합니다:

    ```bash
    wget https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_fastconformer_transducer_large/versions/1.0.0/files/stt_en_fastconformer_transducer_large.nemo -O ./examples/multimodal/speech_llm/models/stt_en_fastconformer_transducer_large.nemo
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
    python examples/multimodal/speech_llm/modular_audio_gpt_eval.py --config-path=conf --config-name=salm_config_cpu.yaml
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