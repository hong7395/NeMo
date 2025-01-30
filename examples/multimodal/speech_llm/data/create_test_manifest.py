import os
import json
from tqdm import tqdm

def create_librispeech_manifest(data_dir, output_file):
    """
    LibriSpeech test-clean 데이터셋을 기반으로 매니페스트 파일을 생성합니다.

    Parameters:
        data_dir (str): LibriSpeech test-clean 디렉토리 경로.
        output_file (str): 생성할 매니페스트 파일 경로.
    """
    manifest = []

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
                        audio_sample = {
                            "audio_filepath": audio_file,
                            "offset": 0.0,  # 전체 오디오 사용
                            "duration": None,  # 전체 오디오 사용
                            "context": "What is the transcription of this audio?",
                            "answer": transcript
                        }
                        manifest.append(audio_sample)

    # 매니페스트 파일 저장
    with open(output_file, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry) + "\n")

    print(f"매니페스트 파일이 {output_file}에 저장되었습니다.")

# LibriSpeech 데이터 경로 및 매니페스트 출력 파일 경로
data_dir = "/NeMo/examples/multimodal/speech_llm/data/LibriSpeech/test-clean"  # LibriSpeech test-clean 디렉토리
output_file = "test_manifest.jsonl"  # 생성할 매니페스트 파일

# 매니페스트 생성 실행
create_librispeech_manifest(data_dir, output_file)
