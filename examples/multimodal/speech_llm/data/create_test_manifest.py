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
