import os
import json
import glob
from pathlib import Path
from tqdm import tqdm

# LibriSpeech 데이터셋 경로
librispeech_dir = "/NeMo/examples/multimodal/speech_llm/data/LibriSpeech/test-clean"
output_file = "/NeMo/examples/multimodal/speech_llm/data/test_manifest.jsonl"

# 모든 .flac 파일 찾기
audio_files = glob.glob(f"{librispeech_dir}/**/*.flac", recursive=True)

print(f"총 {len(audio_files)}개의 오디오 파일을 찾았습니다.")

with open(output_file, 'w') as f:
    for audio_file in tqdm(audio_files, desc="매니페스트 생성 중"):
        # 상대 경로를 절대 경로로 변환
        audio_path = os.path.abspath(audio_file)
        
        # 트랜스크립션 파일 찾기 (.trans.txt)
        speaker_id = Path(audio_file).parent.name
        chapter_id = Path(audio_file).parent.parent.name
        trans_file = os.path.join(librispeech_dir, chapter_id, speaker_id, f"{chapter_id}-{speaker_id}.trans.txt")
        
        if os.path.exists(trans_file):
            # 파일 ID 추출 (예: 1089-134686-0000)
            file_id = os.path.basename(audio_file).split('.')[0]
            
            # 트랜스크립션 파일에서 해당 ID의 텍스트 찾기
            with open(trans_file, 'r') as trans:
                for line in trans:
                    if line.startswith(file_id):
                        # 첫 번째 단어(ID)를 제외한 나머지가 트랜스크립션
                        transcription = ' '.join(line.strip().split()[1:])
                        
                        # JSONL 형식으로 저장
                        entry = {
                            "audio_filepath": audio_path,
                            "duration": None,  # 실제 지속 시간은 오디오 파일에서 계산 가능
                            "context": "what is the transcription of the audio?",
                            "answer": transcription
                        }
                        f.write(json.dumps(entry) + '\n')
                        break

print(f"매니페스트 파일이 생성되었습니다: {output_file}")
print(f"생성된 파일을 확인하려면: head -n 1 {output_file}")
