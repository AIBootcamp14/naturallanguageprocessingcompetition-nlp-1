# ==================== 대화 데이터 전처리 모듈 ==================== #
"""
대화 요약 데이터 전처리 시스템

주요 기능:
- 노이즈 제거 (\\n, <br> 등)
- 텍스트 정규화
- 화자 정보 추출
- 특수 토큰 처리
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import re
from typing import List, Optional, Tuple

# ---------------------- 서드파티 라이브러리 ---------------------- #
import pandas as pd


# ==================== DialoguePreprocessor 클래스 정의 ==================== #
class DialoguePreprocessor:
    """대화 데이터 전처리 클래스"""

    # ---------------------- 초기화 함수 ---------------------- #
    def __init__(self):
        """전처리기 초기화"""
        # 화자 패턴 정규식 (예: #Person1#, #Person2# 등)
        self.speaker_pattern = re.compile(r'#Person\d+#')  # 화자 패턴 컴파일


    # ---------------------- 노이즈 제거 함수 ---------------------- #
    def clean_dialogue(self, text: str) -> str:
        """
        대화 텍스트의 노이즈 제거

        제거 항목:
        - \\n → \n 변환 (이스케이프된 개행 문자)
        - <br> 태그 제거
        - 중복 공백 제거
        - 앞뒤 공백 제거

        Args:
            text: 원본 대화 텍스트

        Returns:
            str: 정제된 대화 텍스트
        """
        if not text or not isinstance(text, str):          # 텍스트가 없거나 문자열이 아닌 경우
            return ""                                       # 빈 문자열 반환

        # -------------- 1. 이스케이프된 개행 문자 변환 -------------- #
        text = text.replace('\\n', '\n')                    # \\n을 실제 개행으로 변환

        # -------------- 2. HTML 태그 제거 -------------- #
        text = re.sub(r'<br\s*/?>', '\n', text)             # <br> 태그를 개행으로 변환
        text = re.sub(r'<[^>]+>', '', text)                 # 기타 HTML 태그 제거

        # -------------- 3. 중복 공백 정리 -------------- #
        # 여러 개의 공백을 하나로 통일
        text = re.sub(r'[ \t]+', ' ', text)                 # 연속된 공백/탭을 하나의 공백으로

        # -------------- 4. 중복 개행 정리 -------------- #
        # 3개 이상의 연속된 개행을 2개로 제한
        text = re.sub(r'\n{3,}', '\n\n', text)              # 과도한 개행 제거

        # -------------- 5. 앞뒤 공백 제거 -------------- #
        text = text.strip()                                 # 앞뒤 공백 제거

        return text                                         # 정제된 텍스트 반환


    # ---------------------- 텍스트 정규화 함수 ---------------------- #
    def normalize_dialogue(self, text: str) -> str:
        """
        대화 텍스트 정규화

        정규화 항목:
        - 특수 문자 처리
        - 자모음 단독 사용 처리 (ㅋㅋ, ㅇㅇ 등)
        - 이모티콘 처리 (선택적)

        Args:
            text: 정제된 대화 텍스트

        Returns:
            str: 정규화된 대화 텍스트
        """
        if not text:                                        # 텍스트가 없는 경우
            return ""                                       # 빈 문자열 반환

        # -------------- 자모음 단독 사용 정규화 -------------- #
        # ㅋㅋㅋ → [웃음], ㅇㅇ → 응응 등의 변환은 선택적으로 적용 가능
        # 현재는 원본 유지 (모델이 학습하도록)

        return text                                         # 정규화된 텍스트 반환


    # ---------------------- 화자 추출 함수 ---------------------- #
    def extract_speakers(self, dialogue: str) -> List[str]:
        """
        대화에서 화자 정보 추출

        Args:
            dialogue: 대화 텍스트

        Returns:
            List[str]: 화자 리스트 (예: ['#Person1#', '#Person2#'])
        """
        if not dialogue:                                    # 대화가 없는 경우
            return []                                       # 빈 리스트 반환

        # 화자 패턴 매칭
        speakers = self.speaker_pattern.findall(dialogue)   # 화자 패턴 추출

        # 중복 제거 및 정렬
        unique_speakers = sorted(set(speakers))             # 중복 제거 후 정렬

        return unique_speakers                              # 화자 리스트 반환


    # ---------------------- 대화 턴 개수 계산 함수 ---------------------- #
    def count_turns(self, dialogue: str) -> int:
        """
        대화의 턴 개수 계산

        Args:
            dialogue: 대화 텍스트

        Returns:
            int: 대화 턴 개수
        """
        if not dialogue:                                    # 대화가 없는 경우
            return 0                                        # 0 반환

        # 화자 패턴 개수 = 턴 개수
        turns = len(self.speaker_pattern.findall(dialogue)) # 화자 패턴 개수 계산

        return turns                                        # 턴 개수 반환


    # ---------------------- 대화 분할 함수 ---------------------- #
    def split_dialogue_by_speaker(self, dialogue: str) -> List[Tuple[str, str]]:
        """
        대화를 화자별로 분할

        Args:
            dialogue: 대화 텍스트

        Returns:
            List[Tuple[str, str]]: [(화자, 발화내용), ...] 리스트
        """
        if not dialogue:                                    # 대화가 없는 경우
            return []                                       # 빈 리스트 반환

        # -------------- 화자와 발화 분리 -------------- #
        # #Person1#: 안녕하세요\n#Person2#: 반갑습니다 형태 파싱
        turns = []                                          # 턴 리스트 초기화

        # 개행 문자로 분할
        lines = dialogue.split('\n')                        # 개행으로 분할

        # 각 라인 처리
        for line in lines:
            line = line.strip()                             # 공백 제거

            # 화자 패턴이 있는 경우
            if self.speaker_pattern.search(line):
                # 화자와 발화 분리
                match = re.match(r'(#Person\d+#):\s*(.*)', line)  # 화자:발화 패턴 매칭

                # 매칭 성공 시
                if match:
                    speaker = match.group(1)                # 화자 추출
                    utterance = match.group(2).strip()      # 발화 내용 추출
                    turns.append((speaker, utterance))      # 턴 추가

        return turns                                        # 턴 리스트 반환


    # ---------------------- 배치 전처리 함수 ---------------------- #
    def preprocess_batch(
        self,
        dialogues: List[str],
        summaries: Optional[List[str]] = None
    ) -> Tuple[List[str], Optional[List[str]]]:
        """
        대화 및 요약 배치 전처리

        Args:
            dialogues: 대화 리스트
            summaries: 요약 리스트 (선택적)

        Returns:
            Tuple[List[str], Optional[List[str]]]: (정제된 대화 리스트, 정제된 요약 리스트)
        """
        # -------------- 대화 전처리 -------------- #
        cleaned_dialogues = []                              # 정제된 대화 리스트

        # 각 대화 처리
        for dialogue in dialogues:
            cleaned = self.clean_dialogue(dialogue)         # 노이즈 제거
            normalized = self.normalize_dialogue(cleaned)   # 정규화
            cleaned_dialogues.append(normalized)            # 리스트에 추가

        # -------------- 요약 전처리 (선택적) -------------- #
        cleaned_summaries = None                            # 기본값

        # 요약이 제공된 경우
        if summaries is not None:
            cleaned_summaries = []                          # 정제된 요약 리스트

            # 각 요약 처리
            for summary in summaries:
                cleaned = self.clean_dialogue(summary)      # 노이즈 제거
                normalized = self.normalize_dialogue(cleaned)  # 정규화
                cleaned_summaries.append(normalized)        # 리스트에 추가

        return cleaned_dialogues, cleaned_summaries         # 결과 반환


    # ---------------------- DataFrame 전처리 함수 ---------------------- #
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame 전체 전처리

        Args:
            df: 원본 DataFrame (columns: fname, dialogue, summary)

        Returns:
            pd.DataFrame: 전처리된 DataFrame
        """
        # DataFrame 복사
        df = df.copy()                                      # 원본 보존을 위한 복사

        # -------------- 대화 전처리 -------------- #
        if 'dialogue' in df.columns:                        # dialogue 컬럼이 있는 경우
            df['dialogue'] = df['dialogue'].apply(          # 각 대화에 전처리 적용
                lambda x: self.normalize_dialogue(self.clean_dialogue(x))
            )

        # -------------- 요약 전처리 -------------- #
        if 'summary' in df.columns:                         # summary 컬럼이 있는 경우
            df['summary'] = df['summary'].apply(            # 각 요약에 전처리 적용
                lambda x: self.normalize_dialogue(self.clean_dialogue(x))
            )

        # -------------- 통계 정보 추가 (선택적) -------------- #
        if 'dialogue' in df.columns:
            # 화자 수
            df['num_speakers'] = df['dialogue'].apply(      # 화자 수 계산
                lambda x: len(self.extract_speakers(x))
            )

            # 턴 수
            df['num_turns'] = df['dialogue'].apply(         # 턴 수 계산
                self.count_turns
            )

        return df                                           # 전처리된 DataFrame 반환


# ==================== 편의 함수 ==================== #
# ---------------------- 단일 대화 전처리 함수 ---------------------- #
def preprocess_dialogue(dialogue: str) -> str:
    """
    단일 대화 전처리 편의 함수

    Args:
        dialogue: 원본 대화 텍스트

    Returns:
        str: 전처리된 대화 텍스트
    """
    preprocessor = DialoguePreprocessor()                   # 전처리기 생성
    cleaned = preprocessor.clean_dialogue(dialogue)         # 노이즈 제거
    normalized = preprocessor.normalize_dialogue(cleaned)   # 정규화

    return normalized                                       # 전처리된 텍스트 반환
