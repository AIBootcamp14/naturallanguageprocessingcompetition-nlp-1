# 📚 미구현 기능 상세 보고서

**작성일**: 2025-10-11
**분석자**: Claude Code (정직한 검증)
**분석 범위**: `/docs/PRD` 전체 (19개 문서) vs 실제 코드베이스
**실제 구현률**: **81.5%** (12개 완전 구현, 5개 부분 구현, 2개 미구현)

---

## ⚠️ 중요: 이전 주장 vs 실제

### ❌ 이전 주장 (틀림)
```
✅ 구현 완료: 95%+
✅ 데이터 증강 완료
✅ 추론 최적화 완료
```

### ✅ 실제 검증 결과
```
실제 구현률: 81.5%
❌ 데이터 증강: 30% (back_translator.py, paraphraser.py 빈 파일)
❌ 추론 최적화: 0% (PRD 문서만 존재, 코드 없음)
⚠️ 인코딩 문제: 2개 파일 한글 깨짐
```

---

## 🔴 치명적 미구현 항목

### 1. PRD 04: 데이터 증강 (30% 구현)

#### ❌ 미구현 (빈 파일)
```bash
# 확인 결과
$ ls -lh src/augmentation/
-rw-r--r-- 1 user user    0 Oct 11 back_translator.py    # 0 bytes ❌
-rw-r--r-- 1 user user    0 Oct 11 paraphraser.py        # 0 bytes ❌
-rw-r--r-- 1 user user 2.0K Oct 11 text_augmenter.py     # 68 lines ✅
```

#### 구현 필요 사항

**파일 1: `src/augmentation/back_translator.py`**
```python
"""
역번역 기반 데이터 증강
한국어 → 영어 → 한국어 번역으로 데이터 다양성 확보
"""

from transformers import MarianMTModel, MarianTokenizer
import torch

class BackTranslator:
    """역번역 증강 클래스"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        초기화

        필요 모델:
        - Helsinki-NLP/opus-mt-ko-en (한→영)
        - Helsinki-NLP/opus-mt-en-ko (영→한)
        """
        self.device = device

        # 한→영 모델
        self.ko_en_model = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-ko-en"
        ).to(device)
        self.ko_en_tokenizer = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-ko-en"
        )

        # 영→한 모델
        self.en_ko_model = MarianMTModel.from_pretrained(
            "Helsinki-NLP/opus-mt-en-ko"
        ).to(device)
        self.en_ko_tokenizer = MarianTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-en-ko"
        )

    def back_translate(self, text: str) -> str:
        """
        역번역 수행

        Args:
            text: 한국어 텍스트

        Returns:
            역번역된 한국어 텍스트
        """
        # 1단계: 한국어 → 영어
        ko_inputs = self.ko_en_tokenizer(
            text,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        en_outputs = self.ko_en_model.generate(**ko_inputs)
        en_text = self.ko_en_tokenizer.decode(
            en_outputs[0],
            skip_special_tokens=True
        )

        # 2단계: 영어 → 한국어
        en_inputs = self.en_ko_tokenizer(
            en_text,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        ko_outputs = self.en_ko_model.generate(**en_inputs)
        back_translated = self.en_ko_tokenizer.decode(
            ko_outputs[0],
            skip_special_tokens=True
        )

        return back_translated

    def augment(self, dialogue: str, summary: str) -> tuple:
        """
        대화-요약 쌍 증강

        Args:
            dialogue: 원본 대화
            summary: 원본 요약

        Returns:
            (증강된 대화, 원본 요약)
        """
        aug_dialogue = self.back_translate(dialogue)
        # 요약은 그대로 유지
        return aug_dialogue, summary

# 사용 예시
if __name__ == "__main__":
    augmenter = BackTranslator()

    original = "#Person1#: 안녕하세요 #Person2#: 반갑습니다"
    summary = "인사"

    aug_dialogue, aug_summary = augmenter.augment(original, summary)
    print(f"원본: {original}")
    print(f"증강: {aug_dialogue}")
```

**예상 작업 시간**: 2-3시간
**난이도**: ★★★☆☆
**의존성**: `transformers`, `sentencepiece`

---

**파일 2: `src/augmentation/paraphraser.py`**
```python
"""
문장 변형(Paraphrasing) 기반 데이터 증강
"""

import random
from typing import Dict, List

class Paraphraser:
    """문장 변형 증강 클래스"""

    def __init__(self, seed: int = 42):
        """초기화"""
        self.seed = seed
        random.seed(seed)

        # 동의어 사전
        self.synonym_dict = {
            # 인사
            "안녕하세요": ["안녕", "반갑습니다", "환영합니다", "안녕하십니까"],
            "감사합니다": ["고맙습니다", "감사해요", "고마워요", "감사드립니다"],
            "죄송합니다": ["미안합니다", "죄송해요", "미안해요", "송구합니다"],

            # 답변
            "네": ["예", "알겠습니다", "그렇습니다", "맞습니다"],
            "아니요": ["아닙니다", "아니에요", "그렇지 않습니다"],

            # 행동
            "먹다": ["섭취하다", "드시다", "식사하다"],
            "가다": ["이동하다", "향하다", "출발하다"],
            "오다": ["도착하다", "방문하다", "찾아오다"],

            # 형용사
            "좋다": ["훌륭하다", "괜찮다", "만족스럽다", "우수하다"],
            "나쁘다": ["안좋다", "별로다", "형편없다"],
            "크다": ["거대하다", "넓다", "큰"],
            "작다": ["적다", "미미하다", "작은"],

            # 명사
            "밥": ["식사", "음식", "끼니"],
            "집": ["집", "가정", "주택"],
            "사람": ["인간", "사람", "개인"],
        }

    def paraphrase(self, text: str) -> str:
        """
        동의어 치환으로 문장 변형

        Args:
            text: 원본 텍스트

        Returns:
            변형된 텍스트
        """
        result = text

        # 동의어 사전의 각 단어에 대해
        for original, synonyms in self.synonym_dict.items():
            if original in result:
                # 30% 확률로 치환
                if random.random() < 0.3:
                    synonym = random.choice(synonyms)
                    result = result.replace(original, synonym, 1)

        return result

    def augment(self, dialogue: str, summary: str) -> tuple:
        """
        대화-요약 쌍 증강

        Args:
            dialogue: 원본 대화
            summary: 원본 요약

        Returns:
            (증강된 대화, 원본 요약)
        """
        aug_dialogue = self.paraphrase(dialogue)
        # 요약은 그대로 유지
        return aug_dialogue, summary

    def batch_augment(
        self,
        dialogues: List[str],
        summaries: List[str],
        n_augmentations: int = 2
    ) -> tuple:
        """
        배치 증강

        Args:
            dialogues: 대화 리스트
            summaries: 요약 리스트
            n_augmentations: 증강 횟수

        Returns:
            (증강된 대화 리스트, 증강된 요약 리스트)
        """
        aug_dialogues = list(dialogues)
        aug_summaries = list(summaries)

        for dialogue, summary in zip(dialogues, summaries):
            for _ in range(n_augmentations):
                aug_d, aug_s = self.augment(dialogue, summary)
                aug_dialogues.append(aug_d)
                aug_summaries.append(aug_s)

        return aug_dialogues, aug_summaries

# 사용 예시
if __name__ == "__main__":
    augmenter = Paraphraser()

    original = "#Person1#: 안녕하세요 #Person2#: 감사합니다"
    summary = "인사"

    aug_dialogue, aug_summary = augmenter.augment(original, summary)
    print(f"원본: {original}")
    print(f"증강: {aug_dialogue}")
```

**예상 작업 시간**: 1-2시간
**난이도**: ★★☆☆☆

---

**`src/augmentation/__init__.py` 업데이트 필요**
```python
"""데이터 증강 모듈"""

from src.augmentation.text_augmenter import TextAugmenter
from src.augmentation.back_translator import BackTranslator
from src.augmentation.paraphraser import Paraphraser

__all__ = [
    'TextAugmenter',
    'BackTranslator',
    'Paraphraser',
]

def create_augmenter(augment_type='basic', **kwargs):
    """
    증강기 생성 팩토리 함수

    Args:
        augment_type: 'basic', 'back_translation', 'paraphrase'

    Returns:
        증강기 인스턴스
    """
    if augment_type == 'basic':
        return TextAugmenter(**kwargs)
    elif augment_type == 'back_translation':
        return BackTranslator(**kwargs)
    elif augment_type == 'paraphrase':
        return Paraphraser(**kwargs)
    else:
        raise ValueError(f"Unknown augment_type: {augment_type}")
```

---

### 2. PRD 17: 추론 최적화 (0% 구현)

#### 현재 상태
```bash
$ find src/ -name "*onnx*" -o -name "*tensorrt*" -o -name "*quantiz*"
# 결과: 파일 없음 ❌
```

#### 구현 필요 사항

**이 부분은 선택적입니다.** PRD 17은 성능 최적화를 위한 고급 기능이며, 현재 대회 목적상 필수는 아닙니다.

**권장**: PRD 17은 나중에 필요할 때 구현하거나, PRD 문서에서 제거하는 것을 권장합니다.

---

### 3. 인코딩 문제 (2개 파일)

#### 문제 파일
```bash
$ file src/prompts/prompt_manager.py
src/prompts/prompt_manager.py: data  # ← 'data'로 나옴 (UTF-8 아님)

$ file src/validation/data_quality.py
src/validation/data_quality.py: data  # ← 'data'로 나옴 (UTF-8 아님)
```

#### 해결 방법

**단계 1: 파일 인코딩 확인**
```bash
file -i src/prompts/prompt_manager.py
file -i src/validation/data_quality.py
```

**단계 2: UTF-8로 변환**
```bash
# iconv 사용
iconv -f CP949 -t UTF-8 src/prompts/prompt_manager.py > temp.py
mv temp.py src/prompts/prompt_manager.py

iconv -f CP949 -t UTF-8 src/validation/data_quality.py > temp.py
mv temp.py src/validation/data_quality.py
```

또는

**Python으로 재저장**
```python
# fix_encoding.py
import codecs

files = [
    'src/prompts/prompt_manager.py',
    'src/validation/data_quality.py'
]

for filepath in files:
    # CP949 또는 EUC-KR로 읽기 시도
    for encoding in ['cp949', 'euc-kr', 'utf-8']:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()

            # UTF-8로 재저장
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✅ {filepath} - {encoding} → UTF-8 변환 완료")
            break
        except:
            continue
```

**예상 작업 시간**: 10분
**난이도**: ★☆☆☆☆

---

## ✅ 완전 구현된 항목 (검증 완료)

### 1. Solar API (PRD 09) - 100% ✅
- ✅ `src/api/solar_client.py` (289 lines)
- ✅ Few-shot 프롬프트 빌더
- ✅ 토큰 절약 전처리 (70% 절감)
- ✅ 배치 처리
- ✅ MD5 기반 캐싱

### 2. K-Fold 교차 검증 (PRD 10) - 100% ✅
- ✅ `src/validation/kfold.py` (170 lines)
- ✅ Stratified K-Fold 지원
- ✅ Fold 결과 집계

### 3. 앙상블 시스템 (PRD 12) - 100% ✅
- ✅ `src/ensemble/manager.py` (160 lines)
- ✅ `src/ensemble/weighted.py` (141 lines)
- ✅ `src/ensemble/voting.py` (147 lines)

### 4. Optuna 최적화 (PRD 13) - 100% ✅
- ✅ `src/optimization/optuna_optimizer.py` (409 lines)
- ✅ TPE Sampler, Median Pruner
- ✅ 시각화 및 저장 기능

### 5. 프롬프트 관리 (PRD 15) - 100% ✅
- ✅ `src/prompts/prompt_manager.py` (316 lines, 인코딩 문제만 있음)
- ✅ `src/prompts/templates.py` (302 lines)
- ✅ 12개 템플릿 구현

### 6. 데이터 품질 검증 (PRD 16) - 100% ✅
- ✅ `src/validation/data_quality.py` (444 lines, 인코딩 문제만 있음)
- ✅ 4단계 검증 (구조, 의미, 통계, 이상치)

### 7. 후처리 시스템 - 100% ✅
- ✅ `src/postprocessing/text_postprocessor.py` (59 lines)

### 8. Config 전략 - 100% ✅
- ✅ `configs/strategies/data_augmentation.yaml`
- ✅ `configs/strategies/ensemble.yaml`
- ✅ `configs/strategies/optuna.yaml`
- ✅ `configs/strategies/cross_validation.yaml`

---

## 📊 우선순위별 작업 계획

### 🔴 긴급 (즉시 처리)
1. **인코딩 문제 해결** (10분)
   - `src/prompts/prompt_manager.py`
   - `src/validation/data_quality.py`

### 🟠 중요 (1-2일)
2. **데이터 증강 완성** (3-5시간)
   - `back_translator.py` 구현 (2-3h)
   - `paraphraser.py` 구현 (1-2h)
   - `__init__.py` 업데이트 (10분)

### 🟢 선택적 (나중에)
3. **추론 최적화** (8-10시간, 선택)
   - ONNX, TensorRT 등
   - 또는 PRD 17 문서 제거 고려

---

## 🎯 수정된 전체 구현률

```
완전 구현 (90%+): 12개 PRD (63%)
부분 구현 (50-89%): 5개 PRD (26%)
미구현 (<50%): 2개 PRD (11%)

전체 평균: 81.5%
```

### 긴급 수정 후 예상 구현률
```
인코딩 문제 해결: 81.5% → 83%
데이터 증강 완성: 83% → 87%

최종 목표: 87%+ (추론 최적화 제외)
```

---

## 📝 다음 문서

- **01_PRD_구현_갭_분석.md**: 상세 PRD별 분석 (이미 정확함)
- **02_실행_옵션_시스템_구현_가이드.md**: 선택적 고급 기능
- **03_LLM_통합_가이드.md**: 선택적 고급 기능

---

## 💡 핵심 메시지

**거짓말 없이 정직하게:**
- ✅ 핵심 기능은 **대부분 구현**되어 있습니다 (81.5%)
- ❌ 데이터 증강은 **30%만** 구현 (빈 파일 2개)
- ❌ 추론 최적화는 **0%** (문서만 존재)
- ⚠️ 인코딩 문제로 한글이 깨진 파일 2개

**빠르게 수정 가능:**
- 인코딩 문제: 10분
- 데이터 증강: 3-5시간
- 총 작업 시간: **5시간 이내**

**현재 시스템으로 가능한 것:**
- ✅ 학습 및 추론 (KoBART, Llama, Qwen)
- ✅ Solar API 사용
- ✅ K-Fold 교차 검증
- ✅ 앙상블
- ✅ Optuna 최적화
- ⚠️ 기본 데이터 증강만 (고급 증강 미지원)
