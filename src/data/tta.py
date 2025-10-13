# ==================== TTA (Test Time Augmentation) 모듈 ==================== #
"""
TTA (Test Time Augmentation) 구현

PRD 12: 추론 시점 데이터 증강
- 4가지 TTA 전략: paraphrase, reorder, synonym, mask
- 예측 결과 앙상블로 robustness 향상
"""

# ---------------------- 라이브러리 임포트 ---------------------- #
from typing import List, Dict, Any, Optional
import random
import re
import copy

# ---------------------- 외부 라이브러리 ---------------------- #
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel


# ==================== TTAugmentor 클래스 ==================== #
class TTAugmentor:
    """Test Time Augmentation 수행 클래스"""

    def __init__(
        self,
        strategies: List[str] = None,
        n_augmentations: int = 3,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model: Optional[PreTrainedModel] = None,
        seed: int = 42
    ):
        """
        Args:
            strategies: 사용할 TTA 전략 리스트 (None이면 모든 전략 사용)
            n_augmentations: 각 전략당 생성할 증강 개수
            tokenizer: 토크나이저 (mask 전략에 필요)
            model: 모델 (추론에 필요)
            seed: 랜덤 시드
        """
        # 사용 가능한 전략
        self.available_strategies = ['paraphrase', 'reorder', 'synonym', 'mask']

        # 전략 설정
        if strategies is None:
            self.strategies = self.available_strategies
        else:
            # 유효한 전략만 필터링
            self.strategies = [s for s in strategies if s in self.available_strategies]
            if not self.strategies:
                raise ValueError(f"유효한 TTA 전략이 없습니다. 사용 가능: {self.available_strategies}")

        self.n_augmentations = n_augmentations
        self.tokenizer = tokenizer
        self.model = model
        self.seed = seed

        # 랜덤 시드 설정
        random.seed(seed)

        # 동의어 딕셔너리 (간단한 예시)
        self.synonym_dict = {
            '좋은': ['훌륭한', '멋진', '우수한', '괜찮은'],
            '나쁜': ['안좋은', '형편없는', '별로인', '못난'],
            '크다': ['거대하다', '대형이다', '넓다'],
            '작다': ['소형이다', '좁다', '미세하다'],
            '빠르다': ['신속하다', '재빠르다', '급하다'],
            '느리다': ['더디다', '천천히다', '완만하다'],
        }

    def augment(self, text: str, strategy: str = None) -> List[str]:
        """
        단일 텍스트에 대해 TTA 수행

        Args:
            text: 원본 텍스트
            strategy: 사용할 전략 (None이면 모든 전략 사용)

        Returns:
            증강된 텍스트 리스트 (원본 포함)
        """
        augmented_texts = [text]  # 원본 포함

        # 전략 결정
        strategies_to_use = [strategy] if strategy else self.strategies

        for strat in strategies_to_use:
            if strat == 'paraphrase':
                augmented_texts.extend(self._paraphrase_augment(text))
            elif strat == 'reorder':
                augmented_texts.extend(self._reorder_augment(text))
            elif strat == 'synonym':
                augmented_texts.extend(self._synonym_augment(text))
            elif strat == 'mask':
                augmented_texts.extend(self._mask_augment(text))

        return augmented_texts

    def batch_augment(
        self,
        texts: List[str],
        strategy: str = None
    ) -> List[List[str]]:
        """
        배치 텍스트에 대해 TTA 수행

        Args:
            texts: 원본 텍스트 리스트
            strategy: 사용할 전략 (None이면 모든 전략 사용)

        Returns:
            각 텍스트의 증강 결과 리스트
        """
        return [self.augment(text, strategy) for text in texts]

    def _paraphrase_augment(self, text: str) -> List[str]:
        """
        Paraphrase 전략: 문장 구조를 약간 변경
        (실제로는 paraphrase 모델을 사용해야 하지만, 여기서는 간단한 변환 사용)
        """
        augmented = []

        for _ in range(self.n_augmentations):
            # 간단한 paraphrasing: 문장 순서 변경 + 약간의 수정
            sentences = self._split_sentences(text)

            if len(sentences) > 1:
                # 문장 순서 일부 변경
                modified = sentences.copy()
                if random.random() > 0.5 and len(modified) >= 2:
                    # 랜덤하게 두 문장 위치 교환
                    idx1, idx2 = random.sample(range(len(modified)), 2)
                    modified[idx1], modified[idx2] = modified[idx2], modified[idx1]

                augmented.append(' '.join(modified))
            else:
                # 단일 문장인 경우 동의어 치환
                augmented.append(self._replace_words(text))

        return augmented

    def _reorder_augment(self, text: str) -> List[str]:
        """
        Reorder 전략: 문장/단어 순서 재배열
        """
        augmented = []

        for _ in range(self.n_augmentations):
            sentences = self._split_sentences(text)

            if len(sentences) > 1:
                # 문장 순서 셔플
                shuffled = sentences.copy()
                random.shuffle(shuffled)
                augmented.append(' '.join(shuffled))
            else:
                # 단일 문장인 경우 단어 순서 일부 변경
                words = text.split()
                if len(words) > 3:
                    # 인접한 단어쌍 교환
                    modified = words.copy()
                    idx = random.randint(0, len(modified) - 2)
                    modified[idx], modified[idx + 1] = modified[idx + 1], modified[idx]
                    augmented.append(' '.join(modified))
                else:
                    augmented.append(text)

        return augmented

    def _synonym_augment(self, text: str) -> List[str]:
        """
        Synonym 전략: 동의어 치환
        """
        augmented = []

        for _ in range(self.n_augmentations):
            modified = self._replace_words(text)
            augmented.append(modified)

        return augmented

    def _mask_augment(self, text: str) -> List[str]:
        """
        Mask 전략: 일부 토큰을 마스킹
        """
        if self.tokenizer is None:
            # 토크나이저 없으면 원본 반환
            return [text] * self.n_augmentations

        augmented = []

        for _ in range(self.n_augmentations):
            words = text.split()

            if len(words) <= 2:
                augmented.append(text)
                continue

            # 랜덤하게 10-20%의 단어를 마스킹
            n_mask = max(1, int(len(words) * random.uniform(0.1, 0.2)))
            mask_indices = random.sample(range(len(words)), n_mask)

            masked = words.copy()
            for idx in mask_indices:
                masked[idx] = '[MASK]'

            augmented.append(' '.join(masked))

        return augmented

    def _split_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분리"""
        # 간단한 문장 분리 (마침표, 느낌표, 물음표 기준)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _replace_words(self, text: str) -> str:
        """동의어 사전을 사용해 단어 치환"""
        words = text.split()
        modified = words.copy()

        # 랜덤하게 1-2개 단어 치환
        n_replace = min(len(words), random.randint(1, 2))

        for _ in range(n_replace):
            # 치환 가능한 단어 찾기
            replaceable = [i for i, w in enumerate(words) if w in self.synonym_dict]

            if replaceable:
                idx = random.choice(replaceable)
                word = words[idx]
                synonyms = self.synonym_dict[word]
                modified[idx] = random.choice(synonyms)

        return ' '.join(modified)

    def predict_with_tta(
        self,
        text: str,
        predict_fn: callable,
        ensemble_method: str = 'voting'
    ) -> Dict[str, Any]:
        """
        TTA를 적용하여 예측 수행

        Args:
            text: 입력 텍스트
            predict_fn: 예측 함수 (text -> prediction)
            ensemble_method: 앙상블 방법 ('voting' 또는 'averaging')

        Returns:
            최종 예측 결과 딕셔너리
        """
        # TTA 수행
        augmented_texts = self.augment(text)

        # 각 증강 텍스트에 대해 예측
        predictions = []
        for aug_text in augmented_texts:
            pred = predict_fn(aug_text)
            predictions.append(pred)

        # 앙상블
        if ensemble_method == 'voting':
            # 다수결 투표
            final_pred = max(set(predictions), key=predictions.count)
        elif ensemble_method == 'averaging':
            # 평균 (확률값이 있는 경우)
            if isinstance(predictions[0], (list, torch.Tensor)):
                # 확률 분포의 평균
                avg_probs = torch.mean(torch.stack([torch.tensor(p) for p in predictions]), dim=0)
                final_pred = torch.argmax(avg_probs).item()
            else:
                # 단순 투표
                final_pred = max(set(predictions), key=predictions.count)
        else:
            raise ValueError(f"지원하지 않는 앙상블 방법: {ensemble_method}")

        return {
            'prediction': final_pred,
            'all_predictions': predictions,
            'n_augmentations': len(augmented_texts)
        }


# ==================== 팩토리 함수 ==================== #
def create_tta_augmentor(
    strategies: List[str] = None,
    n_augmentations: int = 3,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    model: Optional[PreTrainedModel] = None,
    seed: int = 42
) -> TTAugmentor:
    """
    TTAugmentor 생성 팩토리 함수

    Args:
        strategies: TTA 전략 리스트
        n_augmentations: 증강 개수
        tokenizer: 토크나이저
        model: 모델
        seed: 랜덤 시드

    Returns:
        TTAugmentor 인스턴스
    """
    return TTAugmentor(
        strategies=strategies,
        n_augmentations=n_augmentations,
        tokenizer=tokenizer,
        model=model,
        seed=seed
    )


# ==================== 사용 예시 ==================== #
if __name__ == "__main__":
    # 기본 TTA 생성
    tta = create_tta_augmentor(
        strategies=['paraphrase', 'reorder', 'synonym'],
        n_augmentations=2
    )

    # 단일 텍스트 증강
    text = "이 제품은 정말 좋은 품질을 가지고 있습니다."
    augmented = tta.augment(text)

    print(f"원본: {text}")
    print(f"\n증강된 텍스트 ({len(augmented)}개):")
    for i, aug in enumerate(augmented, 1):
        print(f"{i}. {aug}")

    # 배치 증강
    texts = [
        "빠른 배송 감사합니다.",
        "제품이 작아서 실망했어요."
    ]
    batch_augmented = tta.batch_augment(texts)

    print(f"\n\n배치 증강 결과:")
    for i, (orig, augs) in enumerate(zip(texts, batch_augmented)):
        print(f"\n[{i+1}] 원본: {orig}")
        print(f"    증강: {len(augs)}개")
