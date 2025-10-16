# ==================== 베이스라인 자동 검증 모듈 ==================== #
"""
베이스라인 설정 자동 검증

PRD 18: 베이스라인 검증 시스템
- 토크나이저 설정 검증
- 학습률 적정성 검증
- 생성 품질 검증
"""

# ---------------------- 라이브러리 임포트 ---------------------- #
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json

# ---------------------- 외부 라이브러리 ---------------------- #
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel


# ==================== BaselineChecker 클래스 ==================== #
class BaselineChecker:
    """베이스라인 설정 자동 검증 클래스"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger=None
    ):
        """
        Args:
            config: 검증 설정 (기본값 사용 가능)
            logger: Logger 인스턴스
        """
        self.config = config or self._get_default_config()
        self.logger = logger

        # 검증 결과 저장
        self.validation_results = {}

        self._log("BaselineChecker 초기화 완료")

    def _log(self, msg: str):
        """로그 출력"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def _get_default_config(self) -> Dict[str, Any]:
        """기본 검증 설정"""
        return {
            'tokenizer': {
                'max_length_min': 128,
                'max_length_max': 2048,
                'vocab_size_min': 1000,
                'special_tokens': ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
            },
            'learning_rate': {
                'min': 1e-6,
                'max': 1e-3,
                'recommended_min': 1e-5,
                'recommended_max': 5e-4
            },
            'generation': {
                'min_length': 10,
                'max_length': 512,
                'min_rouge_l': 0.1,  # 최소 ROUGE-L 점수
                'max_repetition_ratio': 0.3  # 최대 반복 비율
            }
        }

    def check_tokenization(
        self,
        tokenizer: PreTrainedTokenizer,
        sample_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        토크나이저 설정 검증

        Args:
            tokenizer: 토크나이저
            sample_texts: 샘플 텍스트 (None이면 기본 샘플 사용)

        Returns:
            검증 결과 딕셔너리
        """
        self._log("\n=== 토크나이저 검증 ===")

        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        # 1. 기본 속성 검증
        vocab_size = len(tokenizer)
        model_max_length = tokenizer.model_max_length

        results['info']['vocab_size'] = vocab_size
        results['info']['model_max_length'] = model_max_length

        self._log(f"  - Vocab size: {vocab_size:,}")
        self._log(f"  - Max length: {model_max_length}")

        # Vocab size 검증
        if vocab_size < self.config['tokenizer']['vocab_size_min']:
            results['errors'].append(
                f"Vocab size too small: {vocab_size} < {self.config['tokenizer']['vocab_size_min']}"
            )
            results['passed'] = False

        # Max length 검증
        if model_max_length < self.config['tokenizer']['max_length_min']:
            results['warnings'].append(
                f"Max length might be too small: {model_max_length}"
            )
        elif model_max_length > self.config['tokenizer']['max_length_max']:
            results['warnings'].append(
                f"Max length might be too large: {model_max_length}"
            )

        # 2. Special tokens 검증
        self._log("\n  [Special Tokens 검증]")
        special_tokens = {
            'pad_token': tokenizer.pad_token,
            'unk_token': tokenizer.unk_token,
            'bos_token': tokenizer.bos_token,
            'eos_token': tokenizer.eos_token,
        }

        for token_name, token_value in special_tokens.items():
            if token_value is None:
                results['warnings'].append(f"{token_name} is not set")
                self._log(f"    - {token_name}: None (warning)")
            else:
                self._log(f"    - {token_name}: {token_value}")

        results['info']['special_tokens'] = special_tokens

        # 3. 샘플 텍스트 토크나이제이션 테스트
        if sample_texts is None:
            sample_texts = [
                "안녕하세요. 테스트 문장입니다.",
                "This is a test sentence.",
                "토크나이저가 정상적으로 작동하는지 확인합니다."
            ]

        self._log("\n  [샘플 토크나이제이션]")
        tokenization_results = []

        for i, text in enumerate(sample_texts[:3], 1):
            try:
                encoded = tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                )

                token_count = encoded['input_ids'].shape[1]
                tokenization_results.append({
                    'text': text[:50],
                    'token_count': token_count,
                    'success': True
                })

                self._log(f"    - Sample {i}: {token_count} tokens")

            except Exception as e:
                tokenization_results.append({
                    'text': text[:50],
                    'error': str(e),
                    'success': False
                })
                results['errors'].append(f"Tokenization failed for sample {i}: {e}")
                results['passed'] = False

        results['info']['tokenization_samples'] = tokenization_results

        # 결과 요약
        self._log(f"\n  토크나이저 검증: {'✓ PASS' if results['passed'] else '✗ FAIL'}")
        if results['errors']:
            self._log(f"  - Errors: {len(results['errors'])}")
        if results['warnings']:
            self._log(f"  - Warnings: {len(results['warnings'])}")

        self.validation_results['tokenization'] = results
        return results

    def check_learning_rate(
        self,
        learning_rate: float,
        model_size: Optional[int] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        학습률 적정성 검증

        Args:
            learning_rate: 학습률
            model_size: 모델 파라미터 수 (선택)
            batch_size: 배치 크기 (선택)

        Returns:
            검증 결과 딕셔너리
        """
        self._log("\n=== 학습률 검증 ===")

        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'info': {
                'learning_rate': learning_rate,
                'model_size': model_size,
                'batch_size': batch_size
            }
        }

        self._log(f"  - Learning rate: {learning_rate}")

        # 1. 범위 검증
        lr_config = self.config['learning_rate']

        if learning_rate < lr_config['min']:
            results['errors'].append(
                f"Learning rate too small: {learning_rate} < {lr_config['min']}"
            )
            results['passed'] = False

        elif learning_rate > lr_config['max']:
            results['errors'].append(
                f"Learning rate too large: {learning_rate} > {lr_config['max']}"
            )
            results['passed'] = False

        # 2. 권장 범위 검증
        if learning_rate < lr_config['recommended_min']:
            results['warnings'].append(
                f"Learning rate below recommended: {learning_rate} < {lr_config['recommended_min']}"
            )

        elif learning_rate > lr_config['recommended_max']:
            results['warnings'].append(
                f"Learning rate above recommended: {learning_rate} > {lr_config['recommended_max']}"
            )

        # 3. 모델 크기 기반 추천
        if model_size is not None:
            self._log(f"  - Model size: {model_size:,} params")

            # 대형 모델은 작은 학습률 권장
            if model_size > 1e9:  # 1B+ parameters
                recommended_lr = 1e-5
                if learning_rate > 5e-5:
                    results['warnings'].append(
                        f"For large model (>1B), consider smaller LR: {recommended_lr}"
                    )
            elif model_size > 1e8:  # 100M+ parameters
                recommended_lr = 2e-5
                if learning_rate > 1e-4:
                    results['warnings'].append(
                        f"For medium model (>100M), consider smaller LR: {recommended_lr}"
                    )

        # 4. 배치 크기 기반 추천
        if batch_size is not None:
            self._log(f"  - Batch size: {batch_size}")

            # 큰 배치는 큰 학습률 가능
            if batch_size >= 64 and learning_rate < 1e-5:
                results['warnings'].append(
                    "Large batch size allows larger learning rate"
                )
            elif batch_size <= 8 and learning_rate > 1e-4:
                results['warnings'].append(
                    "Small batch size may need smaller learning rate"
                )

        # 결과 요약
        self._log(f"\n  학습률 검증: {'✓ PASS' if results['passed'] else '✗ FAIL'}")
        if results['errors']:
            self._log(f"  - Errors: {len(results['errors'])}")
        if results['warnings']:
            self._log(f"  - Warnings: {len(results['warnings'])}")

        self.validation_results['learning_rate'] = results
        return results

    def check_generation_quality(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        sample_inputs: List[str],
        reference_outputs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        생성 품질 검증

        Args:
            model: 모델
            tokenizer: 토크나이저
            sample_inputs: 샘플 입력 텍스트
            reference_outputs: 참조 출력 (선택)

        Returns:
            검증 결과 딕셔너리
        """
        self._log("\n=== 생성 품질 검증 ===")

        results = {
            'passed': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        model.eval()
        device = next(model.parameters()).device

        generation_results = []

        for i, input_text in enumerate(sample_inputs[:5], 1):
            self._log(f"\n  [Sample {i}]")
            self._log(f"  Input: {input_text[:100]}...")

            try:
                # 생성
                inputs = tokenizer(
                    input_text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512
                ).to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=150,
                        num_beams=4,
                        early_stopping=True
                    )

                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                self._log(f"  Output: {generated_text[:100]}...")

                # 품질 검증
                quality_checks = self._analyze_generation_quality(generated_text)

                generation_results.append({
                    'input': input_text,
                    'output': generated_text,
                    'quality_checks': quality_checks
                })

                # 경고 및 에러 추가
                if quality_checks['repetition_ratio'] > self.config['generation']['max_repetition_ratio']:
                    results['warnings'].append(
                        f"Sample {i}: High repetition ratio {quality_checks['repetition_ratio']:.2%}"
                    )

                if quality_checks['length'] < self.config['generation']['min_length']:
                    results['warnings'].append(
                        f"Sample {i}: Output too short ({quality_checks['length']} tokens)"
                    )

            except Exception as e:
                self._log(f"  Error: {e}")
                results['errors'].append(f"Generation failed for sample {i}: {e}")
                results['passed'] = False

        results['info']['generation_results'] = generation_results

        # 결과 요약
        self._log(f"\n  생성 품질 검증: {'✓ PASS' if results['passed'] else '✗ FAIL'}")
        if results['errors']:
            self._log(f"  - Errors: {len(results['errors'])}")
        if results['warnings']:
            self._log(f"  - Warnings: {len(results['warnings'])}")

        self.validation_results['generation_quality'] = results
        return results

    def _analyze_generation_quality(self, text: str) -> Dict[str, Any]:
        """
        생성 텍스트 품질 분석

        Args:
            text: 생성된 텍스트

        Returns:
            품질 지표 딕셔너리
        """
        words = text.split()
        unique_words = set(words)

        # 반복 비율 계산
        if len(words) > 0:
            repetition_ratio = 1.0 - (len(unique_words) / len(words))
        else:
            repetition_ratio = 0.0

        return {
            'length': len(words),
            'unique_words': len(unique_words),
            'repetition_ratio': repetition_ratio,
            'has_content': len(text.strip()) > 0
        }

    def run_all_checks(
        self,
        tokenizer: PreTrainedTokenizer,
        learning_rate: float,
        model: Optional[PreTrainedModel] = None,
        sample_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        모든 검증 실행

        Args:
            tokenizer: 토크나이저
            learning_rate: 학습률
            model: 모델 (선택)
            sample_texts: 샘플 텍스트 (선택)

        Returns:
            전체 검증 결과
        """
        self._log("\n" + "=" * 60)
        self._log("베이스라인 검증 시작")
        self._log("=" * 60)

        # 1. 토크나이저 검증
        self.check_tokenization(tokenizer, sample_texts)

        # 2. 학습률 검증
        model_size = None
        if model is not None:
            model_size = sum(p.numel() for p in model.parameters())

        self.check_learning_rate(learning_rate, model_size=model_size)

        # 3. 생성 품질 검증 (모델이 있는 경우)
        if model is not None and sample_texts is not None:
            self.check_generation_quality(model, tokenizer, sample_texts)

        # 전체 결과 요약
        summary = self._create_summary()

        self._log("\n" + "=" * 60)
        self._log("검증 완료")
        self._log("=" * 60)
        self._log(f"전체 결과: {'✓ PASS' if summary['all_passed'] else '✗ FAIL'}")
        self._log(f"  - Passed: {summary['passed_count']}/{summary['total_checks']}")
        self._log(f"  - Errors: {summary['total_errors']}")
        self._log(f"  - Warnings: {summary['total_warnings']}")

        return summary

    def _create_summary(self) -> Dict[str, Any]:
        """검증 결과 요약 생성"""
        total_checks = len(self.validation_results)
        passed_count = sum(1 for r in self.validation_results.values() if r['passed'])
        total_errors = sum(len(r['errors']) for r in self.validation_results.values())
        total_warnings = sum(len(r['warnings']) for r in self.validation_results.values())

        return {
            'all_passed': passed_count == total_checks,
            'total_checks': total_checks,
            'passed_count': passed_count,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'details': self.validation_results
        }

    def save_results(self, output_path: str):
        """
        검증 결과를 JSON으로 저장

        Args:
            output_path: 저장 경로
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        summary = self._create_summary()

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self._log(f"\n검증 결과 저장: {output_path}")


# ==================== 팩토리 함수 ==================== #
def create_baseline_checker(
    config: Optional[Dict[str, Any]] = None,
    logger=None
) -> BaselineChecker:
    """
    베이스라인 검증기 생성 팩토리 함수

    Args:
        config: 검증 설정
        logger: Logger 인스턴스

    Returns:
        BaselineChecker 인스턴스
    """
    return BaselineChecker(config=config, logger=logger)


# ==================== 사용 예시 ==================== #
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # 검증기 생성
    checker = create_baseline_checker()

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # 샘플 텍스트
    samples = [
        "This is a test sentence.",
        "안녕하세요. 테스트입니다.",
        "베이스라인 검증을 수행합니다."
    ]

    # 검증 실행
    results = checker.run_all_checks(
        tokenizer=tokenizer,
        learning_rate=2e-5,
        sample_texts=samples
    )

    print(f"\n최종 결과: {'PASS' if results['all_passed'] else 'FAIL'}")
