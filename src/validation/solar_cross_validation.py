"""
Solar API 교차 검증 시스템

PRD 16: Solar API 교차 검증 전략
- 모델 예측 vs Solar API 비교
- 품질 임계값 검증
- 이상치 탐지 및 보고서 생성
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
import time
from collections import defaultdict

# 내부 모듈
from ..evaluation.metrics import RougeCalculator
from ..api.solar_api import SolarAPI


@dataclass
class ValidationSample:
    """
    검증 샘플 데이터 클래스

    Attributes:
        index: 샘플 인덱스
        dialogue: 입력 대화
        model_prediction: 모델 예측
        solar_prediction: Solar API 예측
        rouge_score: ROUGE 점수 (모델 vs Solar)
        quality_score: 품질 점수 (0-1)
        is_outlier: 이상치 여부
        discrepancy: 불일치 정도
    """
    index: int
    dialogue: str
    model_prediction: str
    solar_prediction: str
    rouge_score: float = 0.0
    quality_score: float = 0.0
    is_outlier: bool = False
    discrepancy: float = 0.0


@dataclass
class ValidationReport:
    """
    검증 보고서 데이터 클래스

    Attributes:
        total_samples: 전체 샘플 수
        passed_samples: 통과 샘플 수
        failed_samples: 실패 샘플 수
        avg_rouge_score: 평균 ROUGE 점수
        avg_quality_score: 평균 품질 점수
        outlier_count: 이상치 개수
        quality_threshold: 품질 임계값
        pass_rate: 통과율
    """
    total_samples: int
    passed_samples: int
    failed_samples: int
    avg_rouge_score: float
    avg_quality_score: float
    outlier_count: int
    quality_threshold: float
    pass_rate: float
    samples: List[ValidationSample] = field(default_factory=list)


class SolarCrossValidator:
    """
    Solar API 교차 검증 클래스

    모델 예측과 Solar API 예측을 비교하여 품질을 검증
    """

    def __init__(
        self,
        solar_api: Optional[SolarAPI] = None,
        rouge_calculator: Optional[RougeCalculator] = None,
        logger=None
    ):
        """
        Args:
            solar_api: Solar API 클라이언트
            rouge_calculator: ROUGE 계산기
            logger: Logger 인스턴스
        """
        self.solar_api = solar_api
        self.rouge_calculator = rouge_calculator or RougeCalculator()
        self.logger = logger

        # 검증 결과 저장
        self.validation_samples: List[ValidationSample] = []
        self.report: Optional[ValidationReport] = None

    def _log(self, msg: str):
        """로깅 헬퍼"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def validate_against_solar(
        self,
        model_predictions: List[str],
        dialogues: List[str],
        sample_size: Optional[int] = None
    ) -> List[ValidationSample]:
        """
        모델 예측을 Solar API와 비교 검증

        Args:
            model_predictions: 모델 예측 리스트
            dialogues: 입력 대화 리스트
            sample_size: 검증 샘플 크기 (None이면 전체 사용)

        Returns:
            List[ValidationSample]: 검증 샘플 리스트

        Raises:
            ValueError: 입력 크기가 맞지 않거나 Solar API가 없는 경우
        """
        # 입력 검증
        if len(model_predictions) != len(dialogues):
            raise ValueError(
                f"예측과 대화 개수가 다릅니다: {len(model_predictions)} vs {len(dialogues)}"
            )

        if not self.solar_api:
            raise RuntimeError("Solar API 클라이언트가 설정되지 않았습니다")

        # 샘플링
        if sample_size and sample_size < len(dialogues):
            indices = np.random.choice(len(dialogues), sample_size, replace=False)
            dialogues = [dialogues[i] for i in indices]
            model_predictions = [model_predictions[i] for i in indices]
        else:
            indices = list(range(len(dialogues)))

        self._log(f"\n{'='*60}")
        self._log(f"Solar API 교차 검증 시작")
        self._log(f"  - 검증 샘플: {len(dialogues)}개")
        self._log(f"{'='*60}\n")

        # 각 샘플 검증
        validation_samples = []

        for i, (idx, dialogue, model_pred) in enumerate(
            zip(indices, dialogues, model_predictions), 1
        ):
            try:
                # Solar API로 예측 생성
                self._log(f"[{i}/{len(dialogues)}] 검증 중...")

                solar_pred = self.solar_api.summarize(dialogue)

                # ROUGE 점수 계산 (모델 vs Solar)
                rouge_scores = self.rouge_calculator.calculate_single(
                    model_pred,
                    solar_pred
                )

                # ROUGE-L F1 점수 사용
                rouge_score = rouge_scores['rougeL']['fmeasure']

                # 품질 점수 계산 (0-1)
                quality_score = self._calculate_quality_score(
                    model_pred,
                    solar_pred,
                    rouge_score
                )

                # 불일치 정도 계산
                discrepancy = 1.0 - rouge_score

                # 검증 샘플 생성
                sample = ValidationSample(
                    index=idx,
                    dialogue=dialogue,
                    model_prediction=model_pred,
                    solar_prediction=solar_pred,
                    rouge_score=rouge_score,
                    quality_score=quality_score,
                    discrepancy=discrepancy
                )

                validation_samples.append(sample)

                if i % 10 == 0:
                    avg_rouge = np.mean([s.rouge_score for s in validation_samples])
                    self._log(f"  평균 ROUGE-L: {avg_rouge:.4f}")

            except Exception as e:
                self._log(f"  ⚠️ 오류 발생 (샘플 {idx}): {str(e)}")
                # 에러 발생 시 빈 샘플 추가
                sample = ValidationSample(
                    index=idx,
                    dialogue=dialogue,
                    model_prediction=model_pred,
                    solar_prediction="",
                    rouge_score=0.0,
                    quality_score=0.0,
                    discrepancy=1.0
                )
                validation_samples.append(sample)

            # API Rate limit 고려
            time.sleep(0.1)

        self.validation_samples = validation_samples

        # 이상치 탐지
        self._detect_outliers()

        self._log(f"\n{'='*60}")
        self._log(f"교차 검증 완료")
        self._log(f"  - 평균 ROUGE-L: {np.mean([s.rouge_score for s in validation_samples]):.4f}")
        self._log(f"  - 평균 품질 점수: {np.mean([s.quality_score for s in validation_samples]):.4f}")
        self._log(f"  - 이상치: {sum(s.is_outlier for s in validation_samples)}개")
        self._log(f"{'='*60}\n")

        return validation_samples

    def _calculate_quality_score(
        self,
        model_pred: str,
        solar_pred: str,
        rouge_score: float
    ) -> float:
        """
        품질 점수 계산 (내부 함수)

        Args:
            model_pred: 모델 예측
            solar_pred: Solar 예측
            rouge_score: ROUGE 점수

        Returns:
            float: 품질 점수 (0-1)
        """
        # 1. ROUGE 점수 (가중치 0.6)
        rouge_component = rouge_score * 0.6

        # 2. 길이 유사도 (가중치 0.2)
        len_model = len(model_pred)
        len_solar = len(solar_pred)

        if len_solar == 0:
            length_similarity = 0.0
        else:
            length_ratio = min(len_model, len_solar) / max(len_model, len_solar)
            length_similarity = length_ratio * 0.2

        # 3. 빈 예측 체크 (가중치 0.2)
        empty_penalty = 0.0 if (model_pred and solar_pred) else 0.0
        non_empty_bonus = 0.2 if (model_pred and solar_pred) else 0.0

        # 품질 점수
        quality_score = rouge_component + length_similarity + non_empty_bonus

        return min(1.0, max(0.0, quality_score))

    def _detect_outliers(
        self,
        threshold: float = 2.0
    ):
        """
        이상치 탐지 (내부 함수)

        Args:
            threshold: 표준편차 배수 임계값
        """
        if not self.validation_samples:
            return

        # ROUGE 점수 기준 이상치 탐지
        rouge_scores = [s.rouge_score for s in self.validation_samples]

        mean_rouge = np.mean(rouge_scores)
        std_rouge = np.std(rouge_scores)

        for sample in self.validation_samples:
            # 평균에서 threshold * std 이상 떨어진 경우
            if abs(sample.rouge_score - mean_rouge) > threshold * std_rouge:
                sample.is_outlier = True

    def check_quality_threshold(
        self,
        predictions: List[str],
        threshold: float = 0.7
    ) -> Tuple[bool, List[int]]:
        """
        품질 임계값 체크

        Args:
            predictions: 모델 예측 리스트
            threshold: 품질 임계값 (0-1)

        Returns:
            (전체 통과 여부, 실패한 인덱스 리스트)

        Raises:
            RuntimeError: 검증이 실행되지 않은 경우
        """
        if not self.validation_samples:
            raise RuntimeError("먼저 validate_against_solar()를 실행하세요")

        self._log(f"\n품질 임계값 체크 (threshold={threshold})")
        self._log(f"{'='*60}")

        failed_indices = []

        for sample in self.validation_samples:
            if sample.quality_score < threshold:
                failed_indices.append(sample.index)

        passed_count = len(self.validation_samples) - len(failed_indices)
        pass_rate = passed_count / len(self.validation_samples)

        all_passed = len(failed_indices) == 0

        self._log(f"  통과: {passed_count}/{len(self.validation_samples)} ({pass_rate*100:.1f}%)")
        self._log(f"  실패: {len(failed_indices)}개")
        self._log(f"  결과: {'✓ 모두 통과' if all_passed else '✗ 일부 실패'}")
        self._log(f"{'='*60}\n")

        # 보고서 생성
        self.report = ValidationReport(
            total_samples=len(self.validation_samples),
            passed_samples=passed_count,
            failed_samples=len(failed_indices),
            avg_rouge_score=np.mean([s.rouge_score for s in self.validation_samples]),
            avg_quality_score=np.mean([s.quality_score for s in self.validation_samples]),
            outlier_count=sum(s.is_outlier for s in self.validation_samples),
            quality_threshold=threshold,
            pass_rate=pass_rate,
            samples=self.validation_samples
        )

        return all_passed, failed_indices

    def generate_validation_report(
        self,
        output_path: Optional[str] = None,
        include_samples: bool = False
    ) -> str:
        """
        검증 보고서 생성

        Args:
            output_path: 보고서 저장 경로 (None이면 출력만)
            include_samples: 개별 샘플 정보 포함 여부

        Returns:
            str: 보고서 텍스트
        """
        if not self.report:
            return "먼저 check_quality_threshold()를 실행하세요"

        # 보고서 생성
        lines = []
        lines.append("=" * 80)
        lines.append("Solar API 교차 검증 보고서")
        lines.append("=" * 80)
        lines.append("")

        # 전체 통계
        lines.append("## 전체 통계")
        lines.append(f"  - 전체 샘플: {self.report.total_samples}개")
        lines.append(f"  - 통과 샘플: {self.report.passed_samples}개")
        lines.append(f"  - 실패 샘플: {self.report.failed_samples}개")
        lines.append(f"  - 통과율: {self.report.pass_rate*100:.1f}%")
        lines.append(f"  - 품질 임계값: {self.report.quality_threshold}")
        lines.append("")

        # 점수 분석
        lines.append("## 점수 분석")
        lines.append(f"  - 평균 ROUGE-L: {self.report.avg_rouge_score:.4f}")
        lines.append(f"  - 평균 품질 점수: {self.report.avg_quality_score:.4f}")
        lines.append(f"  - 이상치 개수: {self.report.outlier_count}개")
        lines.append("")

        # 점수 분포
        rouge_scores = [s.rouge_score for s in self.report.samples]
        quality_scores = [s.quality_score for s in self.report.samples]

        lines.append("## 점수 분포")
        lines.append(f"  ROUGE-L 점수:")
        lines.append(f"    - 최소: {np.min(rouge_scores):.4f}")
        lines.append(f"    - 최대: {np.max(rouge_scores):.4f}")
        lines.append(f"    - 중앙값: {np.median(rouge_scores):.4f}")
        lines.append(f"    - 표준편차: {np.std(rouge_scores):.4f}")
        lines.append("")
        lines.append(f"  품질 점수:")
        lines.append(f"    - 최소: {np.min(quality_scores):.4f}")
        lines.append(f"    - 최대: {np.max(quality_scores):.4f}")
        lines.append(f"    - 중앙값: {np.median(quality_scores):.4f}")
        lines.append(f"    - 표준편차: {np.std(quality_scores):.4f}")
        lines.append("")

        # 실패 샘플 분석
        failed_samples = [
            s for s in self.report.samples
            if s.quality_score < self.report.quality_threshold
        ]

        if failed_samples:
            lines.append("## 실패 샘플 분석")
            lines.append(f"  - 총 {len(failed_samples)}개 샘플이 임계값 미달")
            lines.append("")

            # 상위 10개 실패 샘플
            failed_samples_sorted = sorted(
                failed_samples,
                key=lambda s: s.quality_score
            )[:10]

            lines.append("  하위 10개 샘플:")
            for i, sample in enumerate(failed_samples_sorted, 1):
                lines.append(f"    {i}. 인덱스 {sample.index}:")
                lines.append(f"       ROUGE-L: {sample.rouge_score:.4f}")
                lines.append(f"       품질 점수: {sample.quality_score:.4f}")
                lines.append(f"       불일치: {sample.discrepancy:.4f}")
            lines.append("")

        # 이상치 분석
        outliers = [s for s in self.report.samples if s.is_outlier]

        if outliers:
            lines.append("## 이상치 분석")
            lines.append(f"  - 총 {len(outliers)}개 이상치 발견")
            lines.append("")

            lines.append("  이상치 샘플:")
            for i, sample in enumerate(outliers[:10], 1):
                lines.append(f"    {i}. 인덱스 {sample.index}:")
                lines.append(f"       ROUGE-L: {sample.rouge_score:.4f}")
                lines.append(f"       품질 점수: {sample.quality_score:.4f}")
            lines.append("")

        # 권장사항
        lines.append("## 권장사항")
        if self.report.pass_rate >= 0.95:
            lines.append("  ✓ 모델이 Solar API와 높은 일치도를 보입니다.")
            lines.append("  ✓ 현재 모델을 사용해도 좋습니다.")
        elif self.report.pass_rate >= 0.80:
            lines.append("  ⚠️ 일부 샘플에서 품질 이슈가 있습니다.")
            lines.append("  ⚠️ 실패 샘플을 검토하고 모델을 개선하세요.")
        else:
            lines.append("  ✗ 많은 샘플에서 품질이 임계값 미달입니다.")
            lines.append("  ✗ 모델 재학습 또는 프롬프트 개선이 필요합니다.")
        lines.append("")

        # 개별 샘플 정보
        if include_samples:
            lines.append("## 개별 샘플 상세")
            lines.append("")

            for sample in self.report.samples[:20]:  # 상위 20개만
                lines.append(f"### 샘플 {sample.index}")
                lines.append(f"  ROUGE-L: {sample.rouge_score:.4f}")
                lines.append(f"  품질 점수: {sample.quality_score:.4f}")
                lines.append(f"  이상치: {'예' if sample.is_outlier else '아니오'}")
                lines.append(f"  모델 예측: {sample.model_prediction[:100]}...")
                lines.append(f"  Solar 예측: {sample.solar_prediction[:100]}...")
                lines.append("")

        lines.append("=" * 80)

        report = "\n".join(lines)

        # 파일 저장
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            self._log(f"보고서 저장됨: {output_path}")

        return report

    def export_validation_results(
        self,
        output_path: str
    ):
        """
        검증 결과를 JSON으로 내보내기

        Args:
            output_path: JSON 파일 저장 경로
        """
        if not self.report:
            self._log("⚠️ 먼저 check_quality_threshold()를 실행하세요")
            return

        # JSON 직렬화 가능한 딕셔너리 생성
        export_data = {
            'summary': {
                'total_samples': self.report.total_samples,
                'passed_samples': self.report.passed_samples,
                'failed_samples': self.report.failed_samples,
                'pass_rate': self.report.pass_rate,
                'avg_rouge_score': self.report.avg_rouge_score,
                'avg_quality_score': self.report.avg_quality_score,
                'outlier_count': self.report.outlier_count,
                'quality_threshold': self.report.quality_threshold
            },
            'samples': []
        }

        for sample in self.report.samples:
            export_data['samples'].append({
                'index': sample.index,
                'rouge_score': sample.rouge_score,
                'quality_score': sample.quality_score,
                'is_outlier': sample.is_outlier,
                'discrepancy': sample.discrepancy,
                'model_prediction': sample.model_prediction,
                'solar_prediction': sample.solar_prediction
            })

        # JSON 저장
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        self._log(f"결과 저장됨: {output_path}")


def create_solar_validator(
    solar_api: Optional[SolarAPI] = None,
    rouge_calculator: Optional[RougeCalculator] = None,
    logger=None
) -> SolarCrossValidator:
    """
    SolarCrossValidator 팩토리 함수

    Args:
        solar_api: Solar API 클라이언트
        rouge_calculator: ROUGE 계산기
        logger: Logger 인스턴스

    Returns:
        SolarCrossValidator 인스턴스
    """
    return SolarCrossValidator(
        solar_api=solar_api,
        rouge_calculator=rouge_calculator,
        logger=logger
    )


# 사용 예시
if __name__ == "__main__":
    from ..api.solar_api import create_solar_api

    # Solar API 및 검증기 생성
    solar_api = create_solar_api()
    validator = create_solar_validator(solar_api=solar_api)

    # 예시 데이터
    model_predictions = [
        "두 사람이 저녁 메뉴에 대해 이야기하고 있다.",
        "회의 일정을 조율하는 대화이다."
    ]
    dialogues = [
        "#Person1#: 오늘 저녁 뭐 먹을까? #Person2#: 피자 어때?",
        "#Person1#: 내일 회의 가능하세요? #Person2#: 네, 오후 2시는 어때요?"
    ]

    # Solar API와 교차 검증
    # samples = validator.validate_against_solar(model_predictions, dialogues)

    # 품질 임계값 체크
    # passed, failed_idx = validator.check_quality_threshold(model_predictions, threshold=0.7)

    # 보고서 생성
    # report = validator.generate_validation_report("reports/solar_validation_report.txt")
    # print(report)
