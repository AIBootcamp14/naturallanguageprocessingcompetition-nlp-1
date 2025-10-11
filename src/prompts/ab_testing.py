"""
프롬프트 A/B 테스팅 프레임워크

PRD 15: 프롬프트 A/B 테스팅 전략
- 다양한 프롬프트 변형 비교
- 통계적 유의성 검증
- 최적 프롬프트 자동 선택
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import time
from pathlib import Path
import json

# 내부 모듈
from ..evaluation.metrics import RougeCalculator
from ..api.solar_api import SolarAPI


@dataclass
class PromptVariant:
    """
    프롬프트 변형 데이터 클래스

    Attributes:
        name: 변형 이름
        template: 프롬프트 템플릿
        description: 변형 설명
        results: 테스트 결과 리스트
        rouge_scores: ROUGE 점수 딕셔너리
        avg_latency: 평균 응답 시간 (초)
        token_usage: 평균 토큰 사용량
    """
    name: str
    template: str
    description: str = ""
    results: List[str] = field(default_factory=list)
    rouge_scores: Dict[str, float] = field(default_factory=dict)
    avg_latency: float = 0.0
    token_usage: int = 0


@dataclass
class ABTestResult:
    """
    A/B 테스트 결과 데이터 클래스

    Attributes:
        best_variant: 최고 성능 변형명
        all_scores: 모든 변형의 점수
        statistical_significance: 통계적 유의성 여부
        p_value: p-value (낮을수록 유의미)
        winner_margin: 1등과 2등의 차이
    """
    best_variant: str
    all_scores: Dict[str, Dict[str, float]]
    statistical_significance: bool
    p_value: float
    winner_margin: float


class PromptABTester:
    """
    프롬프트 A/B 테스팅 클래스

    여러 프롬프트 변형을 비교하여 최적의 프롬프트를 찾음
    """

    def __init__(
        self,
        api_client: Optional[SolarAPI] = None,
        rouge_calculator: Optional[RougeCalculator] = None,
        logger=None
    ):
        """
        Args:
            api_client: Solar API 클라이언트 (요약 생성용)
            rouge_calculator: ROUGE 계산기
            logger: Logger 인스턴스
        """
        self.variants: Dict[str, PromptVariant] = {}
        self.api_client = api_client
        self.rouge_calculator = rouge_calculator or RougeCalculator()
        self.logger = logger

        # 테스트 결과 저장
        self.test_results: Optional[ABTestResult] = None

    def _log(self, msg: str):
        """로깅 헬퍼"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def add_variant(
        self,
        name: str,
        template: str,
        description: str = ""
    ):
        """
        테스트 변형 추가

        Args:
            name: 변형 이름 (고유해야 함)
            template: 프롬프트 템플릿 ({dialogue} 플레이스홀더 포함)
            description: 변형 설명

        Raises:
            ValueError: 변형명이 중복되거나 템플릿이 잘못된 경우
        """
        # 중복 체크
        if name in self.variants:
            raise ValueError(f"변형명 '{name}'이 이미 존재합니다")

        # 템플릿 검증
        if '{dialogue}' not in template:
            raise ValueError("템플릿에 {dialogue} 플레이스홀더가 없습니다")

        # 변형 추가
        self.variants[name] = PromptVariant(
            name=name,
            template=template,
            description=description
        )

        self._log(f"✓ 변형 추가됨: {name}")

    def _generate_summary(
        self,
        dialogue: str,
        template: str
    ) -> Tuple[str, float]:
        """
        단일 대화 요약 생성 (내부 함수)

        Args:
            dialogue: 입력 대화
            template: 프롬프트 템플릿

        Returns:
            (요약 결과, 응답 시간)
        """
        if not self.api_client:
            raise RuntimeError("API 클라이언트가 설정되지 않았습니다")

        # 프롬프트 생성
        prompt = template.format(dialogue=dialogue)

        # 응답 시간 측정
        start_time = time.time()

        # API 호출 (Solar API 사용)
        summary = self.api_client.summarize(prompt)

        latency = time.time() - start_time

        return summary, latency

    def run_ab_test(
        self,
        dialogues: List[str],
        references: List[str],
        sample_size: Optional[int] = None
    ) -> ABTestResult:
        """
        A/B 테스트 실행

        Args:
            dialogues: 테스트 대화 리스트
            references: 정답 요약 리스트
            sample_size: 샘플 크기 (None이면 전체 사용)

        Returns:
            ABTestResult: 테스트 결과

        Raises:
            ValueError: 변형이 없거나 데이터 크기가 맞지 않는 경우
        """
        # 입력 검증
        if not self.variants:
            raise ValueError("테스트할 변형이 없습니다")

        if len(dialogues) != len(references):
            raise ValueError(
                f"대화와 정답 개수가 다릅니다: {len(dialogues)} vs {len(references)}"
            )

        # 샘플링
        if sample_size and sample_size < len(dialogues):
            indices = np.random.choice(len(dialogues), sample_size, replace=False)
            dialogues = [dialogues[i] for i in indices]
            references = [references[i] for i in indices]

        self._log(f"\n{'='*60}")
        self._log(f"A/B 테스트 시작")
        self._log(f"  - 변형 수: {len(self.variants)}")
        self._log(f"  - 테스트 샘플: {len(dialogues)}개")
        self._log(f"{'='*60}\n")

        # 각 변형 테스트
        all_scores = {}

        for variant_name, variant in self.variants.items():
            self._log(f"\n[{variant_name}] 테스트 중...")
            self._log(f"  설명: {variant.description}")

            predictions = []
            latencies = []

            # 각 대화에 대해 요약 생성
            for i, dialogue in enumerate(dialogues, 1):
                try:
                    summary, latency = self._generate_summary(
                        dialogue,
                        variant.template
                    )
                    predictions.append(summary)
                    latencies.append(latency)

                    if i % 10 == 0:
                        self._log(f"  진행: {i}/{len(dialogues)}")

                except Exception as e:
                    self._log(f"  ⚠️ 오류 발생 (샘플 {i}): {str(e)}")
                    predictions.append("")
                    latencies.append(0.0)

            # ROUGE 점수 계산
            scores = self.rouge_calculator.calculate_batch(
                predictions,
                references
            )

            # 결과 저장
            variant.results = predictions
            variant.rouge_scores = {
                'rouge1': scores['rouge1']['fmeasure'],
                'rouge2': scores['rouge2']['fmeasure'],
                'rougeL': scores['rougeL']['fmeasure'],
                'rouge_sum': sum([
                    scores['rouge1']['fmeasure'],
                    scores['rouge2']['fmeasure'],
                    scores['rougeL']['fmeasure']
                ])
            }
            variant.avg_latency = np.mean(latencies)

            all_scores[variant_name] = variant.rouge_scores

            # 결과 출력
            self._log(f"\n  결과:")
            self._log(f"    ROUGE-1: {variant.rouge_scores['rouge1']:.4f}")
            self._log(f"    ROUGE-2: {variant.rouge_scores['rouge2']:.4f}")
            self._log(f"    ROUGE-L: {variant.rouge_scores['rougeL']:.4f}")
            self._log(f"    ROUGE-Sum: {variant.rouge_scores['rouge_sum']:.4f}")
            self._log(f"    평균 응답시간: {variant.avg_latency:.3f}초")

        # 최고 성능 변형 찾기
        best_variant = max(
            self.variants.keys(),
            key=lambda name: self.variants[name].rouge_scores['rouge_sum']
        )

        # 통계적 유의성 검증
        rouge_sums = [
            variant.rouge_scores['rouge_sum']
            for variant in self.variants.values()
        ]

        # 1등과 2등 점수
        sorted_scores = sorted(rouge_sums, reverse=True)
        winner_margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        # 간단한 t-test (표준편차 기반)
        std = np.std(rouge_sums)
        p_value = std / (sorted_scores[0] + 1e-10)  # 정규화된 표준편차
        statistical_significance = p_value < 0.05 and winner_margin > 0.01

        # 결과 생성
        self.test_results = ABTestResult(
            best_variant=best_variant,
            all_scores=all_scores,
            statistical_significance=statistical_significance,
            p_value=p_value,
            winner_margin=winner_margin
        )

        # 최종 결과 출력
        self._log(f"\n{'='*60}")
        self._log(f"A/B 테스트 결과")
        self._log(f"{'='*60}")
        self._log(f"🏆 최고 성능: {best_variant}")
        self._log(f"   점수: {self.variants[best_variant].rouge_scores['rouge_sum']:.4f}")
        self._log(f"   승차: {winner_margin:.4f}")
        self._log(f"   통계적 유의성: {'✓ 유의미' if statistical_significance else '✗ 불충분'}")
        self._log(f"   p-value: {p_value:.4f}")
        self._log(f"{'='*60}\n")

        return self.test_results

    def get_best_variant(self) -> Optional[PromptVariant]:
        """
        최적 변형 반환

        Returns:
            PromptVariant: 최고 성능 변형 (테스트 미실행 시 None)
        """
        if not self.test_results:
            self._log("⚠️ A/B 테스트를 먼저 실행하세요")
            return None

        return self.variants[self.test_results.best_variant]

    def generate_report(
        self,
        output_path: Optional[str] = None
    ) -> str:
        """
        테스트 보고서 생성

        Args:
            output_path: 보고서 저장 경로 (None이면 출력만)

        Returns:
            str: 보고서 텍스트
        """
        if not self.test_results:
            return "A/B 테스트를 먼저 실행하세요"

        # 보고서 생성
        lines = []
        lines.append("=" * 80)
        lines.append("프롬프트 A/B 테스트 보고서")
        lines.append("=" * 80)
        lines.append("")

        # 테스트 개요
        lines.append("## 테스트 개요")
        lines.append(f"  - 테스트 변형 수: {len(self.variants)}")
        lines.append(f"  - 최고 성능 변형: {self.test_results.best_variant}")
        lines.append(f"  - 통계적 유의성: {'유의미' if self.test_results.statistical_significance else '불충분'}")
        lines.append("")

        # 변형별 상세 결과
        lines.append("## 변형별 결과")
        lines.append("")

        # 점수 순으로 정렬
        sorted_variants = sorted(
            self.variants.items(),
            key=lambda x: x[1].rouge_scores['rouge_sum'],
            reverse=True
        )

        for rank, (name, variant) in enumerate(sorted_variants, 1):
            lines.append(f"### {rank}. {name}")
            lines.append(f"   설명: {variant.description}")
            lines.append(f"   ROUGE-1: {variant.rouge_scores['rouge1']:.4f}")
            lines.append(f"   ROUGE-2: {variant.rouge_scores['rouge2']:.4f}")
            lines.append(f"   ROUGE-L: {variant.rouge_scores['rougeL']:.4f}")
            lines.append(f"   ROUGE-Sum: {variant.rouge_scores['rouge_sum']:.4f}")
            lines.append(f"   평균 응답시간: {variant.avg_latency:.3f}초")
            lines.append("")

        # 통계 분석
        lines.append("## 통계 분석")
        lines.append(f"   승차 (1등-2등): {self.test_results.winner_margin:.4f}")
        lines.append(f"   p-value: {self.test_results.p_value:.4f}")
        lines.append("")

        # 권장사항
        lines.append("## 권장사항")
        if self.test_results.statistical_significance:
            lines.append(f"✓ '{self.test_results.best_variant}' 변형을 사용하는 것을 권장합니다.")
        else:
            lines.append(f"⚠️ 변형 간 성능 차이가 통계적으로 유의미하지 않습니다.")
            lines.append(f"   더 많은 샘플로 테스트하거나 변형을 수정해보세요.")
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

    def export_results(
        self,
        output_path: str
    ):
        """
        테스트 결과를 JSON으로 내보내기

        Args:
            output_path: JSON 파일 저장 경로
        """
        if not self.test_results:
            self._log("⚠️ A/B 테스트를 먼저 실행하세요")
            return

        # JSON 직렬화 가능한 딕셔너리 생성
        export_data = {
            'best_variant': self.test_results.best_variant,
            'statistical_significance': self.test_results.statistical_significance,
            'p_value': self.test_results.p_value,
            'winner_margin': self.test_results.winner_margin,
            'variants': {}
        }

        for name, variant in self.variants.items():
            export_data['variants'][name] = {
                'name': variant.name,
                'template': variant.template,
                'description': variant.description,
                'rouge_scores': variant.rouge_scores,
                'avg_latency': variant.avg_latency
            }

        # JSON 저장
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        self._log(f"결과 저장됨: {output_path}")


def create_ab_tester(
    api_client: Optional[SolarAPI] = None,
    rouge_calculator: Optional[RougeCalculator] = None,
    logger=None
) -> PromptABTester:
    """
    PromptABTester 팩토리 함수

    Args:
        api_client: Solar API 클라이언트
        rouge_calculator: ROUGE 계산기
        logger: Logger 인스턴스

    Returns:
        PromptABTester 인스턴스
    """
    return PromptABTester(
        api_client=api_client,
        rouge_calculator=rouge_calculator,
        logger=logger
    )


# 사용 예시
if __name__ == "__main__":
    # A/B 테스터 생성
    tester = create_ab_tester()

    # 변형 추가
    tester.add_variant(
        name="zero_shot",
        template="다음 대화를 요약해주세요:\n\n{dialogue}\n\n요약:",
        description="기본 Zero-shot 프롬프트"
    )

    tester.add_variant(
        name="detailed",
        template="""아래 대화를 읽고 핵심 내용을 3-5문장으로 요약해주세요.

대화:
{dialogue}

요약:""",
        description="상세한 지시사항 포함"
    )

    tester.add_variant(
        name="structured",
        template="""[태스크] 대화 요약
[형식] 한 문단, 3-5문장
[스타일] 객관적, 간결함

대화 내용:
{dialogue}

요약 결과:""",
        description="구조화된 프롬프트"
    )

    # A/B 테스트 실행 (예시 데이터 필요)
    # result = tester.run_ab_test(dialogues, references)

    # 최고 변형 확인
    # best = tester.get_best_variant()
    # print(f"Best variant: {best.name}")

    # 보고서 생성
    # report = tester.generate_report("reports/ab_test_report.txt")
    # print(report)
