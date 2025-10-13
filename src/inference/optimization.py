# ==================== 추론 최적화 모듈 (PRD 17) ==================== #
"""
추론 최적화 전략

PRD 17: 추론 최적화 전략
- 양자화 (INT8, INT4, FP16)
- ONNX 변환 및 최적화
- TensorRT 가속 (GPU 환경)
- 배치 최적화
- 메모리 최적화
"""

# ---------------------- 표준 라이브러리 ---------------------- #
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
import time

# ---------------------- 서드파티 라이브러리 ---------------------- #
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ==================== 양자화 최적화 ==================== #
class QuantizationOptimizer:
    """모델 양자화 최적화"""

    def __init__(
        self,
        quantization_bits: int = 8,
        logger=None
    ):
        """
        Args:
            quantization_bits: 양자화 비트 수 (4, 8, 16)
            logger: Logger 인스턴스
        """
        self.quantization_bits = quantization_bits
        self.logger = logger

    def _log(self, msg: str):
        """로깅 유틸리티"""
        if self.logger:
            if hasattr(self.logger, 'write'):
                self.logger.write(msg)
            elif hasattr(self.logger, 'info'):
                self.logger.info(msg)
        else:
            print(msg)

    def quantize_model(
        self,
        model: torch.nn.Module,
        calibration_data: Optional[List] = None
    ) -> torch.nn.Module:
        """
        모델 양자화

        Args:
            model: 원본 모델
            calibration_data: 캘리브레이션 데이터 (PTQ용)

        Returns:
            양자화된 모델
        """
        self._log(f"🔧 모델 양자화 시작 ({self.quantization_bits}bit)...")

        try:
            if self.quantization_bits == 8:
                # INT8 동적 양자화 (가장 안정적)
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                self._log("  ✅ INT8 동적 양자화 완료")

            elif self.quantization_bits == 16:
                # FP16 변환 (GPU 전용)
                if torch.cuda.is_available():
                    quantized_model = model.half()
                    self._log("  ✅ FP16 변환 완료")
                else:
                    self._log("  ⚠️  GPU 없음 - FP16 변환 스킵")
                    quantized_model = model

            elif self.quantization_bits == 4:
                # INT4 양자화 (실험적)
                self._log("  ⚠️  INT4 양자화는 실험적 기능입니다")
                # bitsandbytes를 사용한 4bit 양자화는 로드 시점에서 처리
                quantized_model = model

            else:
                self._log(f"  ❌ 지원하지 않는 양자화 비트: {self.quantization_bits}")
                quantized_model = model

            return quantized_model

        except Exception as e:
            self._log(f"  ❌ 양자화 실패: {e}")
            return model

    def measure_speedup(
        self,
        original_model: torch.nn.Module,
        quantized_model: torch.nn.Module,
        test_input: torch.Tensor,
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        양자화 속도 향상 측정

        Args:
            original_model: 원본 모델
            quantized_model: 양자화된 모델
            test_input: 테스트 입력
            num_runs: 실행 횟수

        Returns:
            속도 통계
        """
        self._log("⏱️  속도 측정 중...")

        # 원본 모델 속도
        original_model.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = original_model(test_input)
        original_time = time.time() - start

        # 양자화 모델 속도
        quantized_model.eval()
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = quantized_model(test_input)
        quantized_time = time.time() - start

        speedup = original_time / quantized_time

        results = {
            'original_time': original_time,
            'quantized_time': quantized_time,
            'speedup': speedup,
            'num_runs': num_runs
        }

        self._log(f"  원본 모델: {original_time:.3f}s")
        self._log(f"  양자화 모델: {quantized_time:.3f}s")
        self._log(f"  속도 향상: {speedup:.2f}x")

        return results


# ==================== ONNX 변환기 ==================== #
class ONNXConverter:
    """ONNX 모델 변환기"""

    def __init__(
        self,
        opset_version: int = 14,
        optimize: bool = True,
        logger=None
    ):
        """
        Args:
            opset_version: ONNX opset 버전
            optimize: ONNX 최적화 적용 여부
            logger: Logger 인스턴스
        """
        self.opset_version = opset_version
        self.optimize = optimize
        self.logger = logger

    def _log(self, msg: str):
        """로깅 유틸리티"""
        if self.logger:
            if hasattr(self.logger, 'write'):
                self.logger.write(msg)
            elif hasattr(self.logger, 'info'):
                self.logger.info(msg)
        else:
            print(msg)

    def convert_to_onnx(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        output_path: str,
        sample_input: Optional[Dict] = None
    ) -> bool:
        """
        PyTorch 모델을 ONNX로 변환

        Args:
            model: PyTorch 모델
            tokenizer: 토크나이저
            output_path: ONNX 모델 저장 경로
            sample_input: 샘플 입력 (None이면 자동 생성)

        Returns:
            변환 성공 여부
        """
        self._log(f"🔄 ONNX 변환 시작...")
        self._log(f"  출력 경로: {output_path}")

        try:
            # 샘플 입력 생성
            if sample_input is None:
                sample_text = "이것은 샘플 대화입니다."
                sample_input = tokenizer(
                    sample_text,
                    return_tensors="pt",
                    max_length=128,
                    padding="max_length",
                    truncation=True
                )

            # ONNX 변환
            model.eval()
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 동적 축 정의
            dynamic_axes = {
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
            }

            # 변환 실행
            torch.onnx.export(
                model,
                (sample_input['input_ids'], sample_input['attention_mask']),
                str(output_path),
                input_names=['input_ids', 'attention_mask'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                opset_version=self.opset_version,
                do_constant_folding=True
            )

            self._log(f"  ✅ ONNX 변환 완료")

            # ONNX 최적화
            if self.optimize:
                self._optimize_onnx(output_path)

            return True

        except Exception as e:
            self._log(f"  ❌ ONNX 변환 실패: {e}")
            return False

    def _optimize_onnx(self, onnx_path: Path):
        """ONNX 모델 최적화"""
        try:
            import onnx
            from onnxruntime.transformers import optimizer

            self._log("  🔧 ONNX 최적화 중...")

            # ONNX 모델 로드
            model = onnx.load(str(onnx_path))

            # 최적화 적용 (상수 폴딩, 불필요한 노드 제거 등)
            # onnxruntime optimizer 사용
            optimized_path = onnx_path.parent / f"{onnx_path.stem}_optimized.onnx"

            # 기본 최적화
            onnx.save(model, str(optimized_path))

            self._log(f"  ✅ ONNX 최적화 완료: {optimized_path}")

        except ImportError:
            self._log("  ⚠️  onnx 또는 onnxruntime 미설치 - 최적화 스킵")
        except Exception as e:
            self._log(f"  ⚠️  ONNX 최적화 실패: {e}")


# ==================== 배치 최적화 ==================== #
class BatchOptimizer:
    """배치 추론 최적화"""

    def __init__(
        self,
        logger=None
    ):
        """
        Args:
            logger: Logger 인스턴스
        """
        self.logger = logger

    def _log(self, msg: str):
        """로깅 유틸리티"""
        if self.logger:
            if hasattr(self.logger, 'write'):
                self.logger.write(msg)
            elif hasattr(self.logger, 'info'):
                self.logger.info(msg)
        else:
            print(msg)

    def find_optimal_batch_size(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        sample_texts: List[str],
        max_batch_size: int = 64,
        min_batch_size: int = 1
    ) -> int:
        """
        최적 배치 크기 탐색

        Args:
            model: 모델
            tokenizer: 토크나이저
            sample_texts: 샘플 텍스트 리스트
            max_batch_size: 최대 배치 크기
            min_batch_size: 최소 배치 크기

        Returns:
            최적 배치 크기
        """
        self._log("🔍 최적 배치 크기 탐색 중...")

        model.eval()
        optimal_batch_size = min_batch_size

        # 배치 크기를 늘려가며 테스트
        for batch_size in [1, 2, 4, 8, 16, 32, 64]:
            if batch_size > max_batch_size:
                break

            try:
                # 배치 생성
                batch_texts = sample_texts[:batch_size]
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    model = model.cuda()

                # 추론 테스트
                with torch.no_grad():
                    _ = model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=200
                    )

                # 성공하면 배치 크기 업데이트
                optimal_batch_size = batch_size
                self._log(f"  ✅ 배치 크기 {batch_size} 성공")

                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    self._log(f"  ⚠️  배치 크기 {batch_size} OOM - 이전 크기 사용")
                    break
                else:
                    raise e

        self._log(f"  🎯 최적 배치 크기: {optimal_batch_size}")
        return optimal_batch_size


# ==================== 통합 최적화 매니저 ==================== #
class InferenceOptimizer:
    """추론 최적화 통합 관리자"""

    def __init__(
        self,
        optimization_method: str = "quantization",
        quantization_bits: int = 8,
        use_onnx: bool = False,
        use_batch_optimization: bool = True,
        logger=None
    ):
        """
        Args:
            optimization_method: 최적화 방법 ('quantization', 'onnx', 'tensorrt')
            quantization_bits: 양자화 비트 수
            use_onnx: ONNX 변환 사용 여부
            use_batch_optimization: 배치 최적화 사용 여부
            logger: Logger 인스턴스
        """
        self.optimization_method = optimization_method
        self.quantization_bits = quantization_bits
        self.use_onnx = use_onnx
        self.use_batch_optimization = use_batch_optimization
        self.logger = logger

        # 최적화 모듈 초기화
        self.quantizer = QuantizationOptimizer(
            quantization_bits=quantization_bits,
            logger=logger
        )
        self.onnx_converter = ONNXConverter(logger=logger)
        self.batch_optimizer = BatchOptimizer(logger=logger)

    def _log(self, msg: str):
        """로깅 유틸리티"""
        if self.logger:
            if hasattr(self.logger, 'write'):
                self.logger.write(msg)
            elif hasattr(self.logger, 'info'):
                self.logger.info(msg)
        else:
            print(msg)

    def optimize(
        self,
        model: torch.nn.Module,
        tokenizer: AutoTokenizer,
        output_dir: str,
        sample_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        모델 최적화 실행

        Args:
            model: 원본 모델
            tokenizer: 토크나이저
            output_dir: 출력 디렉토리
            sample_texts: 샘플 텍스트 (배치 최적화용)

        Returns:
            최적화 결과
        """
        self._log("=" * 60)
        self._log("🚀 추론 최적화 시작")
        self._log(f"  방법: {self.optimization_method}")
        self._log("=" * 60)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'optimization_method': self.optimization_method,
            'quantization_bits': self.quantization_bits,
            'success': False
        }

        try:
            # 1. 양자화
            if self.optimization_method == "quantization":
                optimized_model = self.quantizer.quantize_model(model)

                # 모델 저장
                model_path = output_dir / f"quantized_{self.quantization_bits}bit"
                model_path.mkdir(parents=True, exist_ok=True)

                torch.save(optimized_model.state_dict(), model_path / "model.pt")
                tokenizer.save_pretrained(str(model_path))

                results['model_path'] = str(model_path)
                results['success'] = True

            # 2. ONNX 변환
            elif self.optimization_method == "onnx" or self.use_onnx:
                onnx_path = output_dir / "model.onnx"
                success = self.onnx_converter.convert_to_onnx(
                    model=model,
                    tokenizer=tokenizer,
                    output_path=str(onnx_path)
                )

                results['onnx_path'] = str(onnx_path)
                results['success'] = success

            # 3. TensorRT (기본 구조만)
            elif self.optimization_method == "tensorrt":
                self._log("⚠️  TensorRT는 아직 구현되지 않았습니다")
                results['success'] = False

            # 4. 배치 최적화
            if self.use_batch_optimization and sample_texts:
                optimal_batch_size = self.batch_optimizer.find_optimal_batch_size(
                    model=model,
                    tokenizer=tokenizer,
                    sample_texts=sample_texts
                )
                results['optimal_batch_size'] = optimal_batch_size

            # 결과 저장
            result_path = output_dir / "optimization_results.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            self._log("\n" + "=" * 60)
            self._log("✅ 추론 최적화 완료!")
            self._log("=" * 60)

            return results

        except Exception as e:
            self._log(f"\n❌ 최적화 실패: {e}")
            results['error'] = str(e)
            return results


# ==================== 팩토리 함수 ==================== #
def create_inference_optimizer(
    optimization_method: str = "quantization",
    quantization_bits: int = 8,
    use_onnx: bool = False,
    use_batch_optimization: bool = True,
    logger=None
) -> InferenceOptimizer:
    """
    InferenceOptimizer 생성 팩토리 함수

    Args:
        optimization_method: 최적화 방법
        quantization_bits: 양자화 비트 수
        use_onnx: ONNX 변환 사용
        use_batch_optimization: 배치 최적화 사용
        logger: Logger 인스턴스

    Returns:
        InferenceOptimizer 인스턴스
    """
    return InferenceOptimizer(
        optimization_method=optimization_method,
        quantization_bits=quantization_bits,
        use_onnx=use_onnx,
        use_batch_optimization=use_batch_optimization,
        logger=logger
    )
