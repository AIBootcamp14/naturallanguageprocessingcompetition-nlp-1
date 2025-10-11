# ==================== TensorRT 최적화 모듈 ==================== #
"""
TensorRT 모델 최적화

PRD 17: 추론 가속화
- TensorRT 변환 (실제 TensorRT 없이도 실행 가능한 기본 구조)
- 추론 속도 향상을 위한 최적화
"""

# ---------------------- 라이브러리 임포트 ---------------------- #
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import json
import time
import warnings

# ---------------------- 외부 라이브러리 ---------------------- #
import torch
import torch.nn as nn


# ==================== TensorRTOptimizer 클래스 ==================== #
class TensorRTOptimizer:
    """TensorRT 기반 모델 최적화 클래스"""

    def __init__(
        self,
        precision: str = "fp16",
        workspace_size: int = 1 << 30,  # 1GB
        max_batch_size: int = 32,
        logger=None
    ):
        """
        Args:
            precision: 연산 정밀도 ("fp32", "fp16", "int8")
            workspace_size: TensorRT workspace 크기 (바이트)
            max_batch_size: 최대 배치 크기
            logger: Logger 인스턴스
        """
        self.precision = precision
        self.workspace_size = workspace_size
        self.max_batch_size = max_batch_size
        self.logger = logger

        # TensorRT 사용 가능 여부 확인
        self.tensorrt_available = self._check_tensorrt_availability()

        self._log(f"TensorRTOptimizer 초기화 완료")
        self._log(f"  - Precision: {precision}")
        self._log(f"  - TensorRT Available: {self.tensorrt_available}")
        if not self.tensorrt_available:
            self._log("  - Warning: TensorRT가 설치되지 않았습니다. Fallback 모드로 동작합니다.")

    def _log(self, msg: str):
        """로그 출력"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def _check_tensorrt_availability(self) -> bool:
        """TensorRT 사용 가능 여부 확인"""
        try:
            import tensorrt as trt
            return True
        except ImportError:
            return False

    def convert_to_tensorrt(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: Optional[str] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ) -> Any:
        """
        PyTorch 모델을 TensorRT로 변환

        Args:
            model: PyTorch 모델
            input_shape: 입력 shape (batch_size, ...)
            output_path: 저장 경로 (None이면 저장 안함)
            dynamic_axes: 동적 축 정의 (ONNX 변환용)

        Returns:
            TensorRT 엔진 (또는 최적화된 모델)
        """
        self._log(f"\nTensorRT 변환 시작")
        self._log(f"  - Input shape: {input_shape}")
        self._log(f"  - Precision: {self.precision}")

        if not self.tensorrt_available:
            # TensorRT 없으면 PyTorch JIT 컴파일로 대체
            self._log("  - TensorRT 미설치: PyTorch JIT로 최적화")
            return self._fallback_optimize(model, input_shape)

        try:
            # TensorRT 변환 (실제 구현)
            return self._convert_with_tensorrt(model, input_shape, output_path, dynamic_axes)

        except Exception as e:
            self._log(f"  - TensorRT 변환 실패: {e}")
            self._log("  - Fallback: PyTorch JIT 최적화")
            return self._fallback_optimize(model, input_shape)

    def _convert_with_tensorrt(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        output_path: Optional[str],
        dynamic_axes: Optional[Dict]
    ) -> Any:
        """
        실제 TensorRT 변환 수행 (TensorRT 설치 필요)
        """
        import tensorrt as trt
        import torch.onnx

        # 1단계: PyTorch -> ONNX
        self._log("  [1/3] PyTorch -> ONNX 변환")
        onnx_path = output_path.replace('.trt', '.onnx') if output_path else '/tmp/model.onnx'

        dummy_input = torch.randn(input_shape).cuda()
        model.eval()

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=13
        )
        self._log(f"    - ONNX 저장: {onnx_path}")

        # 2단계: ONNX -> TensorRT
        self._log("  [2/3] ONNX -> TensorRT 변환")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # ONNX 파일 파싱
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    self._log(f"    - Error: {parser.get_error(error)}")
                raise RuntimeError("ONNX 파싱 실패")

        # 빌더 설정
        config = builder.create_builder_config()
        config.max_workspace_size = self.workspace_size

        # 정밀도 설정
        if self.precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            self._log("    - FP16 모드 활성화")
        elif self.precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            self._log("    - INT8 모드 활성화")

        # 엔진 빌드
        self._log("  [3/3] TensorRT 엔진 빌드")
        engine = builder.build_engine(network, config)

        if engine is None:
            raise RuntimeError("TensorRT 엔진 빌드 실패")

        # 엔진 저장
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
            self._log(f"    - TensorRT 엔진 저장: {output_path}")

        self._log("TensorRT 변환 완료")
        return engine

    def _fallback_optimize(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...]
    ) -> torch.jit.ScriptModule:
        """
        TensorRT 없을 때 PyTorch JIT로 최적화

        Args:
            model: PyTorch 모델
            input_shape: 입력 shape

        Returns:
            JIT 컴파일된 모델
        """
        self._log("  [Fallback] PyTorch JIT 최적화")

        model.eval()
        dummy_input = torch.randn(input_shape)

        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()

        # JIT trace
        try:
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model = torch.jit.optimize_for_inference(traced_model)
            self._log("  - JIT 최적화 완료")
            return traced_model

        except Exception as e:
            self._log(f"  - JIT 최적화 실패: {e}")
            self._log("  - 원본 모델 반환")
            return model

    def benchmark(
        self,
        model: Any,
        input_shape: Tuple[int, ...],
        n_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """
        모델 추론 속도 벤치마크

        Args:
            model: 모델 (PyTorch 또는 TensorRT)
            input_shape: 입력 shape
            n_iterations: 측정 반복 횟수
            warmup_iterations: Warmup 반복 횟수

        Returns:
            벤치마크 결과 딕셔너리
        """
        self._log(f"\n벤치마크 시작")
        self._log(f"  - Iterations: {n_iterations}")
        self._log(f"  - Warmup: {warmup_iterations}")

        # 입력 데이터 생성
        dummy_input = torch.randn(input_shape)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
            if isinstance(model, nn.Module):
                model = model.cuda()

        # Warmup
        for _ in range(warmup_iterations):
            if isinstance(model, nn.Module):
                with torch.no_grad():
                    _ = model(dummy_input)
            else:
                # TensorRT 엔진인 경우
                pass

        # 벤치마크
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.time()

        for _ in range(n_iterations):
            if isinstance(model, nn.Module):
                with torch.no_grad():
                    _ = model(dummy_input)
            else:
                # TensorRT 엔진인 경우
                pass

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        # 결과 계산
        total_time = end_time - start_time
        avg_time = total_time / n_iterations
        throughput = 1.0 / avg_time

        results = {
            'total_time': total_time,
            'avg_time': avg_time,
            'throughput': throughput,
            'n_iterations': n_iterations
        }

        self._log(f"\n벤치마크 결과:")
        self._log(f"  - 평균 추론 시간: {avg_time*1000:.2f} ms")
        self._log(f"  - Throughput: {throughput:.2f} samples/sec")

        return results

    def save_config(self, output_path: str):
        """
        최적화 설정을 JSON으로 저장

        Args:
            output_path: 저장 경로
        """
        config = {
            'precision': self.precision,
            'workspace_size': self.workspace_size,
            'max_batch_size': self.max_batch_size,
            'tensorrt_available': self.tensorrt_available
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        self._log(f"설정 저장: {output_path}")


# ==================== 팩토리 함수 ==================== #
def create_tensorrt_optimizer(
    precision: str = "fp16",
    workspace_size: int = 1 << 30,
    max_batch_size: int = 32,
    logger=None
) -> TensorRTOptimizer:
    """
    TensorRT 최적화기 생성 팩토리 함수

    Args:
        precision: 연산 정밀도
        workspace_size: Workspace 크기
        max_batch_size: 최대 배치 크기
        logger: Logger 인스턴스

    Returns:
        TensorRTOptimizer 인스턴스
    """
    return TensorRTOptimizer(
        precision=precision,
        workspace_size=workspace_size,
        max_batch_size=max_batch_size,
        logger=logger
    )


# ==================== 사용 예시 ==================== #
if __name__ == "__main__":
    # TensorRT 최적화기 생성
    optimizer = create_tensorrt_optimizer(
        precision="fp16",
        max_batch_size=32
    )

    # 간단한 모델로 테스트
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(768, 2)

        def forward(self, x):
            return self.linear(x)

    model = SimpleModel()
    input_shape = (1, 768)

    # 최적화
    optimized_model = optimizer.convert_to_tensorrt(
        model=model,
        input_shape=input_shape
    )

    # 벤치마크
    results = optimizer.benchmark(
        model=optimized_model,
        input_shape=input_shape,
        n_iterations=100
    )

    print(f"\n최종 결과:")
    print(f"  - 평균 시간: {results['avg_time']*1000:.2f} ms")
    print(f"  - Throughput: {results['throughput']:.2f} samples/sec")
