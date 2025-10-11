# ==================== 모델 Pruning 모듈 ==================== #
"""
모델 Pruning (가지치기)

PRD 17: 추론 가속화
- Magnitude-based Pruning: 가중치 크기 기반
- Structured Pruning: 구조적 가지치기
- 모델 크기 및 추론 속도 최적화
"""

# ---------------------- 라이브러리 임포트 ---------------------- #
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import json
import copy

# ---------------------- 외부 라이브러리 ---------------------- #
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


# ==================== ModelPruner 클래스 ==================== #
class ModelPruner:
    """모델 Pruning (가지치기) 클래스"""

    def __init__(
        self,
        pruning_method: str = "magnitude",
        amount: float = 0.3,
        structured: bool = False,
        logger=None
    ):
        """
        Args:
            pruning_method: Pruning 방법 ("magnitude", "random", "l1")
            amount: Pruning 비율 (0.0 ~ 1.0)
            structured: 구조적 pruning 여부
            logger: Logger 인스턴스
        """
        self.pruning_method = pruning_method
        self.amount = amount
        self.structured = structured
        self.logger = logger

        # Pruning 통계
        self.pruning_stats = {}

        self._log(f"ModelPruner 초기화 완료")
        self._log(f"  - Method: {pruning_method}")
        self._log(f"  - Amount: {amount*100:.1f}%")
        self._log(f"  - Structured: {structured}")

    def _log(self, msg: str):
        """로그 출력"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def magnitude_pruning(
        self,
        model: nn.Module,
        amount: Optional[float] = None,
        layers_to_prune: Optional[List[str]] = None
    ) -> nn.Module:
        """
        Magnitude-based Pruning: 가중치 크기 기반 가지치기

        Args:
            model: PyTorch 모델
            amount: Pruning 비율 (None이면 초기화 값 사용)
            layers_to_prune: Pruning할 레이어 이름 리스트 (None이면 전체)

        Returns:
            Pruning된 모델
        """
        amount = amount if amount is not None else self.amount

        self._log(f"\nMagnitude Pruning 시작")
        self._log(f"  - Amount: {amount*100:.1f}%")

        # 모델 복사
        pruned_model = copy.deepcopy(model)

        # Pruning할 레이어 수집
        layers = self._collect_layers(pruned_model, layers_to_prune)

        self._log(f"  - Target layers: {len(layers)}개")

        # 각 레이어에 대해 pruning 수행
        for name, module in layers:
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Weight pruning
                prune.l1_unstructured(module, name='weight', amount=amount)

                # Bias pruning (있는 경우)
                if hasattr(module, 'bias') and module.bias is not None:
                    prune.l1_unstructured(module, name='bias', amount=amount)

                self._log(f"    - Pruned: {name}")

        # Pruning 통계 수집
        self.pruning_stats = self._compute_pruning_stats(pruned_model)

        self._log(f"\nPruning 완료:")
        self._log(f"  - 전체 파라미터: {self.pruning_stats['total_params']:,}")
        self._log(f"  - Pruned 파라미터: {self.pruning_stats['pruned_params']:,}")
        self._log(f"  - Sparsity: {self.pruning_stats['sparsity']:.2%}")

        return pruned_model

    def structured_pruning(
        self,
        model: nn.Module,
        amount: Optional[float] = None,
        dim: int = 0,
        layers_to_prune: Optional[List[str]] = None
    ) -> nn.Module:
        """
        Structured Pruning: 구조적 가지치기 (전체 뉴런/필터 제거)

        Args:
            model: PyTorch 모델
            amount: Pruning 비율 (None이면 초기화 값 사용)
            dim: Pruning 차원 (0: 출력, 1: 입력)
            layers_to_prune: Pruning할 레이어 이름 리스트

        Returns:
            Pruning된 모델
        """
        amount = amount if amount is not None else self.amount

        self._log(f"\nStructured Pruning 시작")
        self._log(f"  - Amount: {amount*100:.1f}%")
        self._log(f"  - Dimension: {dim}")

        # 모델 복사
        pruned_model = copy.deepcopy(model)

        # Pruning할 레이어 수집
        layers = self._collect_layers(pruned_model, layers_to_prune)

        self._log(f"  - Target layers: {len(layers)}개")

        # 각 레이어에 대해 structured pruning 수행
        for name, module in layers:
            if isinstance(module, nn.Linear):
                # Linear layer: L2-norm 기반 structured pruning
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=amount,
                    n=2,  # L2 norm
                    dim=dim
                )
                self._log(f"    - Pruned: {name}")

            elif isinstance(module, nn.Conv2d):
                # Conv2d layer: 필터 단위 pruning
                prune.ln_structured(
                    module,
                    name='weight',
                    amount=amount,
                    n=2,
                    dim=dim  # 0: 출력 필터, 1: 입력 채널
                )
                self._log(f"    - Pruned: {name}")

        # Pruning 통계 수집
        self.pruning_stats = self._compute_pruning_stats(pruned_model)

        self._log(f"\nStructured Pruning 완료:")
        self._log(f"  - 전체 파라미터: {self.pruning_stats['total_params']:,}")
        self._log(f"  - Pruned 파라미터: {self.pruning_stats['pruned_params']:,}")
        self._log(f"  - Sparsity: {self.pruning_stats['sparsity']:.2%}")

        return pruned_model

    def global_pruning(
        self,
        model: nn.Module,
        amount: Optional[float] = None,
        layers_to_prune: Optional[List[str]] = None
    ) -> nn.Module:
        """
        Global Pruning: 전체 모델에 대해 통합적으로 pruning

        Args:
            model: PyTorch 모델
            amount: Pruning 비율
            layers_to_prune: Pruning할 레이어 이름 리스트

        Returns:
            Pruning된 모델
        """
        amount = amount if amount is not None else self.amount

        self._log(f"\nGlobal Pruning 시작")
        self._log(f"  - Amount: {amount*100:.1f}%")

        # 모델 복사
        pruned_model = copy.deepcopy(model)

        # Pruning할 레이어 수집
        layers = self._collect_layers(pruned_model, layers_to_prune)

        # Global pruning 수행
        parameters_to_prune = [
            (module, 'weight')
            for name, module in layers
            if isinstance(module, (nn.Linear, nn.Conv2d))
        ]

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

        self._log(f"  - Pruned layers: {len(parameters_to_prune)}개")

        # Pruning 통계 수집
        self.pruning_stats = self._compute_pruning_stats(pruned_model)

        self._log(f"\nGlobal Pruning 완료:")
        self._log(f"  - Sparsity: {self.pruning_stats['sparsity']:.2%}")

        return pruned_model

    def make_permanent(self, model: nn.Module) -> nn.Module:
        """
        Pruning을 영구적으로 적용 (mask 제거)

        Args:
            model: Pruning된 모델

        Returns:
            Pruning이 영구 적용된 모델
        """
        self._log("\nPruning 영구 적용")

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Weight mask 영구 적용
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')

                # Bias mask 영구 적용
                if hasattr(module, 'bias') and hasattr(module, 'bias_mask'):
                    prune.remove(module, 'bias')

        self._log("  - Pruning mask 제거 완료")
        return model

    def _collect_layers(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None
    ) -> List[Tuple[str, nn.Module]]:
        """
        Pruning할 레이어 수집

        Args:
            model: 모델
            layer_names: 특정 레이어 이름 (None이면 전체)

        Returns:
            (레이어 이름, 모듈) 튜플 리스트
        """
        layers = []

        for name, module in model.named_modules():
            # 레이어 필터링
            if layer_names is not None and name not in layer_names:
                continue

            # Linear 또는 Conv2d 레이어만
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers.append((name, module))

        return layers

    def _compute_pruning_stats(self, model: nn.Module) -> Dict[str, Any]:
        """
        Pruning 통계 계산

        Args:
            model: Pruning된 모델

        Returns:
            통계 딕셔너리
        """
        total_params = 0
        pruned_params = 0

        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Weight 통계
                if hasattr(module, 'weight'):
                    weight = module.weight.data
                    total_params += weight.numel()
                    pruned_params += (weight == 0).sum().item()

                # Bias 통계
                if hasattr(module, 'bias') and module.bias is not None:
                    bias = module.bias.data
                    total_params += bias.numel()
                    pruned_params += (bias == 0).sum().item()

        sparsity = pruned_params / total_params if total_params > 0 else 0.0

        return {
            'total_params': total_params,
            'pruned_params': pruned_params,
            'active_params': total_params - pruned_params,
            'sparsity': sparsity,
            'compression_ratio': 1.0 / (1.0 - sparsity) if sparsity < 1.0 else float('inf')
        }

    def get_pruning_stats(self) -> Dict[str, Any]:
        """Pruning 통계 반환"""
        return self.pruning_stats

    def save_pruning_stats(self, output_path: str):
        """
        Pruning 통계를 JSON으로 저장

        Args:
            output_path: 저장 경로
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.pruning_stats, f, indent=2)

        self._log(f"통계 저장: {output_path}")


# ==================== 팩토리 함수 ==================== #
def create_model_pruner(
    pruning_method: str = "magnitude",
    amount: float = 0.3,
    structured: bool = False,
    logger=None
) -> ModelPruner:
    """
    모델 Pruner 생성 팩토리 함수

    Args:
        pruning_method: Pruning 방법
        amount: Pruning 비율
        structured: 구조적 pruning 여부
        logger: Logger 인스턴스

    Returns:
        ModelPruner 인스턴스
    """
    return ModelPruner(
        pruning_method=pruning_method,
        amount=amount,
        structured=structured,
        logger=logger
    )


# ==================== 사용 예시 ==================== #
if __name__ == "__main__":
    # 간단한 모델 생성
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = SimpleModel()

    # Pruner 생성
    pruner = create_model_pruner(
        pruning_method="magnitude",
        amount=0.3
    )

    print("=" * 50)
    print("1. Magnitude Pruning")
    print("=" * 50)
    pruned_model_mag = pruner.magnitude_pruning(model)

    print("\n" + "=" * 50)
    print("2. Structured Pruning")
    print("=" * 50)
    pruned_model_struct = pruner.structured_pruning(model, amount=0.2)

    print("\n" + "=" * 50)
    print("3. Global Pruning")
    print("=" * 50)
    pruned_model_global = pruner.global_pruning(model, amount=0.4)

    # 통계 확인
    stats = pruner.get_pruning_stats()
    print(f"\n최종 통계:")
    print(f"  - Sparsity: {stats['sparsity']:.2%}")
    print(f"  - Compression: {stats['compression_ratio']:.2f}x")
