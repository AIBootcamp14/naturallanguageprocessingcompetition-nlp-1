# ==================== 계층적 Config 시스템 ==================== #
"""
계층적 Config 로더

PRD 19: 계층적 Config 구조
- configs/base/: 공통 설정
- configs/models/: 모델별 설정
- configs/strategies/: 전략별 설정
- 설정 병합 및 오버라이드 지원
"""

# ---------------------- 라이브러리 임포트 ---------------------- #
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import yaml
import copy


# ==================== HierarchicalConfigLoader 클래스 ==================== #
class HierarchicalConfigLoader:
    """계층적 Config 로더 클래스"""

    def __init__(
        self,
        config_root: str = "configs",
        logger=None
    ):
        """
        Args:
            config_root: Config 루트 디렉토리
            logger: Logger 인스턴스
        """
        self.config_root = Path(config_root)
        self.logger = logger

        # 계층 구조 정의
        self.hierarchy = {
            'base': self.config_root / 'base',
            'models': self.config_root / 'models',
            'strategies': self.config_root / 'strategies'
        }

        # 로드된 설정 캐시
        self.cache = {}

        self._log("HierarchicalConfigLoader 초기화 완료")
        self._log(f"  - Config root: {self.config_root}")

    def _log(self, msg: str):
        """로그 출력"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def load_config(
        self,
        model_name: Optional[str] = None,
        strategy_name: Optional[str] = None,
        override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        계층적으로 Config 로드

        우선순위 (낮음 -> 높음):
        1. base config
        2. model config
        3. strategy config
        4. override params

        Args:
            model_name: 모델 이름 (예: "solar", "llama")
            strategy_name: 전략 이름 (예: "baseline", "advanced")
            override: 직접 오버라이드할 설정

        Returns:
            병합된 Config 딕셔너리
        """
        self._log(f"\nConfig 로드 시작")
        self._log(f"  - Model: {model_name or 'default'}")
        self._log(f"  - Strategy: {strategy_name or 'default'}")

        # 1. Base config 로드
        base_config = self._load_base_config()

        # 2. Model config 로드 및 병합
        if model_name:
            model_config = self._load_model_config(model_name)
            base_config = self.merge_configs(base_config, model_config)

        # 3. Strategy config 로드 및 병합
        if strategy_name:
            strategy_config = self._load_strategy_config(strategy_name)
            base_config = self.merge_configs(base_config, strategy_config)

        # 4. Override 적용
        if override:
            base_config = self.merge_configs(base_config, override)

        self._log(f"  - Config 로드 완료")

        return base_config

    def _load_base_config(self) -> Dict[str, Any]:
        """
        Base config 로드

        configs/base/default.yaml 또는 configs/base/default.json
        """
        base_dir = self.hierarchy['base']

        # 캐시 확인
        cache_key = 'base_default'
        if cache_key in self.cache:
            return copy.deepcopy(self.cache[cache_key])

        # YAML 우선 시도
        yaml_path = base_dir / 'default.yaml'
        if yaml_path.exists():
            config = self._load_yaml(yaml_path)
            self.cache[cache_key] = config
            self._log(f"  - Base config 로드: {yaml_path}")
            return copy.deepcopy(config)

        # JSON 시도
        json_path = base_dir / 'default.json'
        if json_path.exists():
            config = self._load_json(json_path)
            self.cache[cache_key] = config
            self._log(f"  - Base config 로드: {json_path}")
            return copy.deepcopy(config)

        # 기본 설정 반환
        self._log(f"  - Base config 없음, 기본값 사용")
        return self._get_default_base_config()

    def _load_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Model config 로드

        configs/models/{model_name}.yaml 또는 .json
        """
        model_dir = self.hierarchy['models']

        # 캐시 확인
        cache_key = f'model_{model_name}'
        if cache_key in self.cache:
            return copy.deepcopy(self.cache[cache_key])

        # YAML 우선
        yaml_path = model_dir / f'{model_name}.yaml'
        if yaml_path.exists():
            config = self._load_yaml(yaml_path)
            self.cache[cache_key] = config
            self._log(f"  - Model config 로드: {yaml_path}")
            return copy.deepcopy(config)

        # JSON 시도
        json_path = model_dir / f'{model_name}.json'
        if json_path.exists():
            config = self._load_json(json_path)
            self.cache[cache_key] = config
            self._log(f"  - Model config 로드: {json_path}")
            return copy.deepcopy(config)

        self._log(f"  - Warning: Model config not found: {model_name}")
        return {}

    def _load_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        Strategy config 로드

        configs/strategies/{strategy_name}.yaml 또는 .json
        """
        strategy_dir = self.hierarchy['strategies']

        # 캐시 확인
        cache_key = f'strategy_{strategy_name}'
        if cache_key in self.cache:
            return copy.deepcopy(self.cache[cache_key])

        # YAML 우선
        yaml_path = strategy_dir / f'{strategy_name}.yaml'
        if yaml_path.exists():
            config = self._load_yaml(yaml_path)
            self.cache[cache_key] = config
            self._log(f"  - Strategy config 로드: {yaml_path}")
            return copy.deepcopy(config)

        # JSON 시도
        json_path = strategy_dir / f'{strategy_name}.json'
        if json_path.exists():
            config = self._load_json(json_path)
            self.cache[cache_key] = config
            self._log(f"  - Strategy config 로드: {json_path}")
            return copy.deepcopy(config)

        self._log(f"  - Warning: Strategy config not found: {strategy_name}")
        return {}

    def _load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """YAML 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            self._log(f"  - Error loading YAML {file_path}: {e}")
            return {}

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """JSON 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self._log(f"  - Error loading JSON {file_path}: {e}")
            return {}

    def merge_configs(
        self,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Config 병합 (deep merge)

        override의 값이 base의 값을 덮어씀

        Args:
            base: 기본 설정
            override: 오버라이드 설정

        Returns:
            병합된 설정
        """
        merged = copy.deepcopy(base)

        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # 중첩된 딕셔너리는 재귀적으로 병합
                merged[key] = self.merge_configs(merged[key], value)
            else:
                # 그 외는 덮어쓰기
                merged[key] = copy.deepcopy(value)

        return merged

    def _get_default_base_config(self) -> Dict[str, Any]:
        """기본 Base config 반환"""
        return {
            'model': {
                'name': 'default',
                'max_length': 512,
                'device': 'cuda'
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 2e-5,
                'num_epochs': 3,
                'warmup_ratio': 0.1,
                'weight_decay': 0.01
            },
            'generation': {
                'max_length': 150,
                'num_beams': 4,
                'temperature': 1.0,
                'top_p': 0.9,
                'top_k': 50
            },
            'data': {
                'train_file': 'data/train.json',
                'val_file': 'data/val.json',
                'test_file': 'data/test.json'
            },
            'logging': {
                'log_level': 'INFO',
                'save_steps': 500,
                'eval_steps': 500
            }
        }

    def list_available_configs(self) -> Dict[str, List[str]]:
        """
        사용 가능한 Config 파일 목록 반환

        Returns:
            {
                'base': [...],
                'models': [...],
                'strategies': [...]
            }
        """
        available = {}

        for config_type, config_dir in self.hierarchy.items():
            if not config_dir.exists():
                available[config_type] = []
                continue

            # YAML, JSON 파일 찾기
            configs = []
            for ext in ['.yaml', '.yml', '.json']:
                configs.extend([
                    f.stem for f in config_dir.glob(f'*{ext}')
                ])

            available[config_type] = sorted(set(configs))

        return available

    def save_config(
        self,
        config: Dict[str, Any],
        output_path: str,
        format: str = 'yaml'
    ):
        """
        Config를 파일로 저장

        Args:
            config: 저장할 설정
            output_path: 저장 경로
            format: 파일 형식 ('yaml' 또는 'json')
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if format == 'yaml':
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        elif format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        self._log(f"Config 저장: {output_path}")

    def create_config_template(
        self,
        config_type: str,
        name: str,
        format: str = 'yaml'
    ):
        """
        Config 템플릿 생성

        Args:
            config_type: 'base', 'models', 'strategies' 중 하나
            name: Config 이름
            format: 파일 형식
        """
        if config_type not in self.hierarchy:
            raise ValueError(f"Invalid config type: {config_type}")

        config_dir = self.hierarchy[config_type]
        config_dir.mkdir(parents=True, exist_ok=True)

        # 템플릿 생성
        if config_type == 'base':
            template = self._get_default_base_config()
        elif config_type == 'models':
            template = {
                'model': {
                    'name': name,
                    'pretrained_model_name': f'upstage/{name}',
                    'max_length': 1024
                }
            }
        elif config_type == 'strategies':
            template = {
                'training': {
                    'strategy': name,
                    'description': f'{name} training strategy'
                }
            }

        # 저장
        ext = 'yaml' if format == 'yaml' else 'json'
        output_path = config_dir / f'{name}.{ext}'

        self.save_config(template, str(output_path), format=format)
        self._log(f"Template 생성: {output_path}")

    def clear_cache(self):
        """Config 캐시 초기화"""
        self.cache.clear()
        self._log("Config 캐시 초기화 완료")


# ==================== 팩토리 함수 ==================== #
def create_hierarchical_loader(
    config_root: str = "configs",
    logger=None
) -> HierarchicalConfigLoader:
    """
    계층적 Config 로더 생성 팩토리 함수

    Args:
        config_root: Config 루트 디렉토리
        logger: Logger 인스턴스

    Returns:
        HierarchicalConfigLoader 인스턴스
    """
    return HierarchicalConfigLoader(config_root=config_root, logger=logger)


# ==================== 사용 예시 ==================== #
if __name__ == "__main__":
    # Config 로더 생성
    loader = create_hierarchical_loader(config_root="configs")

    # 사용 가능한 설정 확인
    available = loader.list_available_configs()
    print("\n사용 가능한 Config:")
    for config_type, configs in available.items():
        print(f"  - {config_type}: {configs}")

    # 계층적 로드 예시
    print("\n" + "=" * 50)
    print("예시 1: Base만")
    print("=" * 50)
    config1 = loader.load_config()
    print(f"Model name: {config1.get('model', {}).get('name')}")
    print(f"Batch size: {config1.get('training', {}).get('batch_size')}")

    print("\n" + "=" * 50)
    print("예시 2: Base + Model")
    print("=" * 50)
    config2 = loader.load_config(model_name="solar")
    print(f"Model name: {config2.get('model', {}).get('name')}")

    print("\n" + "=" * 50)
    print("예시 3: Base + Model + Strategy + Override")
    print("=" * 50)
    config3 = loader.load_config(
        model_name="solar",
        strategy_name="baseline",
        override={
            'training': {
                'batch_size': 32,
                'learning_rate': 3e-5
            }
        }
    )
    print(f"Batch size (override): {config3.get('training', {}).get('batch_size')}")
    print(f"Learning rate (override): {config3.get('training', {}).get('learning_rate')}")

    # 템플릿 생성 예시
    print("\n" + "=" * 50)
    print("템플릿 생성")
    print("=" * 50)
    # loader.create_config_template('models', 'my_model', format='yaml')
    # loader.create_config_template('strategies', 'my_strategy', format='yaml')
