# src/utils/output_manager.py
"""
출력 관리 모듈
experiments 폴더 구조를 체계적으로 관리하고 시각화를 통합
"""

# ------------------------- 표준 라이브러리 ------------------------- #
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

# ------------------------- 서드파티 라이브러리 ------------------------- #
import yaml
import json


# ==================== 클래스 정의 ==================== #
# ---------------------- 실험 결과 출력 관리자 클래스 ---------------------- #
class ExperimentOutputManager:
    """실험 결과 출력 관리자"""

    # ---------------------- 초기화 함수 ---------------------- #
    def __init__(self, base_experiments_dir: str = "experiments"):
        """
        Args:
            base_experiments_dir: 실험 결과 기본 디렉토리 경로
        """
        # 기본 디렉토리 설정 및 생성
        self.base_dir = Path(base_experiments_dir)
        self.base_dir.mkdir(exist_ok=True)

        # 현재 날짜 문자열 생성
        self.date_str = datetime.now().strftime('%Y%m%d')  # YYYYMMDD 형식

    # ---------------------- 학습 결과 디렉토리 생성 ---------------------- #
    def create_training_output_dir(self, model_name: str) -> Path:
        """학습 결과 출력 디렉토리 생성

        Args:
            model_name: 모델 이름

        Returns:
            Path: 생성된 출력 디렉토리 경로
        """
        # -------------- 출력 디렉토리 경로 생성 -------------- #
        output_dir = self.base_dir / "train" / self.date_str / model_name  # experiments/train/YYYYMMDD/model_name
        self._create_standard_structure(output_dir)  # 표준 폴더 구조 생성

        # -------------- latest 심볼릭 링크 생성 -------------- #
        # latest-train 심볼릭 링크 생성
        self._create_lastest_link(output_dir, self.base_dir / "train" / "lastest-train")

        return output_dir

    # ---------------------- 추론 결과 디렉토리 생성 ---------------------- #
    def create_inference_output_dir(self, model_name: str) -> Path:
        """추론 결과 출력 디렉토리 생성

        Args:
            model_name: 모델 이름

        Returns:
            Path: 생성된 출력 디렉토리 경로
        """
        # -------------- 출력 디렉토리 경로 생성 -------------- #
        output_dir = self.base_dir / "infer" / self.date_str / model_name  # experiments/infer/YYYYMMDD/model_name
        self._create_standard_structure(output_dir)  # 표준 폴더 구조 생성

        # -------------- latest 심볼릭 링크 생성 -------------- #
        # latest-infer 심볼릭 링크 생성
        self._create_lastest_link(output_dir, self.base_dir / "infer" / "lastest-infer")

        return output_dir

    # ---------------------- 최적화 결과 디렉토리 생성 ---------------------- #
    def create_optimization_output_dir(self, model_name: str) -> Path:
        """최적화 결과 출력 디렉토리 생성

        Args:
            model_name: 모델 이름

        Returns:
            Path: 생성된 출력 디렉토리 경로
        """
        # -------------- 출력 디렉토리 경로 생성 -------------- #
        output_dir = self.base_dir / "optimization" / self.date_str / model_name  # experiments/optimization/YYYYMMDD/model_name
        self._create_standard_structure(output_dir)  # 표준 폴더 구조 생성

        # -------------- latest 심볼릭 링크 생성 -------------- #
        # latest-optimization 심볼릭 링크 생성
        self._create_lastest_link(output_dir, self.base_dir / "optimization" / "lastest-optimization")

        return output_dir

    # ---------------------- 표준 폴더 구조 생성 ---------------------- #
    def _create_standard_structure(self, output_dir: Path):
        """표준 폴더 구조 생성

        Args:
            output_dir: 출력 디렉토리 경로
        """
        # 상위 디렉토리 생성
        output_dir.mkdir(parents=True, exist_ok=True)

        # -------------- 하위 폴더 생성 -------------- #
        # 표준 하위 폴더 목록
        folders = ['images', 'logs', 'configs', 'results']

        # 각 폴더 생성
        for folder in folders:
            (output_dir / folder).mkdir(exist_ok=True)

    # ---------------------- latest 심볼릭 링크 생성 ---------------------- #
    def _create_lastest_link(self, target_dir: Path, link_path: Path):
        """latest 심볼릭 링크 생성

        Args:
            target_dir: 링크 대상 디렉토리
            link_path: 생성할 심볼릭 링크 경로
        """
        # -------------- 예외 처리 블록 -------------- #
        # 심볼릭 링크 생성 시도
        try:
            # 기존 링크 제거
            if link_path.exists() or link_path.is_symlink():  # 링크가 존재하거나 심볼릭 링크인 경우
                link_path.unlink()  # 기존 링크 삭제

            # 상대 경로로 심볼릭 링크 생성
            relative_target = os.path.relpath(target_dir, link_path.parent)  # 상대 경로 계산
            link_path.symlink_to(relative_target)  # 심볼릭 링크 생성
            print(f"🔗 Created lastest link: {link_path} -> {target_dir}")

        # 예외 발생 시 처리
        except Exception as e:
            print(f"⚠️ Could not create lastest link {link_path}: {e}")
            # 심볼릭 링크 생성 실패해도 계속 진행

    # ---------------------- 최적화 파일 이동 ---------------------- #
    def move_optimization_files(self, source_pattern: str, model_name: str):
        """기존 최적화 파일들을 새로운 구조로 이동

        Args:
            source_pattern: 이동할 파일 패턴 (glob 패턴)
            model_name: 모델 이름

        Returns:
            Path: 최적화 출력 디렉토리 경로
        """
        # 최적화 디렉토리 생성
        optimization_dir = self.create_optimization_output_dir(model_name)

        # -------------- 파일 검색 및 이동 -------------- #
        import glob

        # 패턴에 맞는 파일 검색
        files = glob.glob(source_pattern)

        # 각 파일 이동
        for file_path in files:
            # 파일명 추출
            filename = os.path.basename(file_path)

            # -------------- 파일 타입에 따른 경로 설정 -------------- #
            # best_params 파일 처리
            if 'best_params' in filename:
                dest_path = optimization_dir / "results" / filename  # results 폴더로 이동

            # study.pkl 파일 처리
            elif 'study' in filename and filename.endswith('.pkl'):
                dest_path = optimization_dir / "results" / filename  # results 폴더로 이동

            # 기타 파일 처리
            else:
                dest_path = optimization_dir / "results" / filename  # results 폴더로 이동

            # -------------- 파일 이동 시도 -------------- #
            try:
                shutil.move(file_path, dest_path)  # 파일 이동
                print(f"📁 Moved: {file_path} -> {dest_path}")

            # 이동 실패 시 에러 출력
            except Exception as e:
                print(f"❌ Error moving {file_path}: {e}")

        return optimization_dir

    # ---------------------- 실험 메타데이터 저장 ---------------------- #
    def save_experiment_metadata(self, output_dir: Path, metadata: Dict[str, Any]):
        """실험 메타데이터 저장

        Args:
            output_dir: 출력 디렉토리 경로
            metadata: 저장할 메타데이터 딕셔너리
        """
        # 메타데이터 파일 경로
        metadata_file = output_dir / "experiment_metadata.json"

        # -------------- 메타데이터 업데이트 -------------- #
        # 기본 메타데이터 추가
        metadata.update({
            'timestamp': datetime.now().isoformat(),  # ISO 형식 타임스탬프
            'date': self.date_str,                    # 날짜 문자열
            'output_directory': str(output_dir)       # 출력 디렉토리 경로
        })

        # -------------- 파일 저장 -------------- #
        # JSON 파일로 저장
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)  # 들여쓰기 2칸, 유니코드 허용

        print(f"💾 Saved metadata: {metadata_file}")


# ==================== 시각화 통합 관리자 클래스 ==================== #
# ---------------------- 시각화 통합 관리자 클래스 ---------------------- #
class VisualizationIntegrator:
    """시각화 통합 관리자"""

    # ---------------------- 초기화 함수 ---------------------- #
    def __init__(self, output_manager: ExperimentOutputManager):
        """
        Args:
            output_manager: 출력 관리자 인스턴스
        """
        self.output_manager = output_manager  # 출력 관리자 저장

    # ---------------------- 학습 시각화 통합 함수 ---------------------- #
    def integrate_training_visualization(self, model_name: str, fold_results: Dict,
                                       config: Dict, history_data: Optional[Dict] = None):
        """학습 시각화 통합

        Args:
            model_name: 모델 이름
            fold_results: 폴드별 학습 결과
            config: 설정 딕셔너리
            history_data: 학습 히스토리 데이터 (선택)

        Returns:
            Path: 출력 디렉토리 경로
        """
        from src.utils.visualizations import visualize_training_pipeline

        # 출력 디렉토리 생성
        output_dir = self.output_manager.create_training_output_dir(model_name)

        # -------------- 시각화 실행 -------------- #
        # 시각화 실행 시도
        try:
            # 학습 파이프라인 시각화 호출
            visualize_training_pipeline(fold_results, model_name, str(output_dir), history_data)

            # -------------- 메타데이터 생성 및 저장 -------------- #
            # 메타데이터 딕셔너리 생성
            metadata = {
                'pipeline_type': 'training',          # 파이프라인 타입
                'model_name': model_name,             # 모델 이름
                'fold_results': fold_results,         # 폴드 결과
                'config_summary': {                   # 설정 요약
                    'epochs': config.get('train', {}).get('epochs', 'unknown'),          # 에포크 수
                    'batch_size': config.get('train', {}).get('batch_size', 'unknown'),  # 배치 크기
                    'lr': config.get('train', {}).get('lr', 'unknown')                   # 학습률
                }
            }

            # 메타데이터 저장
            self.output_manager.save_experiment_metadata(output_dir, metadata)

            print(f"✅ Training visualization completed: {output_dir / 'images'}")

        # 예외 발생 시 에러 출력
        except Exception as e:
            print(f"❌ Training visualization error: {e}")

        return output_dir

    # ---------------------- 추론 시각화 통합 함수 ---------------------- #
    def integrate_inference_visualization(self, model_name: str, predictions, config: Dict,
                                        confidence_scores=None, ensemble_weights=None, tta_results=None):
        """추론 시각화 통합

        Args:
            model_name: 모델 이름
            predictions: 예측 결과
            config: 설정 딕셔너리
            confidence_scores: 신뢰도 점수 (선택)
            ensemble_weights: 앙상블 가중치 (선택)
            tta_results: TTA 결과 (선택)

        Returns:
            Path: 출력 디렉토리 경로
        """
        from src.utils.visualizations import visualize_inference_pipeline

        # 출력 디렉토리 생성
        output_dir = self.output_manager.create_inference_output_dir(model_name)

        # -------------- 시각화 실행 -------------- #
        # 시각화 실행 시도
        try:
            # 추론 파이프라인 시각화 호출
            visualize_inference_pipeline(predictions, model_name, str(output_dir),
                                       confidence_scores, ensemble_weights, tta_results)

            # -------------- 메타데이터 생성 및 저장 -------------- #
            # 메타데이터 딕셔너리 생성
            metadata = {
                'pipeline_type': 'inference',                  # 파이프라인 타입
                'model_name': model_name,                      # 모델 이름
                'prediction_stats': {                          # 예측 통계
                    'total_samples': len(predictions),         # 총 샘플 수
                    'unique_classes': len(set(predictions)),   # 고유 클래스 수
                    'prediction_distribution': {str(k): int(v) for k, v in zip(*np.unique(predictions, return_counts=True))}  # 예측 분포
                }
            }

            # 메타데이터 저장
            self.output_manager.save_experiment_metadata(output_dir, metadata)

            print(f"✅ Inference visualization completed: {output_dir / 'images'}")

        # 예외 발생 시 에러 출력
        except Exception as e:
            print(f"❌ Inference visualization error: {e}")

        return output_dir

    # ---------------------- 최적화 시각화 통합 함수 ---------------------- #
    def integrate_optimization_visualization(self, model_name: str, study_path: str, config: Dict):
        """최적화 시각화 통합

        Args:
            model_name: 모델 이름
            study_path: Optuna Study 파일 경로
            config: 설정 딕셔너리

        Returns:
            Path: 출력 디렉토리 경로
        """
        from src.utils.visualizations import visualize_optimization_pipeline

        # 출력 디렉토리 생성
        output_dir = self.output_manager.create_optimization_output_dir(model_name)

        # -------------- 기존 최적화 파일 이동 -------------- #
        # 최적화 파일 이동
        optimization_base = str(self.output_manager.base_dir / "optimization")
        self.output_manager.move_optimization_files(f"{optimization_base}/best_params_*.yaml", model_name)

        # -------------- 시각화 실행 -------------- #
        # 시각화 실행 시도
        try:
            # 최적화 파이프라인 시각화 호출
            visualize_optimization_pipeline(study_path, model_name, str(output_dir))

            # -------------- 메타데이터 생성 및 저장 -------------- #
            # 메타데이터 딕셔너리 생성
            metadata = {
                'pipeline_type': 'optimization',           # 파이프라인 타입
                'model_name': model_name,                  # 모델 이름
                'study_path': study_path,                  # Study 파일 경로
                'optimization_config': config.get('optuna', {})  # Optuna 설정
            }

            # 메타데이터 저장
            self.output_manager.save_experiment_metadata(output_dir, metadata)

            print(f"✅ Optimization visualization completed: {output_dir / 'images'}")

        # 예외 발생 시 에러 출력
        except Exception as e:
            print(f"❌ Optimization visualization error: {e}")

        return output_dir


# ==================== 싱글톤 인스턴스 및 팩토리 함수 ==================== #
# 전역 싱글톤 인스턴스
_output_manager = None
_visualization_integrator = None

# ---------------------- 출력 관리자 싱글톤 획득 ---------------------- #
def get_output_manager() -> ExperimentOutputManager:
    """출력 관리자 싱글톤 획득

    Returns:
        ExperimentOutputManager: 전역 출력 관리자 인스턴스
    """
    global _output_manager

    # -------------- 인스턴스 생성 (최초 호출 시) -------------- #
    # 인스턴스가 없으면 생성
    if _output_manager is None:
        _output_manager = ExperimentOutputManager()

    return _output_manager

# ---------------------- 시각화 통합자 싱글톤 획득 ---------------------- #
def get_visualization_integrator() -> VisualizationIntegrator:
    """시각화 통합자 싱글톤 획득

    Returns:
        VisualizationIntegrator: 전역 시각화 통합자 인스턴스
    """
    global _visualization_integrator

    # -------------- 인스턴스 생성 (최초 호출 시) -------------- #
    # 인스턴스가 없으면 생성
    if _visualization_integrator is None:
        _visualization_integrator = VisualizationIntegrator(get_output_manager())

    return _visualization_integrator


# ==================== 편의 함수들 ==================== #
# ---------------------- 학습 출력 디렉토리 생성 ---------------------- #
def create_training_output(model_name: str) -> Path:
    """학습 출력 디렉토리 생성

    Args:
        model_name: 모델 이름

    Returns:
        Path: 생성된 디렉토리 경로
    """
    return get_output_manager().create_training_output_dir(model_name)

# ---------------------- 추론 출력 디렉토리 생성 ---------------------- #
def create_inference_output(model_name: str) -> Path:
    """추론 출력 디렉토리 생성

    Args:
        model_name: 모델 이름

    Returns:
        Path: 생성된 디렉토리 경로
    """
    return get_output_manager().create_inference_output_dir(model_name)

# ---------------------- 최적화 출력 디렉토리 생성 ---------------------- #
def create_optimization_output(model_name: str) -> Path:
    """최적화 출력 디렉토리 생성

    Args:
        model_name: 모델 이름

    Returns:
        Path: 생성된 디렉토리 경로
    """
    return get_output_manager().create_optimization_output_dir(model_name)

# ---------------------- 학습 결과 시각화 통합 함수 ---------------------- #
def visualize_training_results(model_name: str, fold_results: Dict, config: Dict, history_data=None):
    """학습 결과 시각화 통합 함수

    Args:
        model_name: 모델 이름
        fold_results: 폴드별 결과
        config: 설정 딕셔너리
        history_data: 히스토리 데이터 (선택)

    Returns:
        Path: 출력 디렉토리 경로
    """
    return get_visualization_integrator().integrate_training_visualization(
        model_name, fold_results, config, history_data)

# ---------------------- 추론 결과 시각화 통합 함수 ---------------------- #
def visualize_inference_results(model_name: str, predictions, config: Dict, **kwargs):
    """추론 결과 시각화 통합 함수

    Args:
        model_name: 모델 이름
        predictions: 예측 결과
        config: 설정 딕셔너리
        **kwargs: 추가 파라미터

    Returns:
        Path: 출력 디렉토리 경로
    """
    return get_visualization_integrator().integrate_inference_visualization(
        model_name, predictions, config, **kwargs)

# ---------------------- 최적화 결과 시각화 통합 함수 ---------------------- #
def visualize_optimization_results(model_name: str, study_path: str, config: Dict):
    """최적화 결과 시각화 통합 함수

    Args:
        model_name: 모델 이름
        study_path: Study 파일 경로
        config: 설정 딕셔너리

    Returns:
        Path: 출력 디렉토리 경로
    """
    return get_visualization_integrator().integrate_optimization_visualization(
        model_name, study_path, config)
