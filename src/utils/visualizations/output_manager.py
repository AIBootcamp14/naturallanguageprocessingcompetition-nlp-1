# src/utils/output_manager.py
"""
출력 관리 모듈
experiments 폴더 구조를 체계적으로 관리하고 시각화를 통합
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import yaml
import json

class ExperimentOutputManager:
    """실험 결과 출력 관리자"""
    
    def __init__(self, base_experiments_dir: str = "experiments"):
        self.base_dir = Path(base_experiments_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 현재 날짜
        self.date_str = datetime.now().strftime('%Y%m%d')
        
    def create_training_output_dir(self, model_name: str) -> Path:
        """학습 결과 출력 디렉토리 생성"""
        output_dir = self.base_dir / "train" / self.date_str / model_name
        self._create_standard_structure(output_dir)
        
        # lastest-train 심볼릭 링크 생성
        self._create_lastest_link(output_dir, self.base_dir / "train" / "lastest-train")
        
        return output_dir
    
    def create_inference_output_dir(self, model_name: str) -> Path:
        """추론 결과 출력 디렉토리 생성"""
        output_dir = self.base_dir / "infer" / self.date_str / model_name
        self._create_standard_structure(output_dir)
        
        # lastest-infer 심볼릭 링크 생성
        self._create_lastest_link(output_dir, self.base_dir / "infer" / "lastest-infer")
        
        return output_dir
    
    def create_optimization_output_dir(self, model_name: str) -> Path:
        """최적화 결과 출력 디렉토리 생성"""
        output_dir = self.base_dir / "optimization" / self.date_str / model_name
        self._create_standard_structure(output_dir)
        
        # lastest-optimization 심볼릭 링크 생성
        self._create_lastest_link(output_dir, self.base_dir / "optimization" / "lastest-optimization")
        
        return output_dir
    
    def _create_standard_structure(self, output_dir: Path):
        """표준 폴더 구조 생성"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 표준 하위 폴더들
        folders = ['images', 'logs', 'configs', 'results']
        for folder in folders:
            (output_dir / folder).mkdir(exist_ok=True)
    
    def _create_lastest_link(self, target_dir: Path, link_path: Path):
        """lastest 심볼릭 링크 생성"""
        try:
            # 기존 링크가 있으면 제거
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            
            # 상대 경로로 심볼릭 링크 생성
            relative_target = os.path.relpath(target_dir, link_path.parent)
            link_path.symlink_to(relative_target)
            print(f"🔗 Created lastest link: {link_path} -> {target_dir}")
            
        except Exception as e:
            print(f"⚠️ Could not create lastest link {link_path}: {e}")
            # 심볼릭 링크 생성 실패해도 계속 진행
    
    def move_optimization_files(self, source_pattern: str, model_name: str):
        """기존 최적화 파일들을 새로운 구조로 이동"""
        optimization_dir = self.create_optimization_output_dir(model_name)
        
        # 기존 파일들 찾기 및 이동
        import glob
        files = glob.glob(source_pattern)
        
        for file_path in files:
            filename = os.path.basename(file_path)
            if 'best_params' in filename:
                # best_params.yaml -> results/ 폴더로
                dest_path = optimization_dir / "results" / filename
            elif 'study' in filename and filename.endswith('.pkl'):
                # study.pkl -> results/ 폴더로
                dest_path = optimization_dir / "results" / filename
            else:
                # 기타 파일들 -> results/ 폴더로
                dest_path = optimization_dir / "results" / filename
            
            try:
                shutil.move(file_path, dest_path)
                print(f"📁 Moved: {file_path} -> {dest_path}")
            except Exception as e:
                print(f"❌ Error moving {file_path}: {e}")
        
        return optimization_dir
    
    def save_experiment_metadata(self, output_dir: Path, metadata: Dict[str, Any]):
        """실험 메타데이터 저장"""
        metadata_file = output_dir / "experiment_metadata.json"
        
        # 기본 메타데이터 추가
        metadata.update({
            'timestamp': datetime.now().isoformat(),
            'date': self.date_str,
            'output_directory': str(output_dir)
        })
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved metadata: {metadata_file}")

class VisualizationIntegrator:
    """시각화 통합 관리자"""
    
    def __init__(self, output_manager: ExperimentOutputManager):
        self.output_manager = output_manager
    
    def integrate_training_visualization(self, model_name: str, fold_results: Dict, 
                                       config: Dict, history_data: Optional[Dict] = None):
        """학습 시각화 통합"""
        from src.utils.visualizations import visualize_training_pipeline
        
        # 출력 디렉토리 생성
        output_dir = self.output_manager.create_training_output_dir(model_name)
        
        # 시각화 실행
        try:
            visualize_training_pipeline(fold_results, model_name, str(output_dir), history_data)
            
            # 메타데이터 저장
            metadata = {
                'pipeline_type': 'training',
                'model_name': model_name,
                'fold_results': fold_results,
                'config_summary': {
                    'epochs': config.get('train', {}).get('epochs', 'unknown'),
                    'batch_size': config.get('train', {}).get('batch_size', 'unknown'),
                    'lr': config.get('train', {}).get('lr', 'unknown')
                }
            }
            self.output_manager.save_experiment_metadata(output_dir, metadata)
            
            print(f"✅ Training visualization completed: {output_dir / 'images'}")
            
        except Exception as e:
            print(f"❌ Training visualization error: {e}")
        
        return output_dir
    
    def integrate_inference_visualization(self, model_name: str, predictions, config: Dict,
                                        confidence_scores=None, ensemble_weights=None, tta_results=None):
        """추론 시각화 통합"""
        from src.utils.visualizations import visualize_inference_pipeline
        
        # 출력 디렉토리 생성
        output_dir = self.output_manager.create_inference_output_dir(model_name)
        
        # 시각화 실행
        try:
            visualize_inference_pipeline(predictions, model_name, str(output_dir),
                                       confidence_scores, ensemble_weights, tta_results)
            
            # 메타데이터 저장
            metadata = {
                'pipeline_type': 'inference',
                'model_name': model_name,
                'prediction_stats': {
                    'total_samples': len(predictions),
                    'unique_classes': len(set(predictions)),
                    'prediction_distribution': {str(k): int(v) for k, v in zip(*np.unique(predictions, return_counts=True))}
                }
            }
            self.output_manager.save_experiment_metadata(output_dir, metadata)
            
            print(f"✅ Inference visualization completed: {output_dir / 'images'}")
            
        except Exception as e:
            print(f"❌ Inference visualization error: {e}")
        
        return output_dir
    
    def integrate_optimization_visualization(self, model_name: str, study_path: str, config: Dict):
        """최적화 시각화 통합"""
        from src.utils.visualizations import visualize_optimization_pipeline
        
        # 출력 디렉토리 생성
        output_dir = self.output_manager.create_optimization_output_dir(model_name)
        
        # 기존 최적화 파일들 이동
        optimization_base = str(self.output_manager.base_dir / "optimization")
        self.output_manager.move_optimization_files(f"{optimization_base}/best_params_*.yaml", model_name)
        
        # 시각화 실행
        try:
            visualize_optimization_pipeline(study_path, model_name, str(output_dir))
            
            # 메타데이터 저장
            metadata = {
                'pipeline_type': 'optimization',
                'model_name': model_name,
                'study_path': study_path,
                'optimization_config': config.get('optuna', {})
            }
            self.output_manager.save_experiment_metadata(output_dir, metadata)
            
            print(f"✅ Optimization visualization completed: {output_dir / 'images'}")
            
        except Exception as e:
            print(f"❌ Optimization visualization error: {e}")
        
        return output_dir

# 전역 매니저 인스턴스
_output_manager = None
_visualization_integrator = None

def get_output_manager() -> ExperimentOutputManager:
    """출력 매니저 싱글톤 획득"""
    global _output_manager
    if _output_manager is None:
        _output_manager = ExperimentOutputManager()
    return _output_manager

def get_visualization_integrator() -> VisualizationIntegrator:
    """시각화 통합자 싱글톤 획득"""
    global _visualization_integrator
    if _visualization_integrator is None:
        _visualization_integrator = VisualizationIntegrator(get_output_manager())
    return _visualization_integrator

# 편의 함수들
def create_training_output(model_name: str) -> Path:
    """학습 출력 디렉토리 생성"""
    return get_output_manager().create_training_output_dir(model_name)

def create_inference_output(model_name: str) -> Path:
    """추론 출력 디렉토리 생성"""
    return get_output_manager().create_inference_output_dir(model_name)

def create_optimization_output(model_name: str) -> Path:
    """최적화 출력 디렉토리 생성"""
    return get_output_manager().create_optimization_output_dir(model_name)

def visualize_training_results(model_name: str, fold_results: Dict, config: Dict, history_data=None):
    """학습 결과 시각화 (통합 함수)"""
    return get_visualization_integrator().integrate_training_visualization(
        model_name, fold_results, config, history_data)

def visualize_inference_results(model_name: str, predictions, config: Dict, **kwargs):
    """추론 결과 시각화 (통합 함수)"""
    return get_visualization_integrator().integrate_inference_visualization(
        model_name, predictions, config, **kwargs)

def visualize_optimization_results(model_name: str, study_path: str, config: Dict):
    """최적화 결과 시각화 (통합 함수)"""
    return get_visualization_integrator().integrate_optimization_visualization(
        model_name, study_path, config)
