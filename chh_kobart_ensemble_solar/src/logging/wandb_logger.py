# ---------------------- WandB 로깅 모듈 ---------------------- #
"""
WandB 로깅 유틸리티
팀 프로젝트용 WandB 통합 로깅 시스템
"""

import os                                                   # 운영체제 관련 기능
import wandb                                                # Weights & Biases 로깅 라이브러리
from datetime import datetime                               # 현재 시간 처리
from typing import Dict, Any, Optional                      # 타입 힌트
import torch                                                # PyTorch 텐서 처리


# ---------------------- WandB Logger 클래스 ---------------------- #
# WandbLogger 클래스 정의
class WandbLogger:
    # 초기화 함수 정의
    def __init__(
        self,
        project_name: str = "document-classification-team", # 프로젝트 이름
        entity: Optional[str] = None,                       # WandB 엔티티 (팀/사용자)
        experiment_name: str = "experiment",                # 실험 이름
        config: Optional[Dict[str, Any]] = None,            # 설정 딕셔너리
        tags: Optional[list] = None,                        # 태그 리스트
    ):
        self.project_name = project_name                    # 프로젝트 이름 저장
        self.entity = entity                                # 엔티티 저장
        self.experiment_name = experiment_name              # 실험 이름 저장
        self.config = config or {}                          # 설정 저장 (기본값: 빈 딕셔너리)
        self.tags = tags or []                              # 태그 저장 (기본값: 빈 리스트)
        
        # 타임스탬프 추가
        timestamp = datetime.now().strftime("%m%d-%H%M")    # 현재 시간 타임스탬프 생성
        self.run_name = f"{timestamp}-{experiment_name}"    # 실행 이름 생성
        
        self.run = None                                     # WandB 실행 객체 초기화
        self.is_initialized = False                         # 초기화 상태 플래그
    
    
    # WandB 로그인 함수 정의
    def login(self):
        try:
            # API 키가 없는 경우
            if wandb.api.api_key is None:
                print("WandB에 로그인이 필요합니다.")          # 로그인 필요 메시지
                wandb.login()                               # WandB 로그인 수행
            # API 키가 있는 경우
            else:
                print(f"WandB 로그인 상태: {wandb.api.viewer()['username']}")   # 로그인 상태 출력
                
        # 예외 발생 시
        except:
            print("WandB 로그인을 진행합니다...")              # 로그인 진행 메시지
            wandb.login()                                   # WandB 로그인 수행
    
    
    # 실행 초기화 함수 정의
    def init_run(self, fold: Optional[int] = None):
        # 이미 초기화된 경우
        if self.is_initialized:
            return      # 함수 종료

        self.login()    # WandB 로그인

        # fold가 지정된 경우 run name에 추가
        run_name = self.run_name                 # 기본 실행 이름

        # 폴드가 지정된 경우
        if fold is not None:
            run_name = f"fold-{fold}-{run_name}" # 폴드 번호 추가

        # WandB 디렉토리를 프로젝트 루트로 설정
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        wandb_dir = os.path.join(project_root, "wandb")
        os.environ["WANDB_DIR"] = wandb_dir

        # WandB run 초기화
        self.run = wandb.init(
            project=self.project_name,           # 프로젝트 이름
            entity=self.entity,                  # 엔티티
            name=run_name,                       # 실행 이름
            config=self.config,                  # 설정
            tags=self.tags,                      # 태그
            dir=wandb_dir,                       # WandB 디렉토리 지정
            reinit=True                          # 재초기화 허용
        )
        
        self.is_initialized = True               # 초기화 상태 업데이트
        print(f"📋 실험명: {run_name}")           # 실험명 출력
        print(f"🔗 WandB URL: {self.run.url}")   # WandB URL 출력
    
    
    # 메트릭 로깅 함수 정의
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        # 초기화되지 않은 경우
        if not self.is_initialized:
            return  # 함수 종료

        wandb.log(metrics, step=step)            # WandB에 메트릭 로깅


    # 텍스트 로깅 함수 정의
    def log_text(self, message: str, level: str = "info"):
        """
        텍스트 메시지 로깅 (WandB에는 로깅하지 않음, 호환성 유지용)

        Args:
            message: 로그 메시지
            level: 로그 레벨 (info, warning, error 등)
        """
        # WandB에는 일반 텍스트를 직접 로깅하지 않음
        # 이 메서드는 base_trainer.py와의 호환성을 위해 존재
        pass
    
    
    # 모델 로깅 함수 정의
    def log_model(self, model_path: str, name: str = "model"):
        # 초기화되지 않은 경우
        if not self.is_initialized:
            return  # 함수 종료

        artifact = wandb.Artifact(name, type="model")   # 모델 아티팩트 생성
        artifact.add_file(model_path)                   # 모델 파일 추가
        wandb.log_artifact(artifact)                    # 아티팩트 로깅
    
    
    # 혼동 행렬 로깅 함수 정의
    def log_confusion_matrix(self, y_true, y_pred, class_names=None):
        # 초기화되지 않은 경우
        if not self.is_initialized:
            return  # 함수 종료
        
        # WandB에 혼동 행렬 로깅
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(# 혼동 행렬 플롯 생성
                y_true=y_true,                              # 실제 값
                preds=y_pred,                               # 예측 값
                class_names=class_names                     # 클래스 이름
            )
        })

    
    # 예측 결과 로깅 함수 정의
    def log_predictions(self, images, predictions, targets, class_names=None):
        # 초기화되지 않은 경우
        if not self.is_initialized:
            return  # 함수 종료

        # 최대 100개 샘플만 로깅
        max_samples = min(100, len(images))      # 최대 샘플 수 제한

        data = []                                # 데이터 리스트 초기화

        # 샘플 수만큼 반복
        for i in range(max_samples):
            img = images[i]                      # i번째 이미지
            pred = predictions[i]                # i번째 예측
            target = targets[i]                  # i번째 정답

            # 이미지를 wandb Image로 변환
            if torch.is_tensor(img):             # 텐서인 경우
                # numpy로 변환 후 차원 순서 변경
                img = img.cpu().numpy().transpose(1, 2, 0)

            # 예측 클래스명
            pred_class = class_names[pred] if class_names else str(pred)
            # 정답 클래스명
            target_class = class_names[target] if class_names else str(target)

            # 데이터 추가
            data.append([
                wandb.Image(img),                # WandB 이미지 객체
                pred_class,                      # 예측 클래스
                target_class,                    # 정답 클래스
                pred == target                   # 정답 여부
            ])

        # WandB 테이블 생성
        table = wandb.Table(
            data=data,                           # 테이블 데이터
            columns=["Image", "Prediction", "Target", "Correct"]  # 컬럼명
        )

        wandb.log({"predictions": table})        # 예측 테이블 로깅


    # ========== PRD 11: 추가 시각화 5가지 ========== #

    # 1. 학습률 스케줄 로깅
    def log_learning_rate_schedule(self, step: int, learning_rate: float):
        """
        학습률 스케줄 로깅

        Args:
            step: 현재 스텝
            learning_rate: 현재 학습률
        """
        if not self.is_initialized:
            return

        wandb.log({
            "learning_rate": learning_rate,
            "step": step
        }, step=step)


    # 2. 그래디언트 norm 로깅
    def log_gradient_norms(self, model, step: int):
        """
        그래디언트 norm 로깅

        Args:
            model: PyTorch 모델
            step: 현재 스텝
        """
        if not self.is_initialized:
            return

        # 전체 그래디언트 norm 계산
        total_norm = 0.0
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2

                # 레이어별 norm 저장 (상위 10개만)
                layer_norms[f"grad_norm/{name}"] = param_norm

        total_norm = total_norm ** 0.5

        # 로깅
        log_data = {
            "gradient/total_norm": total_norm,
            "step": step
        }

        # 상위 10개 레이어만 로깅 (너무 많으면 느려짐)
        sorted_layers = sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)[:10]
        for layer_name, norm_value in sorted_layers:
            log_data[layer_name] = norm_value

        wandb.log(log_data, step=step)


    # 3. Loss curve 로깅
    def log_loss_curve(self, train_loss: float, val_loss: float = None, step: int = None):
        """
        Loss curve 로깅

        Args:
            train_loss: 학습 손실
            val_loss: 검증 손실 (선택)
            step: 스텝 번호 (선택)
        """
        if not self.is_initialized:
            return

        log_data = {
            "loss/train": train_loss
        }

        if val_loss is not None:
            log_data["loss/val"] = val_loss

            # Loss 차이도 로깅 (overfitting 모니터링)
            log_data["loss/train_val_diff"] = train_loss - val_loss

        wandb.log(log_data, step=step)


    # 4. GPU 메모리 사용량 로깅
    def log_gpu_memory(self, step: int = None):
        """
        GPU 메모리 사용량 로깅

        Args:
            step: 스텝 번호 (선택)
        """
        if not self.is_initialized:
            return

        # CUDA 사용 가능 여부 확인
        if not torch.cuda.is_available():
            return

        # GPU 개수만큼 반복
        log_data = {}

        for i in range(torch.cuda.device_count()):
            # 메모리 정보 (바이트 -> GB)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3

            log_data[f"gpu_{i}/memory_allocated_gb"] = allocated
            log_data[f"gpu_{i}/memory_reserved_gb"] = reserved
            log_data[f"gpu_{i}/memory_max_allocated_gb"] = max_allocated

            # 사용률 (%)
            if reserved > 0:
                log_data[f"gpu_{i}/memory_utilization"] = (allocated / reserved) * 100

        wandb.log(log_data, step=step)


    # 5. 학습 속도 로깅
    def log_training_speed(
        self,
        samples_per_second: float = None,
        steps_per_second: float = None,
        epoch_time: float = None,
        step: int = None
    ):
        """
        학습 속도 로깅

        Args:
            samples_per_second: 초당 샘플 수
            steps_per_second: 초당 스텝 수
            epoch_time: 에포크 소요 시간 (초)
            step: 스텝 번호 (선택)
        """
        if not self.is_initialized:
            return

        log_data = {}

        if samples_per_second is not None:
            log_data["speed/samples_per_second"] = samples_per_second

        if steps_per_second is not None:
            log_data["speed/steps_per_second"] = steps_per_second

        if epoch_time is not None:
            log_data["speed/epoch_time_seconds"] = epoch_time
            log_data["speed/epoch_time_minutes"] = epoch_time / 60

        wandb.log(log_data, step=step)
    
    
    # 실행 종료 함수 정의
    def finish(self):
        # 실행 객체가 존재하는 경우
        if self.run is not None:
            wandb.finish()                       # WandB 실행 종료
            self.is_initialized = False          # 초기화 상태 리셋
    
    
    # 컨텍스트 매니저 진입 함수 정의
    def __enter__(self):
        return self                              # 자기 자신 반환
    
    
    # 컨텍스트 매니저 종료 함수 정의
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()                            # 실행 종료


# 편의 함수들
def create_wandb_config(
    model_name: str,                            # 모델 이름
    img_size: int,                              # 이미지 크기
    batch_size: int,                            # 배치 크기
    learning_rate: float,                       # 학습률
    epochs: int,                                # 에포크 수
    **kwargs                                    # 추가 인자
) -> Dict[str, Any]:                            # 설정 딕셔너리 반환
    """WandB config 생성 함수"""
    config = {
        "architecture": model_name,             # 모델 구조
        "image_size": img_size,                 # 이미지 크기
        "batch_size": batch_size,               # 배치 크기
        "learning_rate": learning_rate,         # 학습률
        "epochs": epochs,                       # 에포크 수
        "framework": "PyTorch",                 # 프레임워크
        "dataset": "Document Classification",   # 데이터셋
    }
    
    config.update(kwargs)                       # 추가 설정 업데이트
    return config                               # 설정 딕셔너리 반환


# 폴드 결과 로깅 함수 정의
def log_fold_results(logger: WandbLogger, fold: int, metrics: Dict[str, float]):
    # 폴드별 메트릭 로깅
    logger.log_metrics({
        f"fold_{fold}_train_f1": metrics.get("train_f1", 0),        # 폴드별 학습 F1
        f"fold_{fold}_val_f1": metrics.get("val_f1", 0),            # 폴드별 검증 F1
        f"fold_{fold}_train_loss": metrics.get("train_loss", 0),    # 폴드별 학습 손실
        f"fold_{fold}_val_loss": metrics.get("val_loss", 0),        # 폴드별 검증 손실
    })
