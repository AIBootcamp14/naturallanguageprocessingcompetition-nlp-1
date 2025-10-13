# ==================== KFoldTrainer ==================== #
"""
K-Fold 교차 검증 Trainer

K-Fold 교차 검증을 통한 안정적인 모델 성능 평가 및 학습
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import json
from pathlib import Path

# ---------------------- 서드파티 라이브러리 ---------------------- #
from sklearn.model_selection import KFold

# ---------------------- 프로젝트 모듈 ---------------------- #
from src.trainers.base_trainer import BaseTrainer
from src.config import load_config, load_model_config
from src.models import load_model_and_tokenizer
from src.data import DialogueSummarizationDataset
from src.training import create_trainer


# ==================== KFoldTrainer ==================== #
class KFoldTrainer(BaseTrainer):
    """K-Fold 교차 검증 학습 Trainer"""

    def train(self):
        """
        K-Fold 교차 검증 학습 실행

        Returns:
            dict: 학습 결과
                - mode: 'kfold'
                - k_folds: fold 수
                - fold_results: 각 fold별 결과 리스트
                - avg_metrics: 평균 메트릭
        """
        self.log("=" * 60)
        self.log("🔄 K-FOLD 교차검증 모드 학습 시작")
        self.log(f"📋 K-Folds: {self.args.k_folds}")
        self.log(f"📋 모델: {self.args.models[0]}")
        self.log(f"📋 Fold Seed: {self.args.fold_seed}")
        self.log("=" * 60)

        # 1. 전체 데이터 로드 (K-Fold는 train 데이터만 사용)
        self.log("\n[1/3] 전체 데이터 로딩...")
        train_df, _ = self.load_data()
        self.log(f"  ✅ 전체 데이터: {len(train_df)}개")

        # 2. Config 로드
        self.log("\n[2/3] Config 로딩...")
        # 모델명으로 직접 config 로드 (PRD 19)
        config = load_model_config(self.args.models[0])

        # 명령행 인자로 Config 오버라이드
        self._override_config(config)

        self.log(f"  ✅ Config 로드 완료: {self.args.models[0]}")

        # 3. K-Fold 분할 및 학습
        self.log("\n[3/3] K-Fold 교차검증 실행...")
        kf = KFold(
            n_splits=self.args.k_folds,
            shuffle=True,
            random_state=self.args.fold_seed
        )

        fold_results = []

        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(train_df)):
            self.log(f"\n{'=' * 40}")
            # FIXME: Corrupted log message
            self.log(f"{'=' * 40}")

            # Fold별 데이터 분할
            fold_train_df = train_df.iloc[train_indices]
            fold_val_df = train_df.iloc[val_indices]

            self.log(f"  학습: {len(fold_train_df)}개")
            self.log(f"  검증: {len(fold_val_df)}개")

            # Fold별 학습
            fold_result = self._train_fold(
                fold_idx=fold_idx,
                train_df=fold_train_df,
                val_df=fold_val_df,
                config=config
            )
            fold_results.append(fold_result)

            # Fold 결과 출력
            if 'eval_metrics' in fold_result:
                # FIXME: Corrupted log message
                for key, value in fold_result['eval_metrics'].items():
                    if 'rouge' in key.lower():
                        self.log(f"    {key}: {value:.4f}")

        # 4. 평균 성능 계산
        avg_metrics = self._calculate_average_metrics(fold_results)

        self.log("\n" + "=" * 60)
        self.log("✅ K-FOLD 교차검증 완료!")
        self.log("\n📊 평균 성능:")
        for key, value in avg_metrics.items():
            if 'rouge' in key.lower():
                self.log(f"  {key}: {value:.4f}")
        self.log("=" * 60)

        return {
            'mode': 'kfold',
            'model': self.args.models[0],
            'k_folds': self.args.k_folds,
            'fold_results': fold_results,
            'avg_metrics': avg_metrics
        }

    def _train_fold(self, fold_idx, train_df, val_df, config):
        """
        개별 Fold 학습

        Args:
            fold_idx: Fold 인덱스
            train_df: 학습 데이터프레임
            val_df: 검증 데이터프레임
            config: Config 객체

        Returns:
            dict: Fold 학습 결과
        """
        # 모델 및 토크나이저 로드 (각 Fold마다 새로 로드)
        model, tokenizer = load_model_and_tokenizer(config, logger=self.logger)

        # 모델 타입 가져오기 (기본값: encoder_decoder)
        model_type = config.model.get('type', 'encoder_decoder')

        # Dataset 생성
        train_dataset = DialogueSummarizationDataset(
            dialogues=train_df['dialogue'].tolist(),
            summaries=train_df['summary'].tolist(),
            tokenizer=tokenizer,
            encoder_max_len=config.tokenizer.encoder_max_len,
            decoder_max_len=config.tokenizer.decoder_max_len,
            preprocess=True,
            model_type=model_type  # PRD 08: LLM 지원
        )

        val_dataset = DialogueSummarizationDataset(
            dialogues=val_df['dialogue'].tolist(),
            summaries=val_df['summary'].tolist(),
            tokenizer=tokenizer,
            encoder_max_len=config.tokenizer.encoder_max_len,
            decoder_max_len=config.tokenizer.decoder_max_len,
            preprocess=True,
            model_type=model_type  # PRD 08: LLM 지원
        )

        # Fold별 출력 디렉토리
        fold_output_dir = self.output_dir / f"fold_{fold_idx + 1}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        # Config에 fold별 출력 디렉토리 설정
        config.training.output_dir = str(fold_output_dir)

        # Trainer 생성
        trainer = create_trainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            use_wandb=getattr(self.args, 'use_wandb', True),
            logger=self.logger
        )

        # 학습 실행
        train_results = trainer.train()

        # 결과 수집
        fold_result = {
            'fold': fold_idx + 1,
            'train_results': train_results,
            'model_path': str(fold_output_dir / 'final_model')
        }

        # 평가 메트릭 추가
        if hasattr(trainer, 'state') and trainer.state.log_history:
            eval_metrics = self._extract_eval_metrics(trainer.state.log_history)
            fold_result['eval_metrics'] = eval_metrics

        return fold_result

    def save_results(self, results):
        """
        결과 저장

        Args:
            results: 학습 결과 딕셔너리
        """
        result_path = self.output_dir / "kfold_results.json"

        # 저장 가능한 형태로 변환
        saveable_results = {
            'mode': results['mode'],
            'model': results['model'],
            'k_folds': results['k_folds'],
            'avg_metrics': results.get('avg_metrics', {}),
            'fold_results': [
                {
                    'fold': fr['fold'],
                    'eval_metrics': fr.get('eval_metrics', {}),
                    'model_path': fr['model_path']
                }
                for fr in results['fold_results']
            ]
        }

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(saveable_results, f, indent=2, ensure_ascii=False)

        self.log(f"\n💾 결과 저장: {result_path}")

    def _extract_eval_metrics(self, log_history):
        """
        학습 로그에서 평가 메트릭 추출

        Args:
            log_history: Trainer의 로그 히스토리

        Returns:
            dict: 평가 메트릭
        """
        eval_metrics = {}

        # 마지막 eval 로그 찾기
        for log_entry in reversed(log_history):
            if 'eval_loss' in log_entry:
                for key, value in log_entry.items():
                    if key.startswith('eval_'):
                        eval_metrics[key] = value
                break

        return eval_metrics

    def _calculate_average_metrics(self, fold_results):
        """
        모든 Fold의 평균 메트릭 계산

        Args:
            fold_results: Fold별 결과 리스트

        Returns:
            dict: 평균 메트릭
        """
        avg_metrics = {}

        # 메트릭 키 수집
        all_keys = set()
        for fr in fold_results:
            if 'eval_metrics' in fr:
                all_keys.update(fr['eval_metrics'].keys())

        # 각 메트릭의 평균 계산
        for key in all_keys:
            values = []
            for fr in fold_results:
                if 'eval_metrics' in fr and key in fr['eval_metrics']:
                    values.append(fr['eval_metrics'][key])

            if values:
                avg_metrics[key] = sum(values) / len(values)

        return avg_metrics
