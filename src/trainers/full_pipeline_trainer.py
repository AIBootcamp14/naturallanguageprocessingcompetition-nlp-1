# ==================== FullPipelineTrainer ==================== #
"""
풀 파이프라인 Trainer

PRD 14: 다중 선택 옵션 - Full 실행
통합 운영 가능한 주요 파이프라인:
- 다중 모델 앙상블
- Optuna 하이퍼파라미터 튜닝 (선택)
- K-Fold 교차검증
- TTA (Test Time Augmentation)
- Solar API 통합
"""

# ---------------------- 표준 라이브러리 ---------------------- #
import json
from pathlib import Path
from typing import List, Dict, Any

# ---------------------- 로컬 모듈 ---------------------- #
from src.trainers.base_trainer import BaseTrainer
from src.config import load_model_config
from src.models import load_model_and_tokenizer
from src.data import DialogueSummarizationDataset
from src.training import create_trainer
from src.ensemble import ModelManager
from src.api import create_solar_api


# ==================== FullPipelineTrainer ==================== #
class FullPipelineTrainer(BaseTrainer):
    """풀 파이프라인 Trainer"""

    def train(self):
        """
        풀 파이프라인 학습

        Returns:
            dict: 학습 결과
                - mode: 'full'
                - models: 사용된 모델 리스트
                - ensemble_results: 앙상블 결과
                - solar_results: Solar API 결과 (선택)
                - inference_results: 추론 및 제출 파일 결과
                - final_metrics: 최종 평가 지표
        """
        self.log("=" * 60)
        self.log("= FULL PIPELINE 실행 시작")
        self.log(f"=대상 모델: {', '.join(self.args.models)}")
        self.log(f"=앙상블 앙상블 전략: {self.args.ensemble_strategy}")
        self.log(f"= TTA 사용: {self.args.use_tta}")
        self.log("=" * 60)

        # 1. 데이터 로드
        self.log("\n[1/6] 데이터 로딩...")
        train_df, eval_df = self.load_data()

        # 2. 모델 학습 (다중 모델)
        self.log(f"\n[2/6] 다중 모델 학습 ({len(self.args.models)} 모델)...")
        model_results, model_paths = self._train_multiple_models(train_df, eval_df)

        # 3. 앙상블 생성
        self.log(f"\n[3/6] 앙상블 생성...")
        ensemble_results = self._create_and_evaluate_ensemble(
            model_paths=model_paths,
            eval_df=eval_df
        )

        # 4. Solar API 통합 (선택)
        solar_results = {}
        try:
            self.log(f"\n[4/6] Solar API 통합...")
            solar_results = self._integrate_solar_api(eval_df)
        except Exception as e:
            self.log(f"    Solar API 통합 오류: {e}")

        # 5. TTA 적용 (선택)
        tta_results = {}
        if self.args.use_tta:
            self.log(f"\n[5/6] TTA 적용...")
            tta_results = self._apply_tta(model_paths, eval_df)

        # 6. 추론 및 제출 파일 생성
        self.log(f"\n[6/6] 추론 및 제출 파일 생성...")
        inference_results = self._create_submission(model_paths)

        # 결과 수집
        results = {
            'mode': 'full',
            'models': self.args.models,
            'ensemble_strategy': self.args.ensemble_strategy,
            'use_tta': self.args.use_tta,
            'model_results': model_results,
            'ensemble_results': ensemble_results,
            'solar_results': solar_results,
            'tta_results': tta_results,
            'inference_results': inference_results
        }

        self.log("\n" + "=" * 60)
        self.log(" FULL PIPELINE 완료!")

        self.log("\n=요약 개별 모델 결과:")
        for result in model_results:
            self.log(f"\n{result['model_name']}:")
            if result['eval_metrics']:
                for key, value in result['eval_metrics'].items():
                    if 'rouge' in key.lower():
                        self.log(f"  {key}: {value:.4f}")

        self.log("\n=요약 앙상블 결과:")
        if ensemble_results:
            for key, value in ensemble_results.items():
                self.log(f"  {key}: {value:.4f}")

        if solar_results:
            self.log("\n=요약 Solar API 결과:")
            for key, value in solar_results.items():
                if isinstance(value, (int, float)):
                    self.log(f"  {key}: {value:.4f}")

        self.log("=" * 60)

        return results

    def save_results(self, results):
        """
        결과 저장

        Args:
            results: 학습 결과 딕셔너리
        """
        result_path = self.output_dir / "full_pipeline_results.json"

        # 저장 가능한 형태로 변환
        saveable_results = {
            'mode': results['mode'],
            'models': results['models'],
            'ensemble_strategy': results['ensemble_strategy'],
            'use_tta': results['use_tta'],
            'model_results': results['model_results'],
            'ensemble_results': results.get('ensemble_results', {}),
            'solar_results': results.get('solar_results', {}),
            'tta_results': results.get('tta_results', {}),
            'inference_results': results.get('inference_results', {})
        }

        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(saveable_results, f, indent=2, ensure_ascii=False)

        self.log(f"\n=저장 결과 저장: {result_path}")

        # 제출 파일 경로 출력
        inference_results = results.get('inference_results', {})
        if inference_results.get('submission_path'):
            self.log(f"=저장 제출 파일: {inference_results['submission_path']}")

    def _train_multiple_models(self, train_df, eval_df):
        """
        다중 모델 학습

        Args:
            train_df: 학습 데이터
            eval_df: 평가 데이터

        Returns:
            tuple: (model_results, model_paths)
        """
        model_results = []
        model_paths = []
        failed_models = []

        for idx, model_name in enumerate(self.args.models):
            self.log(f"\n{'='*50}")
            self.log(f"모델 {idx+1}/{len(self.args.models)}: {model_name}")
            self.log(f"{'='*50}")

            try:
                # Config 로드
                config = load_model_config(model_name)
                self._override_config(config)

                # 모델 및 토크나이저 로드
                model, tokenizer = load_model_and_tokenizer(config, logger=self.logger)

                # Dataset 생성
                model_type = config.model.get('type', 'encoder_decoder')

                train_dataset = DialogueSummarizationDataset(
                    dialogues=train_df['dialogue'].tolist(),
                    summaries=train_df['summary'].tolist(),
                    tokenizer=tokenizer,
                    encoder_max_len=config.tokenizer.encoder_max_len,
                    decoder_max_len=config.tokenizer.decoder_max_len,
                    preprocess=True,
                    model_type=model_type
                )

                eval_dataset = DialogueSummarizationDataset(
                    dialogues=eval_df['dialogue'].tolist(),
                    summaries=eval_df['summary'].tolist(),
                    tokenizer=tokenizer,
                    encoder_max_len=config.tokenizer.encoder_max_len,
                    decoder_max_len=config.tokenizer.decoder_max_len,
                    preprocess=True,
                    model_type=model_type
                )

                # Trainer 생성 및 학습
                model_output_dir = self.output_dir / f"model_{idx}_{model_name.replace('-', '_')}"
                model_output_dir.mkdir(parents=True, exist_ok=True)

                config.training.output_dir = str(model_output_dir)

                trainer = create_trainer(
                    config=config,
                    model=model,
                    tokenizer=tokenizer,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    use_wandb=getattr(self.args, 'use_wandb', True),
                    logger=self.logger
                )

                # 학습 수행
                train_result = trainer.train()

                # Get model path from training result (model already saved by train())
                final_model_path = train_result.get('final_model_path', str(model_output_dir / 'final_model'))
                model_paths.append(final_model_path)

                # Get evaluation metrics from training result
                eval_metrics = train_result.get('eval_metrics', {})
                model_results.append({
                    'model_name': model_name,
                    'model_path': final_model_path,
                    'eval_metrics': eval_metrics,
                    'status': 'success'
                })

                self.log(f"\n✅ {model_name} 학습 완료")

            except Exception as e:
                # 오류 발생 시 로깅하고 다음 모델로 계속
                error_msg = f"❌ {model_name} 학습 실패: {type(e).__name__}: {str(e)}"
                self.log(f"\n{error_msg}")

                # 오류 상세 로그 저장
                import traceback
                error_log_dir = self.output_dir / "errors"
                error_log_dir.mkdir(parents=True, exist_ok=True)
                error_log_path = error_log_dir / f"{model_name}_error.log"

                with open(error_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"모델: {model_name}\n")
                    f.write(f"오류 타입: {type(e).__name__}\n")
                    f.write(f"오류 메시지: {str(e)}\n\n")
                    f.write("상세 트레이스백:\n")
                    f.write(traceback.format_exc())

                self.log(f"  오류 로그 저장: {error_log_path}")

                # 실패 모델 기록
                failed_models.append({
                    'model_name': model_name,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'status': 'failed'
                })

                model_results.append({
                    'model_name': model_name,
                    'model_path': None,
                    'eval_metrics': {},
                    'status': 'failed',
                    'error': str(e)
                })

                # 다음 모델로 계속
                continue

        # 최종 결과 요약
        self.log(f"\n{'='*50}")
        self.log("모델 학습 결과 요약")
        self.log(f"{'='*50}")
        success_count = sum(1 for r in model_results if r.get('status') == 'success')
        failed_count = len(model_results) - success_count
        self.log(f"✅ 성공: {success_count}/{len(self.args.models)} 모델")
        self.log(f"❌ 실패: {failed_count}/{len(self.args.models)} 모델")

        if failed_models:
            self.log("\n실패한 모델 목록:")
            for failed in failed_models:
                self.log(f"  - {failed['model_name']}: {failed['error_type']}")

        return model_results, model_paths

    def _create_and_evaluate_ensemble(self, model_paths, eval_df):
        """
        앙상블 생성 및 평가

        Args:
            model_paths: 모델 경로 리스트
            eval_df: 평가 데이터

        Returns:
            dict: 앙상블 평가 지표
        """
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from rouge import Rouge
            import torch

            # 모델 로드
            models = []
            tokenizers = []

            for model_path in model_paths:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)

                if torch.cuda.is_available():
                    model = model.cuda()
                model.eval()

                models.append(model)
                tokenizers.append(tokenizer)

            # 앙상블 생성
            manager = ModelManager(logger=self.logger)
            manager.models = models
            manager.tokenizers = tokenizers
            manager.model_names = self.args.models

            if self.args.ensemble_strategy in ['weighted_avg', 'rouge_weighted']:
                ensemble = manager.create_ensemble(
                    ensemble_type='weighted',
                    weights=self.args.ensemble_weights
                )
            else:
                ensemble = manager.create_ensemble(
                    ensemble_type='voting',
                    voting='hard'
                )

            # 샘플 추출
            dialogues = eval_df['dialogue'].tolist()[:100]
            references = eval_df['summary'].tolist()[:100]

            predictions = ensemble.predict(
                dialogues=dialogues,
                max_length=200,
                num_beams=4,
                batch_size=8
            )

            # ROUGE 계산
            rouge = Rouge()
            scores = rouge.get_scores(predictions, references, avg=True)

            return {
                'ensemble_rouge_1_f1': scores['rouge-1']['f'],
                'ensemble_rouge_2_f1': scores['rouge-2']['f'],
                'ensemble_rouge_l_f1': scores['rouge-l']['f'],
            }

        except Exception as e:
            self.log(f"    앙상블 평가 오류 발생: {e}")
            return {}

    def _integrate_solar_api(self, eval_df):
        """
        Solar API 통합

        Args:
            eval_df: 평가 데이터

        Returns:
            dict: Solar API 평가 결과
        """
        try:
            # Solar API 클라이언트 생성
            solar_client = create_solar_api(use_cache=True)

            # 소량 샘플 추출
            dialogues = eval_df['dialogue'].tolist()[:50]
            references = eval_df['summary'].tolist()[:50]

            self.log(f"  Solar API processing {len(dialogues)} samples ...")

            predictions = solar_client.batch_generate(
                dialogues=dialogues,
                batch_size=10,
                use_few_shot=True,
                preprocess=True
            )

            # ROUGE 계산
            from rouge import Rouge
            rouge = Rouge()
            scores = rouge.get_scores(predictions, references, avg=True)

            return {
                'solar_rouge_1_f1': scores['rouge-1']['f'],
                'solar_rouge_2_f1': scores['rouge-2']['f'],
                'solar_rouge_l_f1': scores['rouge-l']['f'],
                'n_samples': len(predictions)
            }

        except Exception as e:
            self.log(f"    Solar API 통합 오류 발생: {e}")
            return {}

    def _apply_tta(self, model_paths, eval_df):
        """
        TTA (Test Time Augmentation) 적용

        Args:
            model_paths: 모델 경로 리스트
            eval_df: 평가 데이터

        Returns:
            dict: TTA 결과
        """
        try:
            self.log(f"  TTA 전략: {', '.join(self.args.tta_strategies)}")
            self.log(f"  증강 횟수: {self.args.tta_num_aug}")

            # TTA 기능 구현 예정
            self.log("    TTA 기능은 아직 구현 중입니다.")

            return {
                'tta_applied': False,
                'strategies': self.args.tta_strategies,
                'num_aug': self.args.tta_num_aug
            }

        except Exception as e:
            self.log(f"    TTA 적용 오류 발생: {e}")
            return {}

    def _create_submission(self, model_paths):
        """
        추론 및 제출 파일 생성

        Args:
            model_paths: 학습된 모델 경로 리스트

        Returns:
            dict: 추론 결과
                - submission_path: 제출 파일 경로
                - num_predictions: 예측 개수
                - best_model_used: 사용된 최적 모델
        """
        try:
            import pandas as pd
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch

            # 테스트 데이터 로드
            test_data_path = getattr(self.args, 'test_data', 'data/raw/test.csv')
            self.log(f"  테스트 데이터 로드: {test_data_path}")
            test_df = pd.read_csv(test_data_path)
            self.log(f"  테스트 샘플 수: {len(test_df)}")

            # 성공한 모델 중 첫 번째 모델 사용 (가장 먼저 학습 완료된 모델)
            if not model_paths:
                self.log("    ❌ 사용 가능한 모델이 없습니다.")
                return {
                    'submission_path': None,
                    'num_predictions': 0,
                    'error': '사용 가능한 모델 없음'
                }

            best_model_path = model_paths[0]
            self.log(f"  사용 모델: {best_model_path}")

            # 모델 및 토크나이저 로드
            model = AutoModelForSeq2SeqLM.from_pretrained(best_model_path)
            tokenizer = AutoTokenizer.from_pretrained(best_model_path)

            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()

            # 배치 추론
            predictions = []
            batch_size = getattr(self.args, 'inference_batch_size', 32)
            self.log(f"  배치 크기: {batch_size}")

            dialogues = test_df['dialogue'].tolist()

            for i in range(0, len(dialogues), batch_size):
                batch_dialogues = dialogues[i:i+batch_size]

                # 토크나이징
                inputs = tokenizer(
                    batch_dialogues,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )

                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                # 생성
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=getattr(self.args, 'max_length', 100),
                        num_beams=getattr(self.args, 'num_beams', 4),
                        early_stopping=True,
                        no_repeat_ngram_size=getattr(self.args, 'no_repeat_ngram_size', 2)
                    )

                # 디코딩
                batch_predictions = tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )
                predictions.extend(batch_predictions)

                if (i // batch_size + 1) % 10 == 0:
                    self.log(f"    진행: {i+len(batch_predictions)}/{len(dialogues)}")

            # 제출 파일 생성
            submission_df = pd.DataFrame({
                'id': test_df['id'],
                'summary': predictions
            })

            # 제출 파일 저장
            submission_dir = self.output_dir / "submissions"
            submission_dir.mkdir(parents=True, exist_ok=True)

            experiment_name = getattr(self.args, 'experiment_name', 'full_pipeline')
            submission_path = submission_dir / f"{experiment_name}_submission.csv"

            submission_df.to_csv(submission_path, index=False)

            self.log(f"  ✅ 제출 파일 생성 완료: {submission_path}")
            self.log(f"  예측 개수: {len(predictions)}")

            return {
                'submission_path': str(submission_path),
                'num_predictions': len(predictions),
                'best_model_used': best_model_path
            }

        except Exception as e:
            import traceback
            self.log(f"    ❌ 추론 오류 발생: {e}")
            self.log(f"    상세: {traceback.format_exc()}")
            return {
                'submission_path': None,
                'num_predictions': 0,
                'error': str(e)
            }

    def _override_config(self, config):
        """Config 오버라이드"""
        if hasattr(self.args, 'epochs') and self.args.epochs is not None:
            config.training.epochs = self.args.epochs

        if hasattr(self.args, 'batch_size') and self.args.batch_size is not None:
            config.training.batch_size = self.args.batch_size

        if hasattr(self.args, 'learning_rate') and self.args.learning_rate is not None:
            config.training.learning_rate = self.args.learning_rate

    def _extract_eval_metrics(self, log_history):
        """평가 지표 추출"""
        eval_metrics = {}

        for log_entry in reversed(log_history):
            if 'eval_loss' in log_entry:
                for key, value in log_entry.items():
                    if key.startswith('eval_'):
                        eval_metrics[key] = value
                break

        return eval_metrics


# ==================== 편의 함수 ==================== #
def create_full_pipeline_trainer(args, logger, wandb_logger=None):
    """
    FullPipelineTrainer 생성 편의 함수

    Args:
        args: 명령행 인자
        logger: Logger 인스턴스
        wandb_logger: WandB Logger (선택 사항)

    Returns:
        FullPipelineTrainer 인스턴스
    """
    return FullPipelineTrainer(args, logger, wandb_logger)
