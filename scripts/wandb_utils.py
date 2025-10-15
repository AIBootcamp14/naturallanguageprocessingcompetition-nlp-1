#!/usr/bin/env python3
"""
Wandb 유틸리티 함수 모음
- 실험 추적, 시각화, Artifacts 관리
"""

import os
import wandb
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path


def init_wandb(
    experiment_name: str,
    config: Dict[str, Any],
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    notes: Optional[str] = None
) -> wandb.sdk.wandb_run.Run:
    """
    Wandb 초기화

    Args:
        experiment_name: 실험 이름 (예: "exp4-lp-0.6")
        config: 하이퍼파라미터 딕셔너리
        tags: 태그 리스트 (예: ["length-penalty", "exp4"])
        group: 실험 그룹 (예: "exp4-length-penalty")
        notes: 실험 노트

    Returns:
        wandb.Run 객체
    """
    # .env에서 설정 가져오기
    project = os.getenv("WANDB_PROJECT", "dialogue-summarization-competition")
    entity = os.getenv("WANDB_ENTITY", "bkan-ai")

    run = wandb.init(
        project=project,
        entity=entity,
        name=experiment_name,
        config=config,
        tags=tags or [],
        group=group,
        notes=notes,
        reinit=True  # 여러 run을 순차적으로 실행 가능
    )

    print(f"✅ Wandb 초기화 완료: {project}/{experiment_name}")
    print(f"   Dashboard: {run.url}")

    return run


def log_inference_results(
    dataset_name: str,
    rouge_scores: Dict[str, float],
    final_score: float,
    predictions: Optional[List[str]] = None,
    num_samples: Optional[int] = None,
    prefix: str = ""
):
    """
    추론 결과를 Wandb에 기록

    Args:
        dataset_name: 데이터셋 이름 ("dev", "test")
        rouge_scores: ROUGE 점수 딕셔너리 {"rouge1": 0.56, ...}
        final_score: 최종 점수
        predictions: 예측 결과 리스트 (optional)
        num_samples: 샘플 개수
        prefix: 로그 키 prefix (예: "sweep/")
    """
    # 기본 메트릭
    metrics = {
        f"{prefix}{dataset_name}/rouge1": rouge_scores.get("rouge1", 0.0),
        f"{prefix}{dataset_name}/rouge2": rouge_scores.get("rouge2", 0.0),
        f"{prefix}{dataset_name}/rougeL": rouge_scores.get("rougeL", 0.0),
        f"{prefix}{dataset_name}/final_score": final_score,
    }

    if num_samples:
        metrics[f"{prefix}{dataset_name}/num_samples"] = num_samples

    wandb.log(metrics)

    # 예측 결과 샘플 저장 (처음 10개)
    if predictions:
        samples = predictions[:10]
        table = wandb.Table(
            columns=["idx", "prediction"],
            data=[[i, pred] for i, pred in enumerate(samples)]
        )
        wandb.log({f"{prefix}{dataset_name}/predictions_sample": table})

    print(f"✅ {dataset_name.upper()} 결과 Wandb 기록 완료")


def log_dev_test_comparison(
    dev_score: float,
    test_score: float,
    experiment_id: str,
    config_changes: Optional[Dict[str, Any]] = None
):
    """
    Dev/Test 점수 비교 기록

    Args:
        dev_score: Dev set ROUGE 점수
        test_score: Test set ROUGE 점수
        experiment_id: 실험 ID (예: "exp4")
        config_changes: 변경된 설정 (optional)
    """
    gap = abs(dev_score - test_score)
    ratio = dev_score / test_score if test_score > 0 else 0

    metrics = {
        "comparison/dev_score": dev_score,
        "comparison/test_score": test_score,
        "comparison/gap": gap,
        "comparison/gap_percentage": (gap / test_score * 100) if test_score > 0 else 0,
        "comparison/ratio": ratio,
        "comparison/experiment_id": experiment_id,
    }

    wandb.log(metrics)

    # 비교 테이블 생성
    table = wandb.Table(
        columns=["Metric", "Dev", "Test", "Gap", "Ratio"],
        data=[
            ["ROUGE Score", f"{dev_score:.4f}", f"{test_score:.4f}", f"{gap:.4f}", f"{ratio:.2f}"]
        ]
    )
    wandb.log({"comparison/summary_table": table})

    # Config 변경사항 기록
    if config_changes:
        wandb.config.update({"changes": config_changes})

    print(f"✅ Dev/Test 비교 기록 완료 (Gap: {gap:.4f}, Ratio: {ratio:.2f})")


def log_artifact(
    artifact_name: str,
    artifact_type: str,
    artifact_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None
):
    """
    Artifact 저장 (모델 checkpoint, 예측 결과 등)

    Args:
        artifact_name: Artifact 이름 (예: "model-exp4-lp-0.6")
        artifact_type: Artifact 타입 ("model", "predictions", "dataset")
        artifact_path: 저장할 파일/디렉토리 경로
        metadata: 메타데이터 딕셔너리
        description: Artifact 설명
    """
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=description,
        metadata=metadata or {}
    )

    # 파일 또는 디렉토리 추가
    if os.path.isdir(artifact_path):
        artifact.add_dir(artifact_path)
    elif os.path.isfile(artifact_path):
        artifact.add_file(artifact_path)
    else:
        print(f"⚠️ 경로를 찾을 수 없습니다: {artifact_path}")
        return

    wandb.log_artifact(artifact)
    print(f"✅ Artifact 저장 완료: {artifact_name} ({artifact_type})")


def create_comparison_table(experiments: List[Dict[str, Any]]) -> wandb.Table:
    """
    실험 비교 테이블 생성

    Args:
        experiments: 실험 정보 리스트
            [{"id": "exp0", "config": {...}, "dev": 50.0, "test": 46.9, ...}, ...]

    Returns:
        wandb.Table 객체
    """
    columns = ["Exp ID", "Config Changes", "Dev Score", "Test Score", "Gap", "Status"]
    data = []

    for exp in experiments:
        exp_id = exp.get("id", "Unknown")
        config_str = ", ".join([f"{k}={v}" for k, v in exp.get("config", {}).items()])
        dev_score = exp.get("dev", 0.0)
        test_score = exp.get("test", 0.0)
        gap = abs(dev_score - test_score)
        status = "✅" if exp.get("success", False) else "❌"

        data.append([exp_id, config_str, dev_score, test_score, gap, status])

    table = wandb.Table(columns=columns, data=data)
    return table


def log_length_penalty_sweep(lp_value: float, dev_rouge: float, step: int):
    """
    Length Penalty sweep 결과 기록

    Args:
        lp_value: Length Penalty 값
        dev_rouge: Dev ROUGE 점수
        step: Sweep step (0, 1, 2, 3)
    """
    wandb.log({
        "sweep/length_penalty": lp_value,
        "sweep/dev_rouge": dev_rouge,
        "sweep/step": step
    })


def create_correlation_plot(
    dev_scores: List[float],
    test_scores: List[float],
    exp_names: List[str]
):
    """
    Dev vs Test 상관관계 산점도 생성

    Args:
        dev_scores: Dev set 점수 리스트
        test_scores: Test set 점수 리스트
        exp_names: 실험 이름 리스트

    Returns:
        wandb.Plot 객체
    """
    # DataFrame 생성
    df = pd.DataFrame({
        "Dev Score": dev_scores,
        "Test Score": test_scores,
        "Experiment": exp_names
    })

    # Wandb scatter plot
    table = wandb.Table(dataframe=df)
    plot = wandb.plot.scatter(
        table,
        x="Dev Score",
        y="Test Score",
        title="Dev vs Test Score Correlation"
    )

    return plot


def calculate_correlation(
    dev_scores: List[float],
    test_scores: List[float]
) -> Dict[str, float]:
    """
    Dev/Test 상관관계 계산

    Args:
        dev_scores: Dev set 점수 리스트
        test_scores: Test set 점수 리스트

    Returns:
        상관관계 통계 딕셔너리
    """
    dev_arr = np.array(dev_scores)
    test_arr = np.array(test_scores)

    # Pearson 상관계수
    correlation = np.corrcoef(dev_arr, test_arr)[0, 1]

    # 평균 절대 오차
    mae = np.mean(np.abs(dev_arr - test_arr))

    # 평균 비율
    ratios = dev_arr / test_arr
    mean_ratio = np.mean(ratios)

    return {
        "correlation": correlation,
        "mae": mae,
        "mean_ratio": mean_ratio,
        "std_ratio": np.std(ratios)
    }


def log_correlation_analysis(
    dev_scores: List[float],
    test_scores: List[float],
    exp_names: List[str]
):
    """
    상관관계 분석 결과를 Wandb에 기록

    Args:
        dev_scores: Dev set 점수 리스트
        test_scores: Test set 점수 리스트
        exp_names: 실험 이름 리스트
    """
    # 상관관계 계산
    stats = calculate_correlation(dev_scores, test_scores)

    # 메트릭 기록
    wandb.log({
        "correlation/pearson": stats["correlation"],
        "correlation/mae": stats["mae"],
        "correlation/mean_ratio": stats["mean_ratio"],
        "correlation/std_ratio": stats["std_ratio"]
    })

    # 산점도 생성
    plot = create_correlation_plot(dev_scores, test_scores, exp_names)
    wandb.log({"correlation/scatter_plot": plot})

    # 상세 테이블
    table_data = []
    for exp, dev, test in zip(exp_names, dev_scores, test_scores):
        gap = abs(dev - test)
        ratio = dev / test if test > 0 else 0
        table_data.append([exp, f"{dev:.4f}", f"{test:.4f}", f"{gap:.4f}", f"{ratio:.2f}"])

    table = wandb.Table(
        columns=["Experiment", "Dev", "Test", "Gap", "Ratio"],
        data=table_data
    )
    wandb.log({"correlation/detail_table": table})

    print(f"✅ 상관관계 분석 완료:")
    print(f"   Pearson Correlation: {stats['correlation']:.4f}")
    print(f"   Mean Absolute Error: {stats['mae']:.4f}")
    print(f"   Mean Ratio: {stats['mean_ratio']:.4f} ± {stats['std_ratio']:.4f}")

    return stats


def finish_run(summary_metrics: Optional[Dict[str, Any]] = None):
    """
    Wandb run 종료

    Args:
        summary_metrics: 최종 요약 메트릭 (optional)
    """
    if summary_metrics:
        for key, value in summary_metrics.items():
            wandb.run.summary[key] = value

    wandb.finish()
    print("✅ Wandb run 종료")


if __name__ == "__main__":
    # 테스트 코드
    print("Wandb Utils 모듈 로드 완료")
    print("사용 가능한 함수:")
    print("  - init_wandb()")
    print("  - log_inference_results()")
    print("  - log_dev_test_comparison()")
    print("  - log_artifact()")
    print("  - create_comparison_table()")
    print("  - log_length_penalty_sweep()")
    print("  - log_correlation_analysis()")
    print("  - finish_run()")