# ==================== 데이터 품질 검증 ==================== #
"""
데이터 품질 검증 체크

PRD 11: 데이터 품질 검증 모듈
- 구조 검증 (컬럼, 데이터 타입)
- 의미 검증 (중복, 패턴)
- 통계 검증 (길이, 분포)
"""

# ---------------------- 외부 라이브러리 ---------------------- #
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json

# ---------------------- 외부 패키지 ---------------------- #
import pandas as pd
import numpy as np


# ==================== DataQualityValidator 클래스 ==================== #
class DataQualityValidator:
    """데이터 품질 검증 클래스"""

    def __init__(self, logger=None):
        """
        Args:
            logger: Logger 인스턴스
        """
        self.logger = logger
        self.validation_results = []

        self._log("DataQualityValidator 초기화")

    def _log(self, msg: str):
        """로그 출력"""
        if self.logger:
            self.logger.write(msg)
        else:
            print(msg)

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        required_columns: List[str] = ['dialogue', 'summary'],
        check_duplicates: bool = True,
        check_statistics: bool = True,
        check_anomalies: bool = True
    ) -> Dict[str, Any]:
        """
        데이터프레임 검증 수행

        Args:
            df: 검증할 데이터프레임
            required_columns: 필수 컬럼 목록
            check_duplicates: 중복 검사 여부
            check_statistics: 통계 검사 여부
            check_anomalies: 이상치 검사 여부

        Returns:
            검증 결과 딕셔너리
        """
        self._log("\n" + "="*60)
        self._log("데이터 품질 검증 수행")
        self._log("="*60)

        results = {
            'total_rows': len(df),
            'structural_validation': {},
            'semantic_validation': {},
            'statistical_validation': {},
            'anomaly_detection': {},
            'issues': [],
            'passed': True
        }

        # 1. 구조 검증
        self._log("\n[1/4] 구조 검증...")
        structural_result = self._validate_structure(df, required_columns)
        results['structural_validation'] = structural_result

        if not structural_result['passed']:
            results['passed'] = False
            results['issues'].extend(structural_result['issues'])

        # 2. 의미 검증
        self._log("\n[2/4] 의미 검증...")
        if check_duplicates:
            semantic_result = self._validate_semantics(df, required_columns)
            results['semantic_validation'] = semantic_result

            if semantic_result['duplicate_count'] > 0:
                results['issues'].append(
                    f"중복 행: {semantic_result['duplicate_count']}개"
                )

        # 3. 통계 검증
        self._log("\n[3/4] 통계 검증...")
        if check_statistics:
            statistical_result = self._validate_statistics(df, required_columns)
            results['statistical_validation'] = statistical_result

        # 4. 이상치 검증
        self._log("\n[4/4] 이상치 검증...")
        if check_anomalies:
            anomaly_result = self._detect_anomalies(df, required_columns)
            results['anomaly_detection'] = anomaly_result

            if anomaly_result['anomaly_count'] > 0:
                results['issues'].append(
                    f"이상치 발견: {anomaly_result['anomaly_count']}개"
                )

        # 최종 요약
        self._log("\n" + "="*60)
        if results['passed'] and len(results['issues']) == 0:
            self._log("✅ 데이터 품질 검증 통과!")
        else:
            self._log("⚠️ 데이터 품질 문제 발견:")
            for issue in results['issues']:
                self._log(f"  - {issue}")

        self._log("="*60)

        # 결과 저장
        self.validation_results.append(results)

        return results

    def _validate_structure(
        self,
        df: pd.DataFrame,
        required_columns: List[str]
    ) -> Dict[str, Any]:
        """
        구조 검증 (컬럼, 데이터 타입)

        Args:
            df: 데이터프레임
            required_columns: 필수 컬럼 목록

        Returns:
            구조 검증 결과
        """
        issues = []
        passed = True

        # 1. 필수 컬럼 존재 여부 체크
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"누락된 컬럼: {missing_columns}")
            passed = False
            self._log(f"  ❌ 누락된 컬럼 발견: {missing_columns}")
        else:
            self._log(f"  ✅ 모든 필수 컬럼 존재: {required_columns}")

        # 2. NULL 값 체크
        null_counts = {}
        for col in required_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                null_counts[col] = int(null_count)

                if null_count > 0:
                    issues.append(f"{col} 컬럼에 NULL 값 {null_count}개")
                    passed = False
                    self._log(f"  ❌ {col}: NULL 값 {null_count}개 발견")

        if not issues:
            self._log("  ✅ NULL 값 없음")

        # 3. 빈 문자열 체크
        empty_counts = {}
        for col in required_columns:
            if col in df.columns and df[col].dtype == 'object':
                empty_count = (df[col].str.strip() == '').sum()
                empty_counts[col] = int(empty_count)

                if empty_count > 0:
                    issues.append(f"{col} 컬럼에 빈 문자열 {empty_count}개")
                    self._log(f"  ⚠️ {col}: 빈 문자열 {empty_count}개 발견")

        return {
            'passed': passed,
            'issues': issues,
            'missing_columns': missing_columns,
            'null_counts': null_counts,
            'empty_counts': empty_counts
        }

    def _validate_semantics(
        self,
        df: pd.DataFrame,
        required_columns: List[str]
    ) -> Dict[str, Any]:
        """
        의미 검증 (중복, 패턴)

        Args:
            df: 데이터프레임
            required_columns: 검사할 컬럼

        Returns:
            의미 검증 결과
        """
        # 1. 전체 행 중복 체크
        duplicate_rows = df.duplicated().sum()
        self._log(f"  📊 전체 중복 행: {duplicate_rows}개")

        # 2. 컬럼별 중복 체크
        column_duplicates = {}
        for col in required_columns:
            if col in df.columns:
                dup_count = df[col].duplicated().sum()
                column_duplicates[col] = int(dup_count)
                self._log(f"  📊 {col} 중복: {dup_count}개")

        # 3. 패턴 검증 (dialogue 컬럼)
        pattern_issues = []
        if 'dialogue' in df.columns:
            # #Person1#, #Person2# 패턴 체크
            has_person_pattern = df['dialogue'].str.contains(
                r'#Person\d+#',
                regex=True,
                na=False
            )
            invalid_pattern_count = (~has_person_pattern).sum()

            if invalid_pattern_count > 0:
                pattern_issues.append(
                    f"대화 패턴 유효하지 않음: {invalid_pattern_count}개"
                )
                self._log(f"  ⚠️ 대화 패턴 유효하지 않음: {invalid_pattern_count}개")

        return {
            'duplicate_count': int(duplicate_rows),
            'column_duplicates': column_duplicates,
            'pattern_issues': pattern_issues
        }

    def _validate_statistics(
        self,
        df: pd.DataFrame,
        required_columns: List[str]
    ) -> Dict[str, Any]:
        """
        통계 검증 (길이, 분포)

        Args:
            df: 데이터프레임
            required_columns: 검사할 컬럼

        Returns:
            통계 검증 결과
        """
        statistics = {}

        for col in required_columns:
            if col in df.columns and df[col].dtype == 'object':
                # 문자열 길이 통계
                lengths = df[col].str.len()

                col_stats = {
                    'mean_length': float(lengths.mean()),
                    'median_length': float(lengths.median()),
                    'min_length': int(lengths.min()),
                    'max_length': int(lengths.max()),
                    'std_length': float(lengths.std())
                }

                statistics[col] = col_stats

                self._log(f"  📊 {col} 통계:")
                self._log(f"    - 평균: {col_stats['mean_length']:.1f}")
                self._log(f"    - 중앙값: {col_stats['median_length']:.1f}")
                self._log(f"    - 범위: [{col_stats['min_length']}, {col_stats['max_length']}]")

        return statistics

    def _detect_anomalies(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
        z_threshold: float = 3.0
    ) -> Dict[str, Any]:
        """
        이상치 검출 (길이 기반)

        Args:
            df: 데이터프레임
            required_columns: 검사할 컬럼
            z_threshold: Z-score 임계값

        Returns:
            이상치 검출 결과
        """
        anomalies = {}
        total_anomalies = 0

        for col in required_columns:
            if col in df.columns and df[col].dtype == 'object':
                # 문자열 길이로 이상치 검출
                lengths = df[col].str.len()
                mean_len = lengths.mean()
                std_len = lengths.std()

                # Z-score 계산
                z_scores = np.abs((lengths - mean_len) / std_len)
                anomaly_mask = z_scores > z_threshold

                anomaly_count = anomaly_mask.sum()
                total_anomalies += anomaly_count

                if anomaly_count > 0:
                    anomaly_indices = df[anomaly_mask].index.tolist()
                    anomaly_lengths = lengths[anomaly_mask].tolist()

                    anomalies[col] = {
                        'count': int(anomaly_count),
                        'indices': anomaly_indices[:10],  # 최대 10개
                        'example_lengths': [int(l) for l in anomaly_lengths[:5]]
                    }

                    self._log(f"  ⚠️ {col}: 이상치 {anomaly_count}개 발견")
                    self._log(f"     예시 길이: {anomalies[col]['example_lengths']}")

        if total_anomalies == 0:
            self._log("  ✅ 이상치 없음")

        return {
            'anomaly_count': total_anomalies,
            'anomalies': anomalies,
            'z_threshold': z_threshold
        }

    def validate_file(
        self,
        file_path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        파일 검증 (CSV)

        Args:
            file_path: CSV 파일 경로
            **kwargs: validate_dataframe 인자

        Returns:
            검증 결과
        """
        self._log(f"\n파일 검증: {file_path}")

        # CSV 로드
        df = pd.read_csv(file_path)

        # 검증 수행
        results = self.validate_dataframe(df, **kwargs)

        # 파일 경로 추가
        results['file_path'] = file_path

        return results

    def save_results(self, output_path: str):
        """
        검증 결과 저장

        Args:
            output_path: 저장 경로
        """
        if not self.validation_results:
            self._log("저장할 결과가 없습니다.")
            return

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)

        self._log(f"검증 결과 저장: {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """
        검증 결과 요약

        Returns:
            요약 정보 딕셔너리
        """
        if not self.validation_results:
            return {'total_validations': 0}

        total_validations = len(self.validation_results)
        passed_count = sum(1 for r in self.validation_results if r['passed'])
        failed_count = total_validations - passed_count

        total_issues = sum(len(r['issues']) for r in self.validation_results)

        return {
            'total_validations': total_validations,
            'passed': passed_count,
            'failed': failed_count,
            'total_issues': total_issues
        }


# ==================== 편의 함수 ==================== #
def create_validator(logger=None) -> DataQualityValidator:
    """
    Validator 생성 편의 함수

    Args:
        logger: Logger 인스턴스

    Returns:
        DataQualityValidator 인스턴스
    """
    return DataQualityValidator(logger=logger)


def quick_validate(
    df: pd.DataFrame,
    required_columns: List[str] = ['dialogue', 'summary']
) -> bool:
    """
    빠른 검증 (구조만)

    Args:
        df: 데이터프레임
        required_columns: 필수 컬럼

    Returns:
        검증 통과 여부
    """
    validator = DataQualityValidator()
    results = validator.validate_dataframe(
        df,
        required_columns=required_columns,
        check_duplicates=False,
        check_statistics=False,
        check_anomalies=False
    )

    return results['passed']
