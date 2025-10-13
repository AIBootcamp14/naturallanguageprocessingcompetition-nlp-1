# ==================== pt0   ==================== #
"""
pt0   ÃÂ¤\

PRD 11: pt0   
- lp  (, D D)
- X  (, tX)
-   (, 8t)
"""

# ---------------------- \ |t ---------------------- #
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json

# ----------------------  |t ---------------------- #
import pandas as pd
import numpy as np


# ==================== DataQualityValidator t ==================== #
class DataQualityValidator:
    """pt0   t"""

    def __init__(self, logger=None):
        """
        Args:
            logger: Logger x4
        """
        self.logger = logger
        self.validation_results = []

        self._log("DataQualityValidator 0T")

    def _log(self, msg: str):
        """\E |"""
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
        pt0 i 

        Args:
            df: ` pt0
            required_columns: D  
            check_duplicates:   
            check_statistics:   
            check_anomalies: tX  

        Returns:
              T
        """
        self._log("\n" + "="*60)
        self._log("pt0   ÃÂ")
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

        # 1. lp 
        self._log("\n[1/4] lp ...")
        structural_result = self._validate_structure(df, required_columns)
        results['structural_validation'] = structural_result

        if not structural_result['passed']:
            results['passed'] = False
            results['issues'].extend(structural_result['issues'])

        # 2. X 
        self._log("\n[2/4] X ...")
        if check_duplicates:
            semantic_result = self._validate_semantics(df, required_columns)
            results['semantic_validation'] = semantic_result

            if semantic_result['duplicate_count'] > 0:
                results['issues'].append(
                    f" : {semantic_result['duplicate_count']}"
                )

        # 3.  
        self._log("\n[3/4]  ...")
        if check_statistics:
            statistical_result = self._validate_statistics(df, required_columns)
            results['statistical_validation'] = statistical_result

        # 4. tX 
        self._log("\n[4/4] tX ...")
        if check_anomalies:
            anomaly_result = self._detect_anomalies(df, required_columns)
            results['anomaly_detection'] = anomaly_result

            if anomaly_result['anomaly_count'] > 0:
                results['issues'].append(
                    f"tX : {anomaly_result['anomaly_count']}"
                )

        # \ 
        self._log("\n" + "="*60)
        if results['passed'] and len(results['issues']) == 0:
            self._log(" pt0   !")
        else:
            self._log(" pt0  t :")
            for issue in results['issues']:
                self._log(f"  - {issue}")

        self._log("="*60)

        #  
        self.validation_results.append(results)

        return results

    def _validate_structure(
        self,
        df: pd.DataFrame,
        required_columns: List[str]
    ) -> Dict[str, Any]:
        """
        lp  (, D D)

        Args:
            df: pt0
            required_columns: D 

        Returns:
            lp  
        """
        issues = []
        passed = True

        # 1. D  t Ux
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"D  }: {missing_columns}")
            passed = False
            self._log(f"  L D  }: {missing_columns}")
        else:
            self._log(f"   D  t: {required_columns}")

        # 2. NULL  Ux
        null_counts = {}
        for col in required_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                null_counts[col] = int(null_count)

                if null_count > 0:
                    issues.append(f"{col}  NULL  {null_count}")
                    passed = False
                    # FIXME: Corrupted log message

        if not issues:
            self._log("   NULL  L")

        # 3. H 8 Ux
        empty_counts = {}
        for col in required_columns:
            if col in df.columns and df[col].dtype == 'object':
                empty_count = (df[col].str.strip() == '').sum()
                empty_counts[col] = int(empty_count)

                if empty_count > 0:
                    issues.append(f"{col}  H 8 {empty_count}")
                    # FIXME: Corrupted log message

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
        X  (, (4)

        Args:
            df: pt0
            required_columns:  

        Returns:
            X  
        """
        # 1. D  Ux
        duplicate_rows = df.duplicated().sum()
        self._log(f"  D : {duplicate_rows}")

        # 2.   Ux
        column_duplicates = {}
        for col in required_columns:
            if col in df.columns:
                dup_count = df[col].duplicated().sum()
                column_duplicates[col] = int(dup_count)
                # FIXME: Corrupted log message

        # 3. (4  (dialogue )
        pattern_issues = []
        if 'dialogue' in df.columns:
            # #Person1#, #Person2# (4 Ux
            has_person_pattern = df['dialogue'].str.contains(
                r'#Person\d+#',
                regex=True,
                na=False
            )
            invalid_pattern_count = (~has_person_pattern).sum()

            if invalid_pattern_count > 0:
                pattern_issues.append(
                    f"T  $X: {invalid_pattern_count}"
                )
                self._log(f"   T  $X: {invalid_pattern_count}")

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
          (, 8t)

        Args:
            df: pt0
            required_columns:  

        Returns:
              
        """
        statistics = {}

        for col in required_columns:
            if col in df.columns and df[col].dtype == 'object':
                # M 8t 
                lengths = df[col].str.len()

                col_stats = {
                    'mean_length': float(lengths.mean()),
                    'median_length': float(lengths.median()),
                    'min_length': int(lengths.min()),
                    'max_length': int(lengths.max()),
                    'std_length': float(lengths.std())
                }

                statistics[col] = col_stats

                # FIXME: Corrupted log message
                self._log(f"    - : {col_stats['mean_length']:.1f}")
                self._log(f"    - Y: {col_stats['median_length']:.1f}")
                self._log(f"    - : [{col_stats['min_length']}, {col_stats['max_length']}]")

        return statistics

    def _detect_anomalies(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
        z_threshold: float = 3.0
    ) -> Dict[str, Any]:
        """
        tX  (8t 0)

        Args:
            df: pt0
            required_columns:  
            z_threshold: Z-score 

        Returns:
            tX  
        """
        anomalies = {}
        total_anomalies = 0

        for col in required_columns:
            if col in df.columns and df[col].dtype == 'object':
                # M 8t\ tX 
                lengths = df[col].str.len()
                mean_len = lengths.mean()
                std_len = lengths.std()

                # Z-score ÃÂ°
                z_scores = np.abs((lengths - mean_len) / std_len)
                anomaly_mask = z_scores > z_threshold

                anomaly_count = anomaly_mask.sum()
                total_anomalies += anomaly_count

                if anomaly_count > 0:
                    anomaly_indices = df[anomaly_mask].index.tolist()
                    anomaly_lengths = lengths[anomaly_mask].tolist()

                    anomalies[col] = {
                        'count': int(anomaly_count),
                        'indices': anomaly_indices[:10],  # \ 10
                        'example_lengths': [int(l) for l in anomaly_lengths[:5]]
                    }

                    # FIXME: Corrupted log message
                    self._log(f"     8t: {anomalies[col]['example_lengths']}")

        if total_anomalies == 0:
            self._log("   tX L")

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
        |  (CSV)

        Args:
            file_path: CSV | \
            **kwargs: validate_dataframe |0

        Returns:
             
        """
        self._log(f"\n| : {file_path}")

        # CSV \
        df = pd.read_csv(file_path)

        #  
        results = self.validate_dataframe(df, **kwargs)

        # | \ 
        results['file_path'] = file_path

        return results

    def save_results(self, output_path: str):
        """
          

        Args:
            output_path:  \
        """
        if not self.validation_results:
            self._log("`   ÃÂµ.")
            return

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)

        self._log(f"  : {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """
          }

        Returns:
            } T
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


# ==================== X h ==================== #
def create_validator(logger=None) -> DataQualityValidator:
    """
    Validator 1 X h

    Args:
        logger: Logger x4

    Returns:
        DataQualityValidator x4
    """
    return DataQualityValidator(logger=logger)


def quick_validate(
    df: pd.DataFrame,
    required_columns: List[str] = ['dialogue', 'summary']
) -> bool:
    """
    `x  ( )

    Args:
        df: pt0
        required_columns: D 

    Returns:
          
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
