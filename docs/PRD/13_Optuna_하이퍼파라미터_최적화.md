# 🔬 Optuna 하이퍼파라미터 최적화 전략

## 🎯 개요
베이지안 최적화 기반 Optuna를 활용하여 NLP 대화 요약 모델의 하이퍼파라미터를 자동으로 최적화

## 🏗️ Optuna 최적화 시스템 아키텍처

```mermaid
graph TD
    subgraph "초기 설정"
        A[하이퍼파라미터<br/>탐색 공간 정의] --> B[목적 함수<br/>정의]
        B --> C[Study 생성]
    end

    subgraph "베이지안 최적화"
        C --> D[Trial 생성]
        D --> E[파라미터 샘플링]
        E --> F[모델 학습]
        F --> G[검증 성능 평가]
        G --> H{목표 달성?}
    end

    subgraph "샘플링 전략"
        E --> I[TPE Sampler<br/>권장]
        E --> J[GP Sampler]
        E --> K[Random Sampler]
    end

    subgraph "가지치기"
        G --> L[Median Pruner]
        L --> M{성능 부족?}
        M -->|Yes| N[Trial 중단]
        M -->|No| O[계속 학습]
    end

    H -->|No| D
    H -->|Yes| P[최적 파라미터]

    subgraph "결과 분석"
        P --> Q[파라미터 중요도]
        P --> R[최적화 히스토리]
        P --> S[파레토 프론트]
    end

    style I fill:#f96,stroke:#333,stroke-width:4px
    style L fill:#9ff,stroke:#333,stroke-width:4px
    style P fill:#ff9,stroke:#333,stroke-width:4px
```

## 📊 NLP 특화 하이퍼파라미터 탐색 공간

### 1. 모델 아키텍처 파라미터
```python
def create_model_search_space(trial):
    """NLP 모델용 탐색 공간"""
    return {
        # 모델 선택
        'model_name': trial.suggest_categorical(
            'model_name',
            ['solar-10.7b', 'polyglot-ko-12.8b', 'kullm-v2', 'koalpaca']
        ),

        # LoRA 파라미터
        'lora_r': trial.suggest_int('lora_r', 4, 64, step=4),
        'lora_alpha': trial.suggest_int('lora_alpha', 8, 128, step=8),
        'lora_dropout': trial.suggest_float('lora_dropout', 0.0, 0.3, step=0.05),

        # 양자화
        'use_8bit': trial.suggest_categorical('use_8bit', [True, False]),
        'use_4bit': trial.suggest_categorical('use_4bit', [True, False]),
    }
```

### 2. 학습 파라미터
```python
def create_training_search_space(trial):
    """학습용 탐색 공간"""
    return {
        # 학습률 (로그 스케일)
        'learning_rate': trial.suggest_float(
            'learning_rate', 1e-6, 1e-3, log=True
        ),

        # 배치 크기 (GPU 메모리 고려)
        'batch_size': trial.suggest_categorical(
            'batch_size', [4, 8, 16, 32]
        ),

        # 에폭 수
        'num_epochs': trial.suggest_int('num_epochs', 1, 10),

        # Gradient Accumulation
        'gradient_accumulation_steps': trial.suggest_categorical(
            'gradient_accumulation_steps', [1, 2, 4, 8]
        ),

        # 스케줄러
        'scheduler_type': trial.suggest_categorical(
            'scheduler_type',
            ['linear', 'cosine', 'cosine_with_restarts', 'polynomial']
        ),

        # Warmup
        'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.2, step=0.05),

        # Weight Decay
        'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.3, step=0.01),

        # Label Smoothing
        'label_smoothing': trial.suggest_float(
            'label_smoothing', 0.0, 0.2, step=0.05
        ),
    }
```

### 3. 생성 파라미터 (NLP 특화)
```python
def create_generation_search_space(trial):
    """텍스트 생성용 탐색 공간"""
    return {
        # 디코딩 전략
        'do_sample': trial.suggest_categorical('do_sample', [True, False]),

        # Temperature (창의성)
        'temperature': trial.suggest_float('temperature', 0.1, 1.0, step=0.1),

        # Top-p (nucleus sampling)
        'top_p': trial.suggest_float('top_p', 0.7, 0.95, step=0.05),

        # Top-k
        'top_k': trial.suggest_int('top_k', 10, 100, step=10),

        # Repetition Penalty
        'repetition_penalty': trial.suggest_float(
            'repetition_penalty', 1.0, 1.5, step=0.1
        ),

        # Length Penalty
        'length_penalty': trial.suggest_float(
            'length_penalty', 0.5, 2.0, step=0.1
        ),

        # Beam Search
        'num_beams': trial.suggest_int('num_beams', 1, 8),

        # 최대/최소 길이
        'max_new_tokens': trial.suggest_int('max_new_tokens', 50, 200, step=10),
        'min_new_tokens': trial.suggest_int('min_new_tokens', 20, 80, step=10),
    }
```

### 4. 데이터 전처리 파라미터
```python
def create_preprocessing_search_space(trial):
    """전처리용 탐색 공간"""
    return {
        # 토큰 길이
        'max_input_length': trial.suggest_categorical(
            'max_input_length', [512, 768, 1024, 1536]
        ),

        # 데이터 증강
        'use_augmentation': trial.suggest_categorical(
            'use_augmentation', [True, False]
        ),

        # 증강 전략
        'augmentation_strategy': trial.suggest_categorical(
            'augmentation_strategy',
            ['none', 'paraphrase', 'backtranslation', 'both']
        ),

        # 노이즈 제거 수준
        'noise_removal_level': trial.suggest_categorical(
            'noise_removal_level',
            ['minimal', 'moderate', 'aggressive']
        ),

        # Person 토큰 처리
        'person_token_strategy': trial.suggest_categorical(
            'person_token_strategy',
            ['keep_original', 'simplify', 'abstract']
        ),
    }
```

## 🎯 목적 함수 구현

### 1. 단일 목적 최적화 (ROUGE 최대화)
```python
def objective_single(trial):
    """단일 목적 함수: ROUGE-F1 최대화"""

    # 하이퍼파라미터 샘플링
    model_params = create_model_search_space(trial)
    train_params = create_training_search_space(trial)
    gen_params = create_generation_search_space(trial)

    # 모델 생성 및 학습
    model = create_model(model_params)
    trainer = create_trainer(model, train_params)

    # 조기 종료 콜백 (Pruning)
    pruning_callback = OptunaPruningCallback(trial, 'eval_rouge')
    trainer.add_callback(pruning_callback)

    # 학습 실행
    trainer.train()

    # 검증 세트 평가
    eval_results = trainer.evaluate(gen_params)

    # ROUGE-F1 반환 (최대화)
    return eval_results['rouge_f1']
```

### 2. 다중 목적 최적화 (성능 vs 효율성)
```python
def objective_multi(trial):
    """다중 목적 함수: ROUGE와 속도 균형"""

    params = sample_all_parameters(trial)

    # 모델 학습 및 평가
    model = train_model(params)
    rouge_score = evaluate_rouge(model)
    inference_time = measure_inference_time(model)
    model_size = get_model_size(model)

    # 다중 목적 반환
    return rouge_score, -inference_time, -model_size
```

### 3. 제약 조건이 있는 최적화
```python
def objective_with_constraints(trial):
    """제약 조건 포함 최적화"""

    params = sample_all_parameters(trial)

    # GPU 메모리 제약 체크
    estimated_memory = estimate_gpu_memory(params)
    if estimated_memory > MAX_GPU_MEMORY:
        raise optuna.TrialPruned()

    # 추론 시간 제약 체크
    if params['num_beams'] > 5 and params['max_new_tokens'] > 150:
        # 너무 느린 설정
        raise optuna.TrialPruned()

    # 정상 학습
    rouge_score = train_and_evaluate(params)

    return rouge_score
```

## 🚀 Optuna Study 실행

### 1. 기본 실행
```python
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Study 생성
study = optuna.create_study(
    study_name="dialogue_summarization_optimization",
    direction="maximize",  # ROUGE 최대화
    sampler=TPESampler(
        n_startup_trials=10,  # 초기 랜덤 탐색
        n_ei_candidates=24,    # EI 후보 수
        seed=42
    ),
    pruner=MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3,
        interval_steps=1
    ),
    storage="sqlite:///optuna_study.db",  # 결과 저장
    load_if_exists=True  # 이어서 실행 가능
)

# 최적화 실행
study.optimize(
    objective_single,
    n_trials=100,           # 시도 횟수
    timeout=7200,           # 2시간 제한
    n_jobs=1,               # 병렬 실행
    gc_after_trial=True,    # 메모리 정리
    show_progress_bar=True
)

# 최적 파라미터
best_params = study.best_params
best_value = study.best_value
print(f"Best ROUGE-F1: {best_value:.4f}")
print(f"Best params: {best_params}")
```

### 2. 병렬 실행 (여러 GPU)
```python
def run_parallel_optimization():
    """여러 GPU에서 병렬 최적화"""

    # GPU별 프로세스 생성
    def optimize_on_gpu(gpu_id):
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        study = optuna.load_study(
            study_name="parallel_optimization",
            storage="mysql://user:password@localhost/optuna"
        )

        study.optimize(
            lambda trial: objective_single(trial, gpu_id=gpu_id),
            n_trials=25  # GPU당 25회
        )

    # 병렬 실행
    from multiprocessing import Process
    processes = []
    for gpu_id in range(4):  # 4개 GPU
        p = Process(target=optimize_on_gpu, args=(gpu_id,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```

## 📊 최적화 전략

### 1. 단계적 최적화 (Staged Optimization)
```python
class StagedOptimization:
    """단계별 최적화 전략"""

    def __init__(self):
        self.stages = [
            ('model_selection', 20),     # 1단계: 모델 선택
            ('learning_rate', 30),        # 2단계: 학습률
            ('generation_params', 30),    # 3단계: 생성 파라미터
            ('fine_tuning', 20)          # 4단계: 미세 조정
        ]

    def optimize_stage(self, stage_name, n_trials):
        """각 단계별 최적화"""
        if stage_name == 'model_selection':
            # 모델과 LoRA 설정만 최적화
            search_space = ['model_name', 'lora_r', 'lora_alpha']
        elif stage_name == 'learning_rate':
            # 학습 파라미터만 최적화
            search_space = ['learning_rate', 'batch_size', 'scheduler_type']
        elif stage_name == 'generation_params':
            # 생성 파라미터만 최적화
            search_space = ['temperature', 'top_p', 'num_beams']
        else:
            # 전체 미세 조정
            search_space = 'all'

        return self.run_optimization(search_space, n_trials)

    def run_full_optimization(self):
        """전체 단계 순차 실행"""
        best_params = {}
        for stage_name, n_trials in self.stages:
            stage_best = self.optimize_stage(stage_name, n_trials)
            best_params.update(stage_best)
        return best_params
```

### 2. 적응적 탐색 (Adaptive Search)
```python
class AdaptiveSearch:
    """성능에 따라 탐색 전략 조정"""

    def __init__(self):
        self.performance_history = []
        self.search_strategy = 'exploration'

    def update_strategy(self, trial_results):
        """최근 성능에 따라 전략 변경"""
        self.performance_history.append(trial_results)

        if len(self.performance_history) >= 10:
            recent_improvement = self.calculate_improvement()

            if recent_improvement < 0.01:  # 개선 정체
                self.search_strategy = 'exploitation'
                self.narrow_search_space()
            else:
                self.search_strategy = 'exploration'
                self.widen_search_space()

    def narrow_search_space(self):
        """탐색 공간 축소 (exploitation)"""
        # 최고 성능 근처만 탐색
        best_params = self.get_best_params()
        return self.create_narrow_space(best_params)

    def widen_search_space(self):
        """탐색 공간 확대 (exploration)"""
        # 더 넓은 범위 탐색
        return self.create_wide_space()
```

## 📈 결과 분석 및 시각화

### 1. 최적화 히스토리
```python
def visualize_optimization_history(study):
    """최적화 과정 시각화"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Optimization History',
            'Parameter Importance',
            'Parallel Coordinate',
            'Slice Plot'
        )
    )

    # 1. 최적화 히스토리
    trials = study.trials
    x = [t.number for t in trials]
    y = [t.value for t in trials if t.value is not None]

    fig.add_trace(
        go.Scatter(x=x, y=y, mode='markers+lines', name='ROUGE Score'),
        row=1, col=1
    )

    # 2. 파라미터 중요도
    importance = optuna.importance.get_param_importances(study)
    fig.add_trace(
        go.Bar(x=list(importance.values()), y=list(importance.keys()),
               orientation='h', name='Importance'),
        row=1, col=2
    )

    # 3. Parallel Coordinate Plot
    fig_parallel = optuna.visualization.plot_parallel_coordinate(study)

    # 4. Slice Plot
    fig_slice = optuna.visualization.plot_slice(study)

    fig.show()
```

### 2. 베스트 Trial 분석
```python
def analyze_best_trials(study, top_k=10):
    """상위 k개 trial 분석"""

    # 상위 trials
    best_trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value else -float('inf'),
        reverse=True
    )[:top_k]

    # 공통 파라미터 추출
    common_params = {}
    for param_name in best_trials[0].params:
        values = [t.params[param_name] for t in best_trials]

        # 범주형
        if isinstance(values[0], str):
            most_common = max(set(values), key=values.count)
            frequency = values.count(most_common) / len(values)
            if frequency > 0.7:  # 70% 이상 동일하면 공통
                common_params[param_name] = most_common

        # 수치형
        else:
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val / mean_val < 0.2:  # CV < 20%면 공통
                common_params[param_name] = mean_val

    print(f"Common parameters in top {top_k} trials:")
    for param, value in common_params.items():
        print(f"  {param}: {value}")

    return common_params
```

## 🎮 실행 명령어 예시

### 빠른 테스트 (10 trials)
```bash
python train.py \
    --mode optuna \
    --optuna_trials 10 \
    --optuna_timeout 1800 \
    --optuna_sampler random
```

### 표준 최적화 (100 trials)
```bash
python train.py \
    --mode optuna \
    --optuna_trials 100 \
    --optuna_timeout 7200 \
    --optuna_sampler tpe \
    --optuna_pruner median
```

### 대규모 최적화 (500 trials, 병렬)
```bash
python train.py \
    --mode optuna \
    --optuna_trials 500 \
    --optuna_n_jobs 4 \
    --optuna_storage mysql://localhost/optuna \
    --optuna_study_name dialogue_optimization
```

### 단계적 최적화
```bash
# 1단계: 모델 선택
python train.py --mode optuna --optuna_stage model_selection

# 2단계: 학습률 최적화
python train.py --mode optuna --optuna_stage learning_rate

# 3단계: 생성 파라미터
python train.py --mode optuna --optuna_stage generation

# 4단계: 전체 미세조정
python train.py --mode optuna --optuna_stage fine_tuning
```

## 💡 최적화 팁

### 1. 효율적인 탐색
- **초기 랜덤 탐색**: 10-20 trials는 랜덤으로 시작
- **Pruning 활용**: 성능이 낮은 trial 조기 중단
- **병렬 실행**: 여러 GPU 활용하여 시간 단축

### 2. NLP 특화 전략
- **모델 먼저**: 모델 선택을 먼저 최적화
- **생성 파라미터 중요**: Temperature, Top-p가 성능에 큰 영향
- **긴 시퀀스 주의**: max_length 증가는 메모리와 시간 급증

### 3. 실용적 접근
- **제약 조건 설정**: 추론 시간, 메모리 제한
- **단계적 최적화**: 전체보다 단계별이 효율적
- **베스트 프랙티스 활용**: 이미 알려진 좋은 범위부터 시작

## 📊 예상 성능 향상

| 최적화 수준 | Trials | 시간 | ROUGE 향상 | 추천 상황 |
|------------|--------|------|-----------|-----------|
| Quick | 10 | 30분 | +1-2% | 빠른 실험 |
| Standard | 100 | 5시간 | +3-5% | 일반 최적화 |
| Thorough | 300 | 15시간 | +5-7% | 충분한 시간 |
| Extensive | 500+ | 24시간+ | +7-10% | 최종 제출용 |

## 🏆 NLP 대화 요약 추천 설정

### 검증된 최적 설정 (실제 데이터 기반)
```python
# KoBART (Encoder-Decoder) - 검증 완료
kobart_optimal = {
    'model_name': 'digit82/kobart-summarization',
    'learning_rate': 5e-5,
    'batch_size': 8,
    'num_train_epochs': 20,
    'encoder_max_len': 512,
    'decoder_max_len': 100,
    'num_beams': 4,
    # Result: ROUGE Sum 94.51
}

# Llama-3.2-Korean (LLM QLoRA) - 최적화 중
llama_optimal = {
    'model_name': 'Bllossom/llama-3.2-Korean-Bllossom-3B',
    'learning_rate': 2e-5,
    'batch_size': 8,
    'gradient_accumulation_steps': 8,  # effective=64
    'num_train_epochs': 3,

    # ⚠️ 중요: Prompt Truncation 방지
    'encoder_max_len': 1024,  # 512 → 1024 (필수!)
    'decoder_max_len': 200,   # 100 → 200

    # LoRA (검증된 값)
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'gate_proj', 'up_proj', 'down_proj'],

    # 학습 안정성
    'warmup_ratio': 0.1,
    'weight_decay': 0.1,
    'lr_scheduler_type': 'cosine',
    'max_grad_norm': 1.2,

    # Mixed Precision
    'bf16': True,  # Llama는 bf16
    'fp16': False,

    # 생성
    'num_beams': 4,
    'max_new_tokens': 150,  # 100 → 150 (여유)
    'repetition_penalty': 1.1,
    'no_repeat_ngram_size': 3,

    # Target: ROUGE Sum 95+
}

# Qwen3-4B (LLM QLoRA) - 대기
qwen_optimal = {
    'model_name': 'Qwen/Qwen3-4B-Instruct-2507',
    'learning_rate': 2e-5,
    'batch_size': 6,  # Qwen은 메모리 더 사용
    'gradient_accumulation_steps': 8,
    'num_train_epochs': 3,

    'encoder_max_len': 1024,
    'decoder_max_len': 200,

    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,

    # Qwen은 fp16
    'fp16': True,
    'bf16': False,

    'num_beams': 4,
    'max_new_tokens': 150,
}
```

### Optuna 탐색 권장 범위
```python
# 추가 최적화 시 탐색 범위
search_ranges = {
    # 학습률 (검증된 범위 중심)
    'learning_rate': [1e-5, 3e-5, 5e-5],

    # 배치 크기 (GPU 메모리 고려)
    'batch_size': [4, 6, 8],  # effective=32~64

    # LoRA rank (성능 vs 속도)
    'lora_r': [8, 16, 32],
    'lora_alpha': [16, 32, 64],

    # 생성 파라미터
    'num_beams': [3, 4, 5],
    'repetition_penalty': [1.0, 1.1, 1.2],
    'max_new_tokens': [100, 150, 200],

    # ⚠️ 고정 권장 (검증된 값)
    'encoder_max_len': 1024,  # 절대 512로 하지 말것!
    'warmup_ratio': 0.1,
    'lr_scheduler_type': 'cosine',
}
```
