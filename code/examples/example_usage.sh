#!/bin/bash
# 프레임워크 사용 예제 스크립트
# 주의: 실제 학습은 실행하지 말고, 구조 확인용으로만 사용

set -e

# 색상 출력
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}KoBART 모듈화 프레임워크 예제${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# 현재 디렉토리 확인
if [[ ! -f "train.py" ]]; then
    echo "오류: train.py를 찾을 수 없습니다."
    echo "code/ 디렉토리에서 실행하세요."
    exit 1
fi

echo -e "${GREEN}1. Config 검증${NC}"
echo "실험 목록:"
grep "^  exp" config/experiments.yaml | head -5
echo ""

echo -e "${GREEN}2. 학습 예제 (실행 안 함)${NC}"
echo "명령어: python train.py --experiment exp7a"
echo "설명: Exp #7-A 학습 (가중치 없음, 자연 분포)"
echo ""

echo -e "${GREEN}3. 추론 예제 (실행 안 함)${NC}"
echo "명령어: python inference.py --experiment exp7a --checkpoint checkpoint-2068"
echo "설명: Exp #7-A 추론, best checkpoint 사용"
echo ""

echo -e "${GREEN}4. 새 실험 추가 예제${NC}"
echo "단계 1: config/experiments.yaml에 exp8 추가"
echo "단계 2: python train.py --experiment exp8"
echo "단계 3: python inference.py --experiment exp8 --checkpoint checkpoint-XXXX"
echo ""

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}상세 문서: README_framework.md${NC}"
echo -e "${BLUE}빠른 시작: FRAMEWORK_QUICKSTART.md${NC}"
echo -e "${BLUE}======================================${NC}"
