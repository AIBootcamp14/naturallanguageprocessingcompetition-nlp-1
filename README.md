[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HS6nBbT4)
# 프로젝트 이름

<br>

## 프로젝트 소개
### <프로젝트 소개>
- _이번 프로젝트에 대해 소개를 작성해주세요_

### <작품 소개>
- _만드신 작품에 대해 간단한 소개를 작성해주세요_

<br>

## 팀 구성원
| 프로필 | 이름 (깃허브) | MBTI | 전공/학과 | 담당 역할 |
|:------:|:-------------:|:----:|:---------:|:----------|
| <img src="https://github.com/user-attachments/assets/a24cf78c-2c8f-47b9-b53b-867557872d88" width="100" height="100"> | [김선민](https://github.com/nimnusmik) | ENFJ | 경영&AI 융합 학부 | 팀 리드, 담당 역할 |
| <img src="https://github.com/user-attachments/assets/489d401e-f5f5-4998-91a0-3b0f37f4490f" width="100" height="100"> | [김병현](https://github.com/Bkankim) | ENFP | 정보보안 | 담당 역할 |
| <img src="https://github.com/user-attachments/assets/55180131-9401-457e-a600-312eda87ded9" width="100" height="100"> | [임예슬](https://github.com/joy007fun/joy007fun) | ENTP | 관광경영&컴퓨터공학, 클라우드 인프라 | 담당 역할 |
| <img src="https://github.com/user-attachments/assets/10a2c088-72cb-45cd-8772-b683bc2fb550" width="100" height="100"> | [정서우](https://github.com/Seowoo-C) | INFJ | 화학 | 담당 역할 |
| <img src="" width="100" height="100"> | [정소현](https://github.com/soniajhung) | MBTI | 전공 | 담당 역할 |
| <img src="https://github.com/user-attachments/assets/5c04a858-46ed-4043-9762-b7eaf7b1149a" width="100" height="100"> | [최현화](https://github.com/iejob) | ISTP | 컴퓨터공학 | 담당 역할, Git 브랜치·병합·충돌 관리 |

<br>

### 📦 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd natural-language-processing-competition

# Python 환경 (pyenv 권장)
pyenv install 3.11.9
pyenv virtualenv 3.11.9 nlp_py3_11_9
pyenv activate nlp_py3_11_9
pip install -r requirements.txt
```

### 📁 2. 데이터 준비

```bash
# 데이터 구조 확인
data/raw/
├── dev.csv                 # 
├── sample_submission.csv   # 제출 형식
├── test.csv                # 
└── train.csv               # 학습
```

<br>

## 2. 프로젝트 구조
```markdown
natural-language-processing-competition # 최상위 폴더
├── configs                             # yaml 등 설정 파일 경로  
├── data  
│   └── raw                             # 데이터 다운로드 및 압축 해제한 원시 데이터 (예: data.tar.gz 해제 결과)  
├── docs                                # 문서 관련 (보고서, 노트 등)  
├── experiments                         # 모듈화 실행 시 실험 결과 및 체크포인트 저장 경로  
├── notebooks  
│   ├── base                            # 대회에서 제공한 베이스라인 노트북/코드  
│   └── team  
│       └── 이니셜                       # 개인/팀 노트북 저장 경로 (예시: CHH)  
│           ├── config                  # 노트북에서 사용하는 설정 파일  
│           ├── src                     # 노트북에서 참조하는 파이썬 모듈/스크립트  
│           ├── logs                    # 노트북 실행 시 생성되는 로그 파일  
│           └── submissions             # 노트북으로 만든 제출 파일 저장 경로  
├── src                                 # 모듈화된 파이썬 코드(패키지/모듈)  
├── logs                                # 모듈화 실행 시 저장되는 로그 (날짜별 폴더 권장)
│   └── 20250926
└── submissions                         # 제출 파일 저장 (날짜별 폴더 권장)  
    └── 20250926
```

### 원본 링크
- 데이터: https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000365/data/data.tar.gz  
- 베이스라인 코드(노트북): https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000365/data/code.tar.gz

<br>

## 3. 구현 기능
### 기능1
- _작품에 대한 주요 기능을 작성해주세요_
### 기능2
- _작품에 대한 주요 기능을 작성해주세요_
### 기능3
- _작품에 대한 주요 기능을 작성해주세요_

<br>

## 4. 작품 아키텍처(필수X)
- #### _아래 이미지는 예시입니다_
![이미지 설명](https://www.cadgraphics.co.kr/UPLOAD/editor/2024/07/04//2024726410gH04SyxMo3_editor_image.png)

<br>

## 5. 트러블 슈팅
### 1. OOO 에러 발견

#### 설명
- _프로젝트 진행 중 발생한 트러블에 대해 작성해주세요_

#### 해결
- _프로젝트 진행 중 발생한 트러블 해결방법 대해 작성해주세요_

<br>

## 6. 프로젝트 회고
### 박패캠
- _프로젝트 회고를 작성해주세요_

<br>

## 7. 참고자료
- _참고자료를 첨부해주세요_
