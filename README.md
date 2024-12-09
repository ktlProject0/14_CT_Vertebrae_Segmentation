# KTL_project_14_CT_Spine_Segmentation

## Data Description

1. 학습용 데이터 (/Data/Training/...)
   - 원본 CT 영상 (/Data/Training/rawdata/sub-verse###/CT.nii)
   - 척추 분할 마스크 (/Data/Training/derivatives/sub-verse###/msk.nii.gz)
3. 성능 평가 데이터 (/Data/Test/...)
   - 원본 CT 영상 (/Data/Test/rawdata/sub-verse###/CT.nii)
   - 척추 분할 마스크 (/Data/Test/derivatives/sub-verse###/msk.nii.gz)
## Prerequisites
Before you begin, ensure you have met the following requirements:
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- Other dependencies can be installed using `environment.yml`
  
## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/ktlProject0/KTL_project_14_CT_Spine_Segmentation.git
cd KTL_project_14_CT_Spine_Segmentation
```
 - You can create a new Conda environment using `conda env create -f environment.yml`.
   
## Code Description
## Training.ipynb
  - 네트워크 학습 코드
## Evaluation.ipynb
  - 네트워크 성능 평가 및 척추 분할 결과 가시화
## model_torch.py
  - Efficient UNet 아키텍쳐 빌드
## Preprocessing.py
  - Intensity windowing
  - 입력 이미지 해상도 조정
  - CT영상을 Axial plane에서 Sagittal plane으로 변경
  - 28번 척추를 포함하는 데이터 제외 (데이터 불균형 문제)
## _utils_torch.py
  - 네트워크를 구성에 필요한 부속 코드
  - Preprocessing 이후 데이터를 모델에 입력하기 위한 처리
## modules.torch.py
  - 네트워크를 구성에 필요한 부속 코드
## loss.py
  - 학습용 loss 함수 코드
  - Dice loss
  - Focal loss
