# dog-breed-classification.tf
TF-Slim을 이용한 개 품종 분류기 학습


## 데이터셋 다운로드
[Standford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)을 다운로드한 후 압축을 푼다.
데이터는 Pascal-VOC 데이터셋 포맷으로 구조화되어 있으며, localization을 위한 bbox 어노테이션 정보가 포함되어 있다.
이 예제에서는 Images/ 디렉토리 하위의 서브 디렉토리명을 분류 레이블명으로 사용한다.


## TFRecord 포맷으로 변환하기
훈련/평가를 위해 이미지 데이터를 TFRecord 포맷으로 변환한다.
TFRecord로 변환하는 유틸리티인 create_tf_record.py는 이미지 데이터가 포함된 디렉토리를 인자로 받는데 아래와 같은 구조로 데이터를 구조화해야 한다.
그리고 이미지는 jpg 포맷 변환만 지원한다.
```buildoutcfg
${DATASET_DIR}/
  |- images/
      |- class-1/
      |- class-2/
      |- class-3/
      |- class-4/
      |-- ...
```
그리고 다운로드한 [Standford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) 이미지 중 n02105855-Shetland_sheepdog/n02105855_2933.jpg 파일은 깨져있으므로 삭제한다. 


```buildoutcfg
export DATASET_DIR=/home/itrocks/Git/Tensorflow/dog-breed-classification.tf/raw_data/dog
python datasets/create_tf_record.py --dataset_name=dog \
                                    --dataset_dir=$DATASET_DIR \
                                    --num_shards=5 \
                                    --ratio_val=0.2
```
* --dataset_name: TFRecord 파일명 prefix
* --dataset_dir: 이미지 데이터를 포함하는 디렉토리
* --num_shards: TFRecord 파일 샤드 개수
* --ratior_val: 평가 데이터셋 생성 비율


## 훈련하기


## 평가하기