# nia_ventilator_Waening_Success_Prediction

* 2022년 인공지능 학습용 데이터 구축 사업의 일환으로, 인공호흡기 데이터와 유관데이터(임상데이터)를 활용하여 인공호흡기 이탈 성공여부를 예측
* 인공호흡을 위한 기관삽관은 한번 제거하면 다시 삽관하는 것이 어렵고, 환자의 상태가 다시 좋아지지 않을 경우는 재삽관을 해야할 뿐만 아니라, 환자의 예후가 나빠질 수 있음.
* 따라서, 기관삽관을 제거하는 것은 환자의 회복과 직결될 뿐만 아니라 여러 의료자원이 소모되므로 시기적절한 인공호흡기 이탈 시점을 예측하는 것은 임상적으로 매우 중요함.
* 인공호흡기 이탈 성공(Weaning Success)여부를 예측은 환자, 의료인 모두에게 임상적으로 의미가 있는 예측이라고 할 수 있음.


----


# 1. Data Preprocessing
    python preprocess.py

##### 기관별 Ventilator_Parameter, 유관데이터(임상데이터)를 하나의 dataset으로 통합하는 전처리 코드
--

# 2. Train
    python train.py 

##### 전처리가 완료된 데이터셋과 라벨 파일을 입력받아 훈련셋(training set)과 검증셋(validation set)으로 나누어 Split 한 후, Random-sampling 하여 모델을 학습함.
--

# 3. Test
    python test.py

##### 훈련에 사용되지 않은 테스트셋으로 훈련된 모델을 사용하여 예측하고, 성능지표를 활용하여 평가함.