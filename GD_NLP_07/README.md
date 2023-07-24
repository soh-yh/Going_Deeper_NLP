# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 소용현
- 리뷰어 : 김경훈


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.

- [x] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

네, 정상적으로 동작하고 주어진 문제를 해결했습니다.

>|번호|평가문항|상세기준|평가결과|
>|:---:|---|---|:---:|
>|1|한글 코퍼스를 가공하여 BERT pretrain용 데이터셋을 잘 생성하였다.|MLM, NSP task의 특징이 잘 반영된 pretrain용 데이터셋 생성과정이 체계적으로 진행되었다.|⭐|
>|2|구현한 BERT 모델의 학습이 안정적으로 진행됨을 확인하였다.|학습진행 과정 중에 MLM, NSP loss의 안정적인 감소가 확인되었다.|⭐|
>|3|1M짜리 mini BERT 모델의 제작과 학습이 정상적으로 진행되었다.|학습된 모델 및 학습과정의 시각화 내역이 제출되었다.|⭐|


- [x] 2.주석을 보고 작성자의 코드가 이해되었나요?

네, 주석이 잘 정리되어 있습니다.

``` python

class CosineSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    CosineSchedule Class
    """
    def __init__(self, train_steps=4000, warmup_steps=2000, max_lr=2.5e-4):
        """
        생성자
        :param train_steps: 학습 step 총 합
        :param warmup_steps: warmup steps
        :param max_lr: 최대 learning rate
        """
        super().__init__()

        assert 0 < warmup_steps < train_steps
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps
        self.max_lr = max_lr

    def __call__(self, step_num):
        """
        learning rate 계산
        :param step_num: 현재 step number
        :retrun: 계산된 learning rate
        """
        state = tf.cast(step_num <= self.warmup_steps, tf.float32)
        lr1 = tf.cast(step_num, tf.float32) / self.warmup_steps
        progress = tf.cast(step_num - self.warmup_steps, tf.float32) / max(1, self.train_steps - self.warmup_steps)
        lr2 = 0.5 * (1.0 + tf.math.cos(math.pi * progress))
        return (state * lr1 + (1 - state) * lr2) * self.max_lr
```

- [ ] 3.코드가 에러를 유발할 가능성이 있나요?

에러를 유발할 가능성이 낮은것 같습니다.

- [x] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?

네, 제대로 이해하고 작성했습니다.

- [x] 5.코드가 간결한가요?

네, 코드가 간결합니다.

