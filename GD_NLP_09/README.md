# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 소용현
- 리뷰어 : 남희정


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [x] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
* 두가지 Tokenizer 사용 : PreTrainedTokenizerFast,AutoTokenizer
```python
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name,
                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>',
                                                    model_max_length=512,
                                                     padding_side="right"
    tokenizer = AutoTokenizer.from_pretrained(model_name, bos_token='</s>', eos_token='</s>', unk_token='</s>',
     pad_token='</s>', padding_side="right", model_max_length=512,)
```
* LoRa 적용
```python
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
    peft_config = LoraConfig(
    task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
```
* 모델 선언 및 학습
  ```python
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    trainer.train()
    model.save_pretrained('/aiffel/KoChatGPT/output_1_SFT')
  ```
  ![image](https://github.com/soh-yh/Going_Deeper_NLP/assets/88833290/91e5ec88-e91a-4f1d-af4d-6d3a1b84f486)

- [ ] 2.주석을 보고 작성자의 코드가 이해되었나요?
  네.. 잘 이해가 되었습니다. 
- [ ] 3.코드가 에러를 유발할 가능성이 있나요?
  가능성을 분석할 수 있도록 해 보겠습니다. 
- [ ] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  네. 잘 이해하고 작성되었습니다. 
- [ ] 5.코드가 간결한가요?
  네. 간결하게 잘 작성되었습니다. 

# 예시
1. 코드의 작동 방식을 주석으로 기록합니다.
2. 코드의 작동 방식에 대한 개선 방법을 주석으로 기록합니다.
3. 참고한 링크 및 ChatGPT 프롬프트 명령어가 있다면 주석으로 남겨주세요.
```python
# 사칙 연산 계산기
class calculator:
    # 예) init의 역할과 각 매서드의 의미를 서술
    def __init__(self, first, second):
        self.first = first
        self.second = second
    
    # 예) 덧셈과 연산 작동 방식에 대한 서술
    def add(self):
        result = self.first + self.second
        return result

a = float(input('첫번째 값을 입력하세요.')) 
b = float(input('두번째 값을 입력하세요.')) 
c = calculator(a, b)
print('덧셈', c.add()) 
```

# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
