# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 소용현
- 리뷰어 : 김용석

# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- [o] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
  -> 네, 코드가 정상적으로 동작하고 있으며 주어진 문제를 잘 해결했습니다. 
        주요 코드는 기존 노드에서 가져왔지만 필요에 맞게 코드를 변경하여 적용한 부분이 인상적이었습니다. 

- [o] 2.주석을 보고 작성자의 코드가 이해되었나요?
  -> 주석은 따로 없었지만, 구두로 설명해주셔서 내용이 이해가 되었습니다. 

- [x] 3.코드가 에러를 유발할 가능성이 있나요?
  -> 처음부터 끝까지 코드가 제대로 돌아갔으며, 유발할 가능성이 있는 부분도 작성자가 제대로 고쳐서 수행을 했다고 들었습니다. 

- [o] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?
  -> 작성가 코드를 제대로 이해하고 있었습니다. 

- [o] 5.코드가 간결한가요?
  -> 네, 코드가 간결하게 작성되었습니다. 



# 예시

# 참고 링크 및 코드 개선

■ 형태소 분석기 부분을 수정하여 평가부분에 맞춰주는 작업에 공수가 많이 들어갔지만, 깔끔하게 잘 정리된 코드였습니다. :) 

def translate(tokens, model, src_tokenizer, tgt_tokenizer):
    padded_tokens = tf.keras.preprocessing.sequence.pad_sequences([tokens],
                                                           maxlen=MAX_LEN,
                                                           padding='post')
    ids = []
    output = tf.expand_dims([word_to_index['<start>']], 0)   
    for i in range(MAX_LEN):
        enc_padding_mask, combined_mask, dec_padding_mask = \
        generate_masks(padded_tokens, output)

        predictions, _, _, _ = model(padded_tokens, 
                                      output,
                                      enc_padding_mask,
                                      combined_mask,
                                      dec_padding_mask)

        predicted_id = \
        tf.argmax(tf.math.softmax(predictions, axis=-1)[0, -1]).numpy().item()

        if word_to_index['<end>'] == predicted_id:
#             result = tgt_tokenizer.decode_ids(ids)  
            print(ids)
            print([word_to_index['<start>']])
            result = ids
            return result

        ids.append(predicted_id)
        output = tf.concat([output, tf.expand_dims([predicted_id], 0)], axis=-1)
#     result = tgt_tokenizer.decode_ids(ids)  
    result = ids  
    return result

```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
