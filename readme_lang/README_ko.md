# BloGeeK-Data

BloGeeK-Data는 BloGeek 프로젝트에서 머신러닝을 사용하는 부분을 담은 레포지터리입니다.

| 영어 | 한국어 |
| :---: | :---: |
| [Link](https://github.com/KEA-GeeK/BloGeeK-Data/blob/main/README.md) | Link |


<br> <br>

## <b> 레포지터리 구조 </b>

<br>

* <b> 극성 인식 </b>
    > 극성 인식 모델은 주어진 문장 또는 문장들이 긍정적인지, 부정적인지, 혹은 중립적인지 파악합니다.
    * 파일
        * <b> dataset </b> : 극성 인식 모델을 만들고 테스트하기 위해 사용되는 데이터셋
          *  README.md : 데이터셋을 다운로드하기 위한 링크들
        * <b> model </b> : 모델을 학습하고 극성을 추론하기 위한 파이썬 파일들
          * train.py : 모델 학습을 위한 파이썬 파일
          * infer.py : 극성 추론을 위한 파이썬 파일
        * README.md : 극성 인식 모델을 실행하기 위해 필요한 정보들을 담은 마크다운 파일
    * 구현에 사용한 모델 : [KoBERT](https://github.com/SKTBrain/KoBERT)
      
* <b> 문체 변경 </b>
    > 문제 변경 모델은 주어진 문장을 다양한 문체로 변경하며, 이 프로젝트에서는 데이터의 양을 늘리기 위해 사용하였습니다.
    * 파일
        * <b> dataset </b> : 문체 변경 모델을 만들고 테스트하기 위해 사용되는 데이터셋
          *  README.md : 데이터셋을 다운로드하기 위한 링크
        * <b> model </b> : 모델을 학습하고 문체를 변경하기 위한 파이썬 파일들
          * train.py : 모델 학습을 위한 파이썬 파일
          * change.py : 문체 변경을 위한 파이썬 파일
        * README.md : 문체 변경 모델을 실행하기 위해 필요한 정보들을 담은 마크다운 파일
    * 구현에 사용한 모델 : [KoBART](https://huggingface.co/gogamza/kobart-base-v2)
* <b> readme_lang </b>
    > 언어별 README.md 파일들
    * 파일
        * README_ko.md : 한국어 README.md
* <b> README.md </b>
    > BloGeek-Data에 대한 설명이 담긴 마크다운 파일
  
<br> <br>

## <b> 모델 구현 </b>

<br>

* 모델 구조
    * 극성 인식 : <br><br>
      ![image](https://github.com/KEA-GeeK/BloGeeK-Data/assets/31691750/b7496a6e-778c-476b-908b-27aad3197151) <br>
      (예시고 정확하게 고칠 예정)

      
    * 문체 변경 :
      (모델 구조 이미지 올릴 예정)


<br>

* 사용하는 법
    * 극성 인식 :
        * 학습 :
          ```bash
          python train.py --train_data [파일_위치] --test_data [파일_위치] --num_epoch [숫자]
          ```
  
        * 추론 :
          ```bash
          python infer.py --pt_path [파일_위치]
          ```

    * 문체 변경 :
        * 학습 :
          ```bash
          python train.py --data_path [파일_위치] --output_path [폴더_위치]
          ```
  
        * 변경 :
          ```bash
          python change.py --model_path [폴더_위치] --style [스타일_이름] --sentence [입력_문장]
          ```
            * model_path : _change.py_ 의 __폴더_위치__ 는 _train.py_ 의 __폴더_위치__ 와 동일합니다. 
            * style : __style__ 의 종류는 다음과 같습니다. :
              
              | 스타일_이름 | 설명 |
              |:---:| :---: |
              | formal       | 문어체       |
              | informal     | 구어체       |
              | android      | 안드로이드   |
              | azae         | 아재         |
              | chat         | 채팅         |
              | choding      | 초등학생     |
              | emoticon     | 이모티콘     |
              | enfp         | enfp         |
              | gentle       | 신사         |
              | halbae       | 할아버지     |
              | halmae       | 할머니       |
              | joongding    | 중학생       |
              | king         | 왕           |
              | naruto       | 나루토       |
              | seonbi       | 선비         |
              | sosim        | 소심한       |
              | translator   | 번역기       |
    

<br>

* 할 일
    * <b> 문체 변경 </b>
        * [ ] 데이터 증강시키기[^1]
    
    * <b> 극성 인식 </b>
        * [ ] 충분히 학습시키기
        * [ ] PySpark로 이식하기

<br> <br>

[^1]: 메타버스 인큐베이터 서버룸 사용 허락이 나거나 클라우드를 할당받은 후 진행

<br> <br>

## <b> 기여자 </b>

<br>

| 이름 | 학번 | 대학교 | 기여한 부분 | 깃허브 링크 |
| :---: | :---: | :---: | :---: | :---: |
| 백현정 | 201935059 | 가천대학교 | ? | [깃허브](https://github.com/Baekhyunjung) |
| 김동현 | 201935217 | 가천대학교 | 극성 인식, 문체 변경 | [깃허브](https://github.com/eastlighting1) |
