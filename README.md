# BloGeeK-Data

BloGeeK-Data is a repository for the parts of the BloGeek project that use machine learning.

| English | Korean |
| :---: | :---: |
| Link | [Link](https://github.com/KEA-GeeK/BloGeeK-Data/blob/main/readme_lang/README_ko.md) |

<br> <br>

## <b> Repository Structure </b>

<br>

* <b> PolarityRecognition </b>
    > Polarity Recognition model determines whether a given sentence or set of sentences is positive, negative, or just neutral.
    * File
        * <b> dataset </b> : Datasets we used to create and test the Polarity Recognition Model
          *  README.md : Links to download the dataset
        * <b> model </b> : Python files to train the model and perform the inference
          * train.py : Python file for training
          * infer.py : Python file for inference
        * README.md : Markdown file that writes out what we need to run Polarity Recognition model
    * Implemented Model : [KoBERT](https://github.com/SKTBrain/KoBERT)
      
* <b> StyleTransfer </b>
    > Style Transfer model creates stylistic variations of a given sentence and was used in this project to increase the amount of data.
    * File
        * <b> dataset </b> : Datasets we used to create and test the Style Transfer Model
          *  README.md : Links to download the dataset
        * <b> model </b> : Python files to train the model and change the style
          * train.py : Python file for training
          * change.py : Python file for change
        * README.md : Markdown file that writes out what we need to run Style Transfer model
    * Implemented Model : [KoBART](https://huggingface.co/gogamza/kobart-base-v2)

* <b> HateDetection </b>
    > Hate Detection model detects whether a given sentence contains expressions that include hate or immorality.
    * File
        * <b> dataset </b> : Datasets we used to create and test the Hate Detection Model
          *  README.md : Links to download the dataset
        * <b> model </b> : Python files to train the model and perform the inference
          * train.py : Python file for training
          * infer.py : Python file for inference
        * README.md : Markdown file that writes out what we need to run Hate Detection model
    * Implemented Model : [KoELECTRA](https://github.com/monologg/KoELECTRA)

* <b> readme_lang </b>
    > README.md files by language
    * File
        * README_ko.md : Korean README.md
* <b> README.md </b>
    > A markdown file containing a description of BloGeeK-Data.

<br> <br>

## <b> Model Implementation </b>

<br>

* Model Structure
    * PolarityRecognition : <br><br>
      ![image](https://github.com/KEA-GeeK/BloGeeK-Data/assets/31691750/b7496a6e-778c-476b-908b-27aad3197151) <br>
      (예시고 정확하게 고칠 예정)

      
    * StyleTransfer :
      (모델 구조 이미지 올릴 예정)


<br>

* How to use
    * PolarityRecognition :
        * Training :
          ```bash
          python train.py --train_data [file_path] --test_data [file_path] --num_epoch [number]
          ```
  
        * infer :
          ```bash
          python infer.py --pt_path [file_path]
          ```

    * StyleTransfer :
        * Training :
          ```bash
          python train.py --data_path [file_path] --output_path [folder_path]
          ```
  
        * Change :
          ```bash
          python change.py --model_path [folder_path] --style [style_name] --sentence [input_sentence]
          ```
            * model_path : The __model_path__ in _change.py_ is the same as the __output_path__ in _train.py_.
            * style : There are the following types of __style__ :
              
              | style_name | Description |
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

* To-do
    * <b> Style Transfer </b>
        * [ ] Proceed with data augmentation[^1]
    
    * <b> Polarity Recognition </b>
        * [ ] Do enough training
        * [ ] Porting to Spark

<br> <br>

[^1]: After being granted access to a server room or assigned a cloud

<br> <br>

## <b> Contributors </b>

<br>

| Name | Student Number | University | Contributed Parts | Github Link |
| :---: | :---: | :---: | :---: | :---: |
| Baek Hyunjung | 201935059 | Gachon Univ. | HateDetection, TopicRecommendation | [Github](https://github.com/Baekhyunjung) |
| Kim Donghyeon | 201935217 | Gachon Univ. | PolarityRecognition, StyleTransfer | [Github](https://github.com/eastlighting1) |
