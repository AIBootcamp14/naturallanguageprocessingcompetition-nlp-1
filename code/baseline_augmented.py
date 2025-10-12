#!/usr/bin/env python
# coding: utf-8

# # **💁🏻🗨️💁🏻‍♂️대화 요약 Baseline code**
# > **Dialogue Summarization** 경진대회에 오신 여러분 환영합니다! 🎉    
# > 본 대회에서는 최소 2명에서 최대 7명이 등장하여 나누는 대화를 요약하는 BART 기반 모델의 baseline code를 제공합니다.     
# > 주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 만들어봅시다!

# ## ⚙️ 데이터 및 환경설정

# ### 1) 필요한 라이브러리 설치

# - 필요한 라이브러리를 설치한 후 불러옵니다.

# In[61]:


import pandas as pd
import os
import re
import json
import yaml
from glob import glob
from tqdm import tqdm
from pprint import pprint
import torch
import pytorch_lightning as pl
from rouge import Rouge # 모델의 성능을 평가하기 위한 라이브러리입니다.

from torch.utils.data import Dataset , DataLoader
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig, PreTrainedTokenizerFast
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback

import wandb # 모델 학습 과정을 손쉽게 Tracking하고, 시각화할 수 있는 라이브러리입니다.


# In[62]:


from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경변수 가져오기
WANDB_ENTITY = os.getenv('WANDB_ENTITY')
WANDB_PROJECT = os.getenv('WANDB_PROJECT')
WANDB_NAME = os.getenv('WANDB_NAME', 'kobart-augmented-v1')  # 기본값 설정

# 환경변수 확인 (민감 정보는 마스킹하여 출력)
print(f"WANDB_ENTITY: {os.getenv('WANDB_ENTITY')}")
print(f"WANDB_PROJECT: {os.getenv('WANDB_PROJECT')}")
print(f"WANDB_API_KEY: {'*' * 20 if os.getenv('WANDB_API_KEY') else 'Not set'}")
print(f"HF_TOKEN: {'*' * 20 if os.getenv('HF_TOKEN') else 'Not set'}")
print(f"SOLAR_API_KEY: {'*' * 20 if os.getenv('SOLAR_API_KEY') else 'Not set'}")


# ### 2) Config file 만들기 (선택)
# - 모델 생성에 필요한 다양한 매개변수 정보를 저장할 수 있습니다.  
#   따라서, 코드 상에서 모델의 매개변수를 설정할 수도 있지만 독립적인 매개변수 정보 파일을 생성하여 관리할 수 있습니다.

# In[63]:


# config 설정에 tokenizer 모듈이 사용되므로 미리 tokenizer를 정의해줍니다.
tokenizer = AutoTokenizer.from_pretrained("digit82/kobart-summarization")


# In[64]:


config_data = {
    "general": {
        "data_path": "/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/data", # 모델 생성에 필요한 데이터 경로를 사용자 환경에 맞게 지정합니다.
        "model_name": "digit82/kobart-summarization", # 불러올 모델의 이름을 사용자 환경에 맞게 지정할 수 있습니다.
        "output_dir": "/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/submission" # 모델의 최종 출력 값을 저장할 경로를 설정합니다.
    },
    "tokenizer": {
        "encoder_max_len": 512,
        "decoder_max_len": 100,
        "bos_token": f"{tokenizer.bos_token}",
        "eos_token": f"{tokenizer.eos_token}",
        # 특정 단어들이 분해되어 tokenization이 수행되지 않도록 special_tokens을 지정해줍니다.
        "special_tokens": ['#Person1#', '#Person2#', '#Person3#', '#PhoneNumber#', '#Address#', '#PassportNumber#']
    },
    "training": {
        "overwrite_output_dir": True,
        "num_train_epochs": 20,
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 50,
        "per_device_eval_batch_size": 32,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "lr_scheduler_type": 'cosine',
        "optim": 'adamw_torch',
        "gradient_accumulation_steps": 1,
        "evaluation_strategy": 'epoch',
        "save_strategy": 'epoch',
        "save_total_limit": 5,
        "fp16": True,
        "load_best_model_at_end": True,
        "seed": 42,
        "logging_dir": "./logs",
        "logging_strategy": "epoch",
        "predict_with_generate": True,
        "generation_max_length": 100,
        "do_train": True,
        "do_eval": True,
        "early_stopping_patience": 3,
        "early_stopping_threshold": 0.001,
        "report_to": "wandb" # (선택) wandb를 사용할 때 설정합니다.
    },
    # (선택) wandb 홈페이지에 가입하여 얻은 정보를 기반으로 작성합니다.
    "wandb": {
        "entity": WANDB_ENTITY,
        "project": WANDB_PROJECT,
        "name": WANDB_NAME
    },
    "inference": {
        "ckt_path": "model ckt path", # 사전 학습이 진행된 모델의 checkpoint를 저장할 경로를 설정합니다.
        "result_path": "./prediction/",
        "no_repeat_ngram_size": 2,
        "early_stopping": True,
        "generate_max_length": 100,
        "num_beams": 4,
        "batch_size" : 32,
        # 정확한 모델 평가를 위해 제거할 불필요한 생성 토큰들을 정의합니다.
        "remove_tokens": ['<usr>', f"{tokenizer.bos_token}", f"{tokenizer.eos_token}", f"{tokenizer.pad_token}"]
    }
}


# - 참고✅    
# : wandb 라이브러리를 사용하기 위해선 entity, project, name를 지정해주어야 합니다. wandb 홈페이지에 가입한 후 얻은 정보를 입력하여 작동할 수 있습니다.

# In[65]:


# 모델의 구성 정보를 YAML 파일로 저장합니다.
config_path = "./config.yaml"
with open(config_path, "w") as file:
    yaml.dump(config_data, file, allow_unicode=True)


# ### 3) Configuration 불러오기

# In[66]:


# 저장된 config 파일을 불러옵니다.
config_path = "./config.yaml"

with open(config_path, "r") as file:
    loaded_config = yaml.safe_load(file)

# 불러온 config 파일의 전체 내용을 확인합니다.
pprint(loaded_config)


# In[67]:


# 실험에 쓰일 데이터의 경로, 사용될 모델, 모델의 최종 출력 결과를 저장할 경로에 대해 확인합니다.
loaded_config['general']


# In[68]:


# # 이곳에 사용자가 저장한 데이터 dir 설정하기
loaded_config['general']['data_path'] = "/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/data"


# In[69]:


# 데이터 전처리를 하기 위해 tokenization 과정에서 필요한 정보들을 확인합니다.
loaded_config['tokenizer']


# In[70]:


# 모델이 훈련 시 적용될 매개변수를 확인합니다.
loaded_config['training']


# In[71]:


# 모델 학습 과정에 대한 정보를 제공해주는 wandb 설정 내용을 확인합니다.
loaded_config['wandb']


# In[72]:


# # (선택) 이곳에 사용자가 사용할 wandb config 설정
# loaded_config['wandb']['entity'] = "사용할 wandb repo name"
# loaded_config['wandb']['name'] = "사용할 wandb run의 name"
# loaded_config['wandb']['project'] = "사용할 wandb project name"


# In[73]:


# 모델이 최종 결과를 출력하기 위한 매개변수 정보를 확인합니다.
loaded_config['inference']


# ### 4) 데이터 불러와서 확인해보기
# - 실험에서 쓰일 데이터를 load하여 데이터의 구조와 내용을 살펴보겠습니다.
# - Train, dev, test 순서대로 12457, 499, 250개 씩 데이터가 구성되어 있습니다.

# In[74]:


# config에 저장된 데이터 경로를 통해 train과 validation data를 불러옵니다.
data_path = loaded_config['general']['data_path']


# In[75]:


# train data의 구조와 내용을 확인합니다.
train_df = pd.read_csv(os.path.join(data_path,'train.csv'))
train_df.tail()


# In[76]:


# validation data의 구조와 내용을 확인합니다.
val_df = pd.read_csv(os.path.join(data_path,'dev.csv'))
val_df.tail()


# ## 1. 데이터 가공 및 데이터셋 클래스 구축
# - csv file 을 불러와서 encoder 와 decoder의 입력형태로 가공해줍니다.
# - 가공된 데이터를 torch dataset class 로 구축하여 모델에 입력가능한 형태로 만듭니다.

# In[77]:


# 데이터 전처리를 위한 클래스로, 데이터셋을 데이터프레임으로 변환하고 인코더와 디코더의 입력을 생성합니다.
class Preprocess:
    def __init__(self,
            bos_token: str,
            eos_token: str,
        ) -> None:

        self.bos_token = bos_token
        self.eos_token = eos_token

    @staticmethod
    # 실험에 필요한 컬럼을 가져옵니다.
    def make_set_as_df(file_path, is_train = True):
        if is_train:
            df = pd.read_csv(file_path)
            train_df = df[['fname','dialogue','summary']]
            return train_df
        else:
            df = pd.read_csv(file_path)
            test_df = df[['fname','dialogue']]
            return test_df

    # BART 모델의 입력, 출력 형태를 맞추기 위해 전처리를 진행합니다.
    def make_input(self, dataset,is_test = False):
        if is_test:
            encoder_input = dataset['dialogue']
            decoder_input = [self.bos_token] * len(dataset['dialogue'])
            return encoder_input.tolist(), list(decoder_input)
        else:
            encoder_input = dataset['dialogue']
            decoder_input = dataset['summary'].apply(lambda x : self.bos_token + str(x)) # Ground truth를 디코더의 input으로 사용하여 학습합니다.
            decoder_output = dataset['summary'].apply(lambda x : str(x) + self.eos_token)
            return encoder_input.tolist(), decoder_input.tolist(), decoder_output.tolist()


# In[78]:


# Train에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForTrain(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item[input_ids], item[attention_mask]
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2[input_ids], item2[attention_mask]
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2) #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask]
        item['labels'] = self.labels['input_ids'][idx] #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask], item[labels]
        return item

    def __len__(self):
        return self.len

# Validation에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForVal(Dataset):
    def __init__(self, encoder_input, decoder_input, labels, len):
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.labels = labels
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()} # item[input_ids], item[attention_mask]
        item2 = {key: val[idx].clone().detach() for key, val in self.decoder_input.items()} # item2[input_ids], item2[attention_mask]
        item2['decoder_input_ids'] = item2['input_ids']
        item2['decoder_attention_mask'] = item2['attention_mask']
        item2.pop('input_ids')
        item2.pop('attention_mask')
        item.update(item2) #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask]
        item['labels'] = self.labels['input_ids'][idx] #item[input_ids], item[attention_mask] item[decoder_input_ids], item[decoder_attention_mask], item[labels]
        return item

    def __len__(self):
        return self.len

# Test에 사용되는 Dataset 클래스를 정의합니다.
class DatasetForInference(Dataset):
    def __init__(self, encoder_input, test_id, len):
        self.encoder_input = encoder_input
        self.test_id = test_id
        self.len = len

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encoder_input.items()}
        item['ID'] = self.test_id[idx]
        return item

    def __len__(self):
        return self.len


# In[79]:


# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
def prepare_train_dataset(config, preprocessor, data_path, tokenizer):
    train_file_path = os.path.join(data_path,'train_with_augmentation.csv')
    val_file_path = os.path.join(data_path,'dev.csv')

    # train, validation에 대해 각각 데이터프레임을 구축합니다.
    train_data = preprocessor.make_set_as_df(train_file_path)
    val_data = preprocessor.make_set_as_df(val_file_path)

    print('-'*150)
    print(f'train_data:\n {train_data["dialogue"][0]}')
    print(f'train_label:\n {train_data["summary"][0]}')

    print('-'*150)
    print(f'val_data:\n {val_data["dialogue"][0]}')
    print(f'val_label:\n {val_data["summary"][0]}')

    encoder_input_train , decoder_input_train, decoder_output_train = preprocessor.make_input(train_data)
    encoder_input_val , decoder_input_val, decoder_output_val = preprocessor.make_input(val_data)
    print('-'*10, 'Load data complete', '-'*10,)

    tokenized_encoder_inputs = tokenizer(encoder_input_train, return_tensors="pt", padding=True,
                            add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_inputs = tokenizer(decoder_input_train, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    tokenized_decoder_ouputs = tokenizer(decoder_output_train, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)

    train_inputs_dataset = DatasetForTrain(tokenized_encoder_inputs, tokenized_decoder_inputs, tokenized_decoder_ouputs,len(encoder_input_train))

    val_tokenized_encoder_inputs = tokenizer(encoder_input_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_inputs = tokenizer(decoder_input_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)
    val_tokenized_decoder_ouputs = tokenizer(decoder_output_val, return_tensors="pt", padding=True,
                        add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False)

    val_inputs_dataset = DatasetForVal(val_tokenized_encoder_inputs, val_tokenized_decoder_inputs, val_tokenized_decoder_ouputs,len(encoder_input_val))

    print('-'*10, 'Make dataset complete', '-'*10,)
    return train_inputs_dataset, val_inputs_dataset


# ## 2. Trainer 및 Trainingargs 구축하기
# - Huggingface 의 Trainer 와 Training arguments를 활용하여 모델 학습을 일괄적으로 처리해주는 클래스를 정의합니다.

# In[80]:


# 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.
def compute_metrics(config,tokenizer,pred):
    rouge = Rouge()
    predictions = pred.predictions
    labels = pred.label_ids

    predictions[predictions == -100] = tokenizer.pad_token_id
    labels[labels == -100] = tokenizer.pad_token_id

    decoded_preds = tokenizer.batch_decode(predictions, clean_up_tokenization_spaces=True)
    labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces=True)

    # 정확한 평가를 위해 미리 정의된 불필요한 생성토큰들을 제거합니다.
    replaced_predictions = decoded_preds.copy()
    replaced_labels = labels.copy()
    remove_tokens = config['inference']['remove_tokens']
    for token in remove_tokens:
        replaced_predictions = [sentence.replace(token," ") for sentence in replaced_predictions]
        replaced_labels = [sentence.replace(token," ") for sentence in replaced_labels]

    print('-'*150)
    print(f"PRED: {replaced_predictions[0]}")
    print(f"GOLD: {replaced_labels[0]}")
    print('-'*150)
    print(f"PRED: {replaced_predictions[1]}")
    print(f"GOLD: {replaced_labels[1]}")
    print('-'*150)
    print(f"PRED: {replaced_predictions[2]}")
    print(f"GOLD: {replaced_labels[2]}")

    # 최종적인 ROUGE 점수를 계산합니다.
    results = rouge.get_scores(replaced_predictions, replaced_labels,avg=True)

    # ROUGE 점수 중 F-1 score를 통해 평가합니다.
    result = {key: value["f"] for key, value in results.items()}
    return result


# In[81]:


# 학습을 위한 trainer 클래스와 매개변수를 정의합니다.
def load_trainer_for_train(config,generate_model,tokenizer,train_inputs_dataset,val_inputs_dataset):
    print('-'*10, 'Make training arguments', '-'*10,)
    # set training args
    training_args = Seq2SeqTrainingArguments(
                output_dir=config['general']['output_dir'], # model output directory
                overwrite_output_dir=config['training']['overwrite_output_dir'],
                num_train_epochs=config['training']['num_train_epochs'],  # total number of training epochs
                learning_rate=config['training']['learning_rate'], # learning_rate
                per_device_train_batch_size=config['training']['per_device_train_batch_size'], # batch size per device during training
                per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],# batch size for evaluation
                warmup_ratio=config['training']['warmup_ratio'],  # number of warmup steps for learning rate scheduler
                weight_decay=config['training']['weight_decay'],  # strength of weight decay
                lr_scheduler_type=config['training']['lr_scheduler_type'],
                optim =config['training']['optim'],
                gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
                evaluation_strategy=config['training']['evaluation_strategy'], # evaluation strategy to adopt during training
                save_strategy =config['training']['save_strategy'],
                save_total_limit=config['training']['save_total_limit'], # number of total save model.
                fp16=config['training']['fp16'],
                load_best_model_at_end=config['training']['load_best_model_at_end'], # 최종적으로 가장 높은 점수 저장
                seed=config['training']['seed'],
                logging_dir=config['training']['logging_dir'], # directory for storing logs
                logging_strategy=config['training']['logging_strategy'],
                predict_with_generate=config['training']['predict_with_generate'], #To use BLEU or ROUGE score
                generation_max_length=config['training']['generation_max_length'],
                do_train=config['training']['do_train'],
                do_eval=config['training']['do_eval'],
                report_to=config['training']['report_to'] # (선택) wandb를 사용할 때 설정합니다.
            )

    # # (선택) 모델의 학습 과정을 추적하는 wandb를 사용하기 위해 초기화 해줍니다.
    # wandb.init(
    #     entity=config['wandb']['entity'],
    #     project=config['wandb']['project'],
    #     name=config['wandb']['name'],
    # )

    # # (선택) 모델 checkpoint를 wandb에 저장하도록 환경 변수를 설정합니다.
    # os.environ["WANDB_LOG_MODEL"]="true"
    # os.environ["WANDB_WATCH"]="false"

    # Validation loss가 더 이상 개선되지 않을 때 학습을 중단시키는 EarlyStopping 기능을 사용합니다.
    MyCallback = EarlyStoppingCallback(
        early_stopping_patience=config['training']['early_stopping_patience'],
        early_stopping_threshold=config['training']['early_stopping_threshold']
    )
    print('-'*10, 'Make training arguments complete', '-'*10,)
    print('-'*10, 'Make trainer', '-'*10,)

    # Trainer 클래스를 정의합니다.
    trainer = Seq2SeqTrainer(
        model=generate_model, # 사용자가 사전 학습하기 위해 사용할 모델을 입력합니다.
        args=training_args,
        train_dataset=train_inputs_dataset,
        eval_dataset=val_inputs_dataset,
        compute_metrics = lambda pred: compute_metrics(config,tokenizer, pred),
        callbacks = [MyCallback]
    )
    print('-'*10, 'Make trainer complete', '-'*10,)

    return trainer


# In[82]:


# 학습을 위한 tokenizer와 사전 학습된 모델을 불러옵니다.
def load_tokenizer_and_model_for_train(config,device):
    print('-'*10, 'Load tokenizer & model', '-'*10,)
    print('-'*10, f'Model Name : {config["general"]["model_name"]}', '-'*10,)
    model_name = config['general']['model_name']
    bart_config = BartConfig().from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'],config=bart_config)

    special_tokens_dict={'additional_special_tokens':config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)

    generate_model.resize_token_embeddings(len(tokenizer)) # 사전에 special token을 추가했으므로 재구성 해줍니다.
    generate_model.to(device)
    print(generate_model.config)

    print('-'*10, 'Load tokenizer & model complete', '-'*10,)
    return generate_model , tokenizer


# ## 3. 모델 학습하기

# - 앞에서 구축한 클래스 및 함수를 활용하여 학습 진행합니다.

# In[83]:


def main(config):
    # 사용할 device를 정의합니다.
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    # 사용할 모델과 tokenizer를 불러옵니다.
    generate_model , tokenizer = load_tokenizer_and_model_for_train(config,device)
    print('-'*10,"tokenizer special tokens : ",tokenizer.special_tokens_map,'-'*10)

    # 학습에 사용할 데이터셋을 불러옵니다.
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token']) # decoder_start_token: str, eos_token: str
    data_path = config['general']['data_path']
    train_inputs_dataset, val_inputs_dataset = prepare_train_dataset(config,preprocessor, data_path, tokenizer)

    # Trainer 클래스를 불러옵니다.
    trainer = load_trainer_for_train(config, generate_model,tokenizer,train_inputs_dataset,val_inputs_dataset)
    trainer.train()   # 모델 학습을 시작합니다.

    # # (선택) 모델 학습이 완료된 후 wandb를 종료합니다.
    # wandb.finish()


# In[25]:


if __name__ == "__main__":
    main(loaded_config)


# ## 4. 모델 추론하기

# In[91]:


# 이곳에 내가 사용할 wandb config 설정
loaded_config['inference']['ckt_path'] = os.path.abspath('./checkpoint')


# - test data를 사용하여 모델의 성능을 확인합니다.

# In[ ]:


# tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
def prepare_test_dataset(config,preprocessor, tokenizer):

    test_file_path = os.path.join(config['general']['data_path'],'test.csv')

    test_data = preprocessor.make_set_as_df(test_file_path,is_train=False)
    test_id = test_data['fname']

    print('-'*150)
    print(f'test_data:\n{test_data["dialogue"][0]}')
    print('-'*150)

    encoder_input_test , decoder_input_test = preprocessor.make_input(test_data,is_test=True)
    print('-'*10, 'Load data complete', '-'*10,)

    test_tokenized_encoder_inputs = tokenizer(encoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['encoder_max_len'], return_token_type_ids=False,)
    test_tokenized_decoder_inputs = tokenizer(decoder_input_test, return_tensors="pt", padding=True,
                    add_special_tokens=True, truncation=True, max_length=config['tokenizer']['decoder_max_len'], return_token_type_ids=False,)

    test_encoder_inputs_dataset = DatasetForInference(test_tokenized_encoder_inputs, test_id, len(encoder_input_test))
    print('-'*10, 'Make dataset complete', '-'*10,)

    return test_data, test_encoder_inputs_dataset


# In[ ]:


# # 추론을 위한 tokenizer와 학습시킨 모델을 불러옵니다.
# def load_tokenizer_and_model_for_test(config,device):
#     print('-'*10, 'Load tokenizer & model', '-'*10,)

#     model_name = config['general']['model_name']
#     ckt_path = config['inference']['ckt_path']
#     print('-'*10, f'Model Name : {model_name}', '-'*10,)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
#     tokenizer.add_special_tokens(special_tokens_dict)

#     generate_model = BartForConditionalGeneration.from_pretrained(ckt_path)
#     generate_model.resize_token_embeddings(len(tokenizer))
#     generate_model.to(device)
#     print('-'*10, 'Load tokenizer & model complete', '-'*10,)

#     return generate_model , tokenizer


# In[97]:


# ===== 1. 임포트 =====
import os
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration

# ===== 2. 체크포인트 설정 =====
checkpoint_base = '/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code/checkpoint'
checkpoints = [d for d in os.listdir(checkpoint_base) if d.startswith('checkpoint-')]

if 'checkpoint-best' in checkpoints:
    selected = 'checkpoint-best'
else:
    selected = sorted(checkpoints, 
                     key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0)[-1]

ckt_path = os.path.join(checkpoint_base, selected)
loaded_config['inference']['ckt_path'] = ckt_path

print(f"✅ 선택된 체크포인트: {selected}")
print(f"✅ 전체 경로: {ckt_path}")

# ===== 3. 함수 재정의 =====
def load_tokenizer_and_model_for_test(config, device):
    """
    추론용 모델 및 토크나이저 로드
    """
    print('-'*10, 'Load tokenizer & model', '-'*10)
    
    ckt_path = config['inference']['ckt_path']
    original_model_name = config['general']['model_name']
    
    print(f"📁 체크포인트: {ckt_path}")
    print(f"🤗 원본 모델: {original_model_name}")
    
    # Tokenizer: 원본 모델에서 로드
    print("  → 원본 모델에서 tokenizer 로드")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(original_model_name)
    
    # 특수 토큰 추가
    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"✅ 특수 토큰 추가: {num_added}개")
    
    # Model: 체크포인트에서 로드
    print(f"  → 체크포인트에서 모델 로드")
    generate_model = BartForConditionalGeneration.from_pretrained(
        ckt_path,
        local_files_only=True
    )
    
    # 토큰 임베딩 크기 조정
    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)
    
    print(f"✅ 모델 로드 완료")
    print(f"   - Tokenizer vocab size: {len(tokenizer)}")
    print(f"   - Model vocab size: {generate_model.config.vocab_size}")
    
    return generate_model, tokenizer

print("\n✅ 함수 정의 완료")
print("이제 추론을 실행할 수 있습니다:")
print("  if __name__ == '__main__':")
print("      output = inference(loaded_config)")


# In[47]:


# 학습된 모델이 생성한 요약문의 출력 결과를 보여줍니다.
def inference(config):
    device = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
    print('-'*10, f'device : {device}', '-'*10,)
    print(torch.__version__)

    generate_model , tokenizer = load_tokenizer_and_model_for_test(config,device)

    data_path = config['general']['data_path']
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])

    test_data, test_encoder_inputs_dataset = prepare_test_dataset(config,preprocessor, tokenizer)
    dataloader = DataLoader(test_encoder_inputs_dataset, batch_size=config['inference']['batch_size'])

    summary = []
    text_ids = []
    with torch.no_grad():
        for item in tqdm(dataloader):
            text_ids.extend(item['ID'])
            generated_ids = generate_model.generate(input_ids=item['input_ids'].to('cuda:0'),
                            no_repeat_ngram_size=config['inference']['no_repeat_ngram_size'],
                            early_stopping=config['inference']['early_stopping'],
                            max_length=config['inference']['generate_max_length'],
                            num_beams=config['inference']['num_beams'],
                        )
            for ids in generated_ids:
                result = tokenizer.decode(ids)
                summary.append(result)

    # 정확한 평가를 위하여 노이즈에 해당되는 스페셜 토큰을 제거합니다.
    remove_tokens = config['inference']['remove_tokens']
    preprocessed_summary = summary.copy()
    for token in remove_tokens:
        preprocessed_summary = [sentence.replace(token," ") for sentence in preprocessed_summary]

    output = pd.DataFrame(
        {
            "fname": test_data['fname'],
            "summary" : preprocessed_summary,
        }
    )
    result_path = config['inference']['result_path']
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output.to_csv(os.path.join(result_path, "output.csv"), index=False)

    return output


# In[96]:


import os

# 실제 체크포인트 위치
checkpoint_base = '/Competition/NLP/naturallanguageprocessingcompetition-nlp-1/code/checkpoint'

# 체크포인트 목록
checkpoints = [d for d in os.listdir(checkpoint_base) if d.startswith('checkpoint-')]
print(f"발견된 체크포인트: {checkpoints}")

# 가장 최근 체크포인트 선택 (번호가 가장 큰 것)
if 'checkpoint-best' in checkpoints:
    selected = 'checkpoint-best'
    print("🏆 베스트 모델 선택")
else:
    selected = sorted(checkpoints, 
                     key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0)[-1]
    print(f"📌 최신 체크포인트 선택: {selected}")

# 전체 경로 설정
ckt_path = os.path.join(checkpoint_base, selected)
loaded_config['inference']['ckt_path'] = ckt_path

print(f"\n✅ 최종 경로: {ckt_path}")
print(f"✅ 경로 존재: {os.path.exists(ckt_path)}")

# 체크포인트 내용 확인
print(f"\n=== {selected} 내용 ===")
files = os.listdir(ckt_path)
print(f"파일 수: {len(files)}")
for f in sorted(files)[:10]:
    print(f"  - {f}")


# In[98]:


# 학습된 모델의 test를 진행합니다.
if __name__ == "__main__":
    output = inference(loaded_config)


# In[99]:


output  # 각 대화문에 대한 요약문이 출력됨을 확인할 수 있습니다.


# In[ ]:




