from arguments import NERDeployArguments
import torch
from transformers import BertConfig, BertForTokenClassification
from transformers import BertTokenizer
import re
import pandas as pd
import random

# kcbert-base model deploy
args = NERDeployArguments(
    pretrained_model_name="beomi/kcbert-base",
    downstream_model_dir="checkpoint-ner3",
    max_seq_length=64,
)

fine_tuned_model_ckpt = torch.load(
    args.downstream_model_checkpoint_fpath,
    map_location=torch.device("cpu")
)

pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model_name,
    num_labels=fine_tuned_model_ckpt['state_dict']['model.classifier.bias'].shape.numel(),
)
model = BertForTokenClassification(pretrained_model_config)
model.load_state_dict({k.replace("model.",""): v for k, v in fine_tuned_model_ckpt['state_dict'].items()}, strict=False)
model.eval()

tokenizer = BertTokenizer.from_pretrained(
    args.pretrained_model_name,
    do_lower_case=False,
)

labels = [label.strip() for label in open(args.downstream_model_labelmap_fpath, "r").readlines()]
id_to_label = {}
for idx, label in enumerate(labels):
  if "DUR" in label:
    label = "기간"
  elif "NOP" in label:
    label = "사람수"
  elif "DST" in label:
    label = "식단구조"
  elif "EXA" in label:
    label = "알러지표현"
  elif "EXD" in label:
    label = "비선호표현"
  elif "EXL" in label:
    label = "선호표현"
  elif "ONE" in label:
    label = "식단한종류"
  elif "FOI" in label:
    label = "식재료"
  else:
    label = label
  id_to_label[idx] = label

# 추론 함수
def inference_fn(sentence):
  inputs = tokenizer(
      [sentence],
      max_length=args.max_seq_length,
      padding="max_length",
      truncation=True,
  )
  with torch.no_grad():
      outputs = model(**{k: torch.tensor(v) for k, v in inputs.items()})
      probs = outputs.logits[0].softmax(dim=1)
      top_probs, preds = torch.topk(probs, dim=1, k=1) # 예측 값에서 top-k개의 결과를 받고 싶을 때
      tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
      predicted_tags = [id_to_label[pred.item()] for pred in preds]
      result = []
      for token, predicted_tag, top_prob in zip(tokens, predicted_tags, top_probs):
          if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
              token_result = {
                  "token": token,
                  "predicted_tag": predicted_tag,
                  "top_prob": str(round(top_prob[0].item(), 4)),
              }
              result.append(token_result)
  return {
      "sentence": sentence,
      "result": result,
  }
  
 # 추론함수 사용하고, 추론 결과를 알아들을 수 있는 predicted_tag 리스트와 token_list로 변환하는 함수
def inference_to_words(sentence):
  inference = inference_fn(sentence)
  result = inference['result']
  # print(result)

  token_list = [] # 0 태그에 해당하지 않는 단어토큰들만
  predicted_tag = [] # O 빼고 O 일 경우 그다음 태그로 넘어가기

  pattern = re.compile('[가-힣a-zA-Z0-9,.~-]+') #
  # p1 = pattern.search('2~3일')

  for i in range(len(result)):
    if len(predicted_tag) > 0:
      if result[i]['predicted_tag'] != 'O':
        if (predicted_tag[len(predicted_tag)-1] != result[i]['predicted_tag']) | (result[i-1]['predicted_tag']=='O'): #inference['result'][i]['predicted_tag'] == 'O'
          predicted_tag.append(result[i]['predicted_tag'])
          token_list.append(pattern.search(result[i]['token']).group())
        else:
          temp_str = token_list[len(token_list)-1]
          # print(temp_str, result[i]['token'])
          temp_str += pattern.search(result[i]['token']).group()
          token_list[len(token_list)-1] = temp_str
    else:
      if result[i]['predicted_tag'] != 'O':
        predicted_tag.append(result[i]['predicted_tag'])
        token_list.append(result[i]['token'])
  return predicted_tag, token_list

# 태그되어 있는 모든 용어들의 딕셔너리 <- 숫자로 변환
# 단위사전
unit = {
    '끼':1,
    '식':1,
    '일': 1,
    '주일': 7,
    '달': 30,
    '명':1,
    '인':1,
    '주':7,
}

#DST: 식단구조
dst_dict = {'아침점심저녁': 3, # 각 표현에 대한 하루 끼니수
            '아침': 1,
            '점심저녁': 2,
            '아침점심': 2,
            '한끼': 1,
            '삼시새끼': 3,
            '점심': 1,
            '두끼': 2,
            '저녁': 1,
            '아침,점심,저녁': 3,
            '하루두끼 또는 하루한끼': 2,
            '아침.점심.저녁': 3,
            '아점저': 3,
            '아침이랑 점심, 저녁': 3,
            '삼시세끼': 3,
            '아침이나 점심': 1,
            '세끼': 3,
            '아침/점심/저녁': 3,
            '세 끼': 3,
            '아침점심': 2,
            '아침 점심 저녁 3끼': 3,
            '아침-점심-저녁': 3,
            '아침 / 점심 / 저녁': 3}

#DUR: 기간 -> 최대 한달 식단만
dur_dict = {'일주일': 7, # 기간 일수
            '한달': 30,
            '한주': 7,
            '주말': 2,
            '내일': 1,
            '한 달': 30,
            '토요일': 1,
            '오늘': 1,
            '월~목': 4,
            '금~일': 3,
            '일요일': 1,
            '보름': 15
            }

nop_dict = {'혼자': 1,'자취': 1, '자취생':1, '혼밥':1} #단위: 명,인

# 식재료 리스트에서 '가을를은는'같은 접미사 제거하는 함수
def remove_suffix(foi_list):
  pattern = r'([a-zA-Z가-힣\s]+)(?:는|은|을|를|가|랑)'
  for i in range(len(foi_list)):
    match = re.search(pattern,foi_list[i])
    if match:
      foi_list[i] = match.group(1)
  return foi_list

# 식재료는 선호리스트, 불호리스트로 만들기
def foi_preprocessing(predicted_tag, token_list, foi_like_input=[], foi_dislike_input=[]):
  foi_exp_score = {'비선호표현':0, '선호표현':1, '알러지표현':0}
  foi_like = foi_like_input
  foi_dislike = foi_dislike_input

  # ['사과', '당근', '오이', '일주일']
  # ['식재료', '식재료', '식재료', '기간']

  # 만약 식재료 태그가 있는데, 아무런 표현이 없는 경우는 식재료 선호로 보기
  # 만약 식재료 태그가 한개 이상 있는데, 표현이 하나밖에 없는 경우는, 그 표현으로 모든 식재료 적용하기
  # 만약 식재료 태그가 있는데, 식재료 뒤에 식재료 뒤에 식재료 뒤에 표현, 식재료 뒤에 식재료 뒤에 표현 일 경우 표현 앞에 있는 것들만 각각 그 표현으로 묶기
  count_exp = 0
  exp_dict = {} # predicted_tag에 어떤 exp 태그가 몇번째 인덱스에 있는지에 대한 사전

  if '식재료' in predicted_tag:
    for i in range(len(predicted_tag)):
      if predicted_tag[i] in ['비선호표현','선호표현','알러지표현']:
        count_exp += 1
        exp_dict[predicted_tag[i]] = i
    if count_exp == 0: # 만약 식재료 태그가 있는데, 아무런 표현이 없는 경우는 식재료 선호로 보고 foi_like 리스트에 추가
      for f in range(len(predicted_tag)):
        if predicted_tag[f] == '식재료':
          foi_like.append(token_list[f])
    elif count_exp == 1: # 만약 식재료 태그가 한개 이상 있는데, 표현이 하나밖에 없는 경우는, 그 표현으로 모든 식재료 적용하기
      for f in range(len(predicted_tag)):
        if predicted_tag[f] == '식재료':
          if foi_exp_score[list(exp_dict.keys())[0]] == 0:
            foi_dislike.append(token_list[f])
          else:
            foi_like.append(token_list[f])
    else: # 만약 식재료 태그가 있는데, 식재료 뒤에 식재료 뒤에 식재료 뒤에 표현, 식재료 뒤에 식재료 뒤에 표현 일 경우 표현 앞에 있는 것들만 각각 그 표현으로 묶기 조금더 발전해야함.
      for f in range(len(predicted_tag)):
        if predicted_tag[f] == '식재료':
          for g in range(f+1,len(predicted_tag)):
            if predicted_tag[g] =='식재료':
              continue
            elif predicted_tag[g] in ['비선호표현','선호표현','알러지표현']:
              # print(predicted_tag[g])
              if foi_exp_score[predicted_tag[g]] == 0:
                foi_dislike.append(token_list[f])
                break
              else:
                foi_like.append(token_list[f])
                break

  foi_like = remove_suffix(foi_like)
  foi_dislike = remove_suffix(foi_dislike)

  return foi_like, foi_dislike

## 개체명 전처리 함수들
# DST
def dst_preprocessing(predicted_tag, token_list, dst=3):
  pattern_dst = r'([0-9]+\s*)(?:끼|번|식)'

  if "식단구조" in predicted_tag:
    try:
      dst = dst_dict[token_list[predicted_tag.index("식단구조")]] # 인덱스 함수를 썼음으로 첫번째 식단구조 표현
    except:
      match = re.search(pattern_dst, token_list[predicted_tag.index("식단구조")])
      if match:
        dst = int(match.group(1).strip())
      else:
        print(f'{token_list[predicted_tag.index("식단구조")]} 식단구조를 숫자로 표현하는데 실패하였습니다.')
  return dst

# NOP
def nop_preprocessing(predicted_tag, token_list, nop=1):
  pattern_nop = r'([0-9]+\s*)(?:인|명|인분|사람)'

  if "사람수" in predicted_tag:
    try:
      nop = nop_dict[token_list[predicted_tag.index("사람수")]] # 인덱스 함수를 썼음으로 첫번째 식단구조 표현
    except:
      match = re.search(pattern_nop, token_list[predicted_tag.index("사람수")])
      if match:
        nop = int(match.group(1).strip())
      else:
        print(f'{token_list[predicted_tag.index("사람수")]} 사람수를 숫자로 표현하는데 실패하였습니다.')
  return nop

# DUR
def dur_preprocessing(predicted_tag, token_list, dur=1):
  pattern_dur = r'([0-9]+)\s*(일|주일|달|년|주)'
  dur_unit = '' # 단위 넣어두는 str

  if "기간" in predicted_tag:
    try:
      dur = dur_dict[token_list[predicted_tag.index("기간")]] # 인덱스 함수를 썼음으로 첫번째 식단구조 표현
    except:
      try:
        matches = re.findall(pattern_dur, token_list[predicted_tag.index("기간")])
        for match in matches:
          dur, dur_unit= match
          dur = int(dur)
          if dur_unit != '':
            dur *= unit[dur_unit]
      except:
        print(f'{token_list[predicted_tag.index("기간")]} 기간을 숫자로 표현하는데 실패하였습니다.')
  return dur

# ONE
def one_preprocessing(predicted_tag, token_list, one=False):
  if "식단한종류" in predicted_tag:
    one = True
  return one

# 유저의 인풋 summary 출력
def user_input_askback(dur, dst, nop, one, foi_like, foi_dislike):
  summary = ''

  if len(foi_dislike) > 0:
    summary += '[제외할 식재료] '
    for i in range(len(foi_dislike)):
      summary += foi_dislike[i]
      if i != (len(foi_dislike) - 1):
        summary += ', '
      else:
        summary += '<br/>'

  if len(foi_like) > 0:
    summary += '[포함할 식재료] '
    for i in range(len(foi_like)):
      summary += foi_like[i]
      if i != (len(foi_like) - 1):
        summary += ', '
      else:
        summary += '<br/>'

  summary += '<br/> 위 정보를 바탕으로 '
  if one == True:
    summary += '간편식만으로 '
  summary += f'총 {dur}일치 {dst}끼 {nop}인 식단 생성합니다.'

  return summary

# 표준 끼니 생성 함수 밥1,국1,반찬2,김치1
def meal_generator(foi_like, foi_dislike):
  df = pd.read_excel('./dataset/dataset_FoodNutritionTable_final.xlsx')
  # df의 columns=['식품명',	'라벨링', '식품군'	'1인 기준량(g)',	'에너지(㎉)',	'단백질(g)',	'지방(g)',	'탄수화물(g)',	'재료', '레시피','Frequency'])
  # dur_dst = dur*dst # 몇일 * 하루 끼니 수 = 총 끼니수

  grain_req = 1 # 곡류 1
  vege_req = 3 # 채소 식단 3
  protein_req = 1 # 큰접시로 하나

  grain_count = 0
  vege_count = 0
  protein_count = 0

  calory = 0

  # 1. 사용자의 비선호 식재료가 포함된 데이터 삭제처리 (foi_dislike :: 사용자의 비선호 식재료 리스트)
  for strN in foi_dislike:
    idx = df[df['재료'].str.contains(strN)].index
    df.drop(idx, inplace=True)
    df.reset_index(drop=True, inplace=True)

  # 2. 사용자의 선호 식재료가 포함된 데이터는 frequency 칼럼에 가중치 +7 하기
  for strY in foi_like:
    idx = df[df['재료'].str.contains(strY)].index
    df.loc[idx,'Frequency'] += 7

  # 3. 첫번째로, 밥 종류부터 1가지 가중치+랜덤으로 추출 후 곡류에 +1 하기
  rice_df = df[df['라벨링']== '밥&죽']
  rice_df.reset_index(drop=True, inplace=True)
  selected_rice = random.choices(rice_df['식품명'], weights=rice_df['Frequency'], k=1)[0]
  grain_count += 1

  # 4. 두번째로, 국/탕/찜/찌개를 1가지 가중치+랜덤으로 추출 후 식품군에 맞춰서 vege/protein_count에 +1하기
  soup_df = df[(df['라벨링']=='찜&찌개')|(df['라벨링']=='국&탕')]
  soup_df = soup_df[soup_df['식품군']!='곡류']
  soup_df.reset_index(drop=True, inplace=True)
  selected_soup = random.choices(soup_df['식품명'], weights=soup_df['Frequency'], k=1)[0]
  soup_category = soup_df.loc[soup_df[soup_df['식품명']==selected_soup].index,'식품군'].values
  if '채소류' in soup_category:
    vege_count +=1
  else:
    protein_count +=1

  # 5. 세번째로, 구이류/반찬류 이미 protein_count가 protein_req=1 과 같을 경우, 채소류만 두가지 뽑고, 아닐 경우 구이류에서 한가지, 채소류에서 한가지 뽑기
  selected_sides = []
  if protein_count == protein_req:
    sides_df = df[(df['라벨링']=='반찬류')&(df['식품군']=='채소류')]
    sides_df.reset_index(drop=True, inplace=True)
    selected_s = random.choices(sides_df['식품명'], weights=sides_df['Frequency'], k=1)[0]
    selected_sides.append(selected_s)
    sides_df.loc[sides_df[sides_df['식품명']==selected_s].index,'Frequency'] = 0
    selected_s1 = random.choices(sides_df['식품명'], weights=sides_df['Frequency'], k=1)[0]
    selected_sides.append(selected_s1)
    vege_count += 2
  else:
    sides_df = df[(df['라벨링']=='반찬류')]
    sides_df = sides_df[sides_df['식품군']!='곡류']
    sides_df.reset_index(drop=True, inplace=True)
    grilled_sides_df = df[(df['라벨링']=='구이류')]
    grilled_sides_df.reset_index(drop=True, inplace=True)
    selected_sides.append(random.choices(sides_df['식품명'], weights=sides_df['Frequency'], k=1)[0])
    selected_sides.append(random.choices(grilled_sides_df['식품명'], weights=grilled_sides_df['Frequency'], k=1)[0])
    protein_count += 1
    sides_category = sides_df.loc[sides_df[sides_df['식품명']==selected_soup].index,'식품군'].values
    if '채소류' in sides_category:
      vege_count +=1
    else:
      protein_count +=1

  # 6. 마지막으로 김치를 정한다.
  kimchi_df = df[df['라벨링']== '김치류']
  kimchi_df.reset_index(drop=True, inplace=True)
  selected_kimchi = random.choices(kimchi_df['식품명'], weights=kimchi_df['Frequency'], k=1)[0]
  vege_count += 1

  # 7. 칼로리 계산하기.
  calory += df.loc[df[df['식품명']==selected_rice].index,'에너지(㎉)'].values[0]
  calory += df.loc[df[df['식품명']==selected_soup].index,'에너지(㎉)'].values[0]
  calory += df.loc[df[df['식품명']==selected_sides[0]].index,'에너지(㎉)'].values[0]
  calory += df.loc[df[df['식품명']==selected_sides[1]].index,'에너지(㎉)'].values[0]
  calory += df.loc[df[df['식품명']==selected_kimchi].index,'에너지(㎉)'].values[0]

  # print(selected_rice)
  # print(selected_soup)
  # print(selected_sides)
  # print(selected_kimchi)
  # print(calory)

  # print(grain_count, vege_count, protein_count)
  return selected_rice, selected_soup, selected_sides, selected_kimchi, grain_count, vege_count, protein_count, calory

# 끼니를 표현하는 식
def meal_string (nth,selected_rice, selected_soup, selected_sides, selected_kimchi, grain_count, vege_count, protein_count, calory):
  string_result = f"<dl><dt> 끼니 {nth} </dt><dd> {selected_rice}, {selected_soup}, {selected_sides[0]}, {selected_sides[1]}, {selected_kimchi} </dd></dl><p>(끼니구성: 곡류 {grain_count}가지, 채소류 {vege_count}가지, 단백질류 {protein_count}가지 // 인당 칼로리: {round(calory,1)}kcal)</p>"
  return string_result

#간단식 생성 함수
def light_meal_generator(foi_like, foi_dislike):
  df = pd.read_excel('./dataset/dataset_FoodNutritionTable_yubin수정1.xlsx')
  # df의 columns=['식품명',	'라벨링', '식품군'	'1인 기준량(g)',	'에너지(㎉)',	'단백질(g)',	'지방(g)',	'탄수화물(g)',	'재료', '레시피','Frequency'])
  # dur_dst = dur*dst # 몇일 * 하루 끼니 수 = 총 끼니수

  calory = 0
  gram = 0
  protein = 0
  fat = 0
  carb = 0

  # 1. 비선호 식재료 제거
  for strN in foi_dislike:
    idx = df[df['재료'].str.contains(strN)].index
    df.drop(idx, inplace=True)
    df.reset_index(drop=True, inplace=True)

  # 2. 사용자의 선호 식재료가 포함된 데이터는 frequency 칼럼에 가중치 +100 하기
  for strY in foi_like:
    idx = df[df['재료'].str.contains(strY)].index
    df.loc[idx,'Frequency'] += 100

  # 3. 랜덤하게 뽑기
  light_meal_df = df[df['라벨링']=='간편식']
  light_meal_df.reset_index(drop=True, inplace=True)
  selected_meal = random.choices(light_meal_df['식품명'], weights=light_meal_df['Frequency'], k=1)[0]

  # 4. 칼로리, 탄단백 수치 뽑기
  calory = df.loc[df[df['식품명']==selected_meal].index,'에너지(㎉)'].values[0]
  protein = df.loc[df[df['식품명']==selected_meal].index,'단백질(g)'].values[0]
  fat = df.loc[df[df['식품명']==selected_meal].index,'지방(g)'].values[0]
  carb = df.loc[df[df['식품명']==selected_meal].index,'탄수화물(g)'].values[0]
  gram = df.loc[df[df['식품명']==selected_meal].index,'1인 기준량(g)'].values[0]
  return selected_meal, calory, protein, fat, carb, gram

# meal_generator 함수를 사용하여 총 끼니수만큼 식단 구성을 만드는 함수
def multiple_meal_generator(one, dur, dst, foi_like, foi_dislike):
  meal_num = dur*dst
  meal_summary = ""
  if one == True:
    for i in range(meal_num):
      meal_light = light_meal_generator(foi_like,foi_dislike)
      selected_meal = meal_light[0]
      calory = meal_light[1]
      protein = meal_light[2]
      fat = meal_light[3]
      carb = meal_light[4]
      gram = meal_light[5]
      meal_summary += f"<dl><dt> 끼니 {i+1} </dt><dd> {selected_meal} </dd></dl><p>(식품정보: 1인 기준량 {gram}g, 칼로리 {round(calory,1)}kcal, 단백질 {round(protein,1)}g, 지방 {round(fat,1)}g, 단백질 {round(carb,1)}g)</p>"

      if i+1 == meal_num:
        meal_summary += "간편식은 식품정보를 참고하여 부족한 영양분을 간식으로 채우기를 권장합니다."
  else:
    for i in range(meal_num):
      meal1 = meal_generator(foi_like, foi_dislike)
      selected_rice = meal1[0]
      selected_soup = meal1[1]
      selected_sides = meal1[2]
      selected_kimchi = meal1[3]
      grain_count = meal1[4]
      vege_count = meal1[5]
      protein_count = meal1[6]
      calory = meal1[7]
      meal_summary += meal_string (i+1,selected_rice, selected_soup, selected_sides, selected_kimchi, grain_count, vege_count, protein_count, calory)

  return meal_summary

# 식품 재료/레시피 제공 함수
def food_information(food):
  food_summary = ""
  df = pd.read_excel('./dataset/dataset_FoodNutritionTable_yubin수정1.xlsx')
  # df의 columns=['식품명',	'라벨링', '식품군'	'1인 기준량(g)',	'에너지(㎉)',	'단백질(g)',	'지방(g)',	'탄수화물(g)',	'재료', '레시피','Frequency'])
  try:
    recipe = df.loc[df[df['식품명']==food].index,'레시피'].values[0]
    ingredients = df.loc[df[df['식품명']==food].index,'재료'].values[0]
    food_summary += f"재료: {ingredients}<br><br>레시피:<br>{recipe}"
  except:
    food_summary = "잘못입력하셨습니다. 추천드린 식단에 대한 레시피/재료 확인을 원하시면 음식명을 입력해주시고, 원치 않으시면 '아니요'를 입력해주세요."
  return food_summary