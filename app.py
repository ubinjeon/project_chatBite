from flask import Flask, request, jsonify, render_template
import deploy
import re

def remove_special_characters(text):
    # 특수문자를 제거하는 정규표현식
    pattern = r'[^a-zA-Z0-9가-힣\s]'  # 알파벳, 숫자, 한글, 공백을 제외한 모든 문자
    return re.sub(pattern, '', text)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

count_no = 0
count_yes = 0
count_else = 0
count_end = 0
count1_else = 0

@app.route("/get", methods=["GET", "POST"])
def chat():
    # for i in range(5):
    msg = request.form["msg"]
    input = msg
    
    yes_list = ['네 마음에 들어요', '좋습니다','괜찮습니다', '예', '괜찮은것 같아요','응','좋아요','넹','네','넵']

    global count_no
    global count_yes
    global count_else
    global count_end
    global count1_else

    if count_yes == 0:
        if remove_special_characters(input) in yes_list:
            count_yes += 1 # 유저 인풋이 예와 같은 긍정의 의미일때
        elif remove_special_characters(input) == '아니요':
            count_no +=1
        elif remove_special_characters(input) == '종료':
            count_end +=1
        else:
            count_else +=1 # 유저 인풋이 그 외의 의미일때
    else:
        if remove_special_characters(input) == '종료':
            count_end +=1
        else:
            count1_else +=1

    return get_Chat_response(input, count_yes, count_no, count_else, count1_else, count_end)


# 숫자로 표현되어져야 할 개체명.
dst = 3 # 끼니수
nop = 1 # 인분
dur = 1 # 기간

# Boolean으로 표현되어져야 할 것
one = False

# List로 표현되어져야 할 것
foi_like = []
foi_dislike = []

def get_Chat_response(text, yes_count, no_count, else_count, else_count_1, end_count):

    # 전역변수를 업데이트할 거라서 전역변수 선언
    global dst
    global nop
    global dur
    global one
    global foi_like
    global foi_dislike

    #출력되어져야 하는 str
    return_answer = ""
    if end_count == 0:
        if yes_count == 0:
            if else_count != no_count:
                inference_result = deploy.inference_to_words(text)
                predicted_tag = inference_result[0]
                token_list = inference_result[1]
                foi_result = deploy.foi_preprocessing(predicted_tag, token_list,foi_like,foi_dislike)
                foi_like = foi_result[0] # foi like list
                foi_dislike = foi_result[1] # foi dislike list
                dur = deploy.dur_preprocessing(predicted_tag, token_list) # 1일
                dst = deploy.dst_preprocessing(predicted_tag, token_list) # 3끼
                nop = deploy.nop_preprocessing(predicted_tag, token_list) # 1인
                one = deploy.one_preprocessing(predicted_tag, token_list) # True / False
                summary = deploy.user_input_askback(dur, dst, nop, one, foi_like, foi_dislike)
                return_answer = summary+"<br><br>"
                return_answer += deploy.multiple_meal_generator(one, dur, dst, foi_like, foi_dislike)
                return_answer += "<br>추천드린 식단 마음에 드시나요?"
            else:
                return_answer = "마음에 들지 않으시다면 식단에 반영해야 할 내용 입력해주시면 다시 반영하여 추천드리겠습니다."
        else:
            if else_count_1 ==0:
                return_answer = "추천드린 식단에 대한 레시피/재료 확인을 원하시면 음식명을 입력해주시고, 원치 않으시면 '종료'를 입력해주세요."
            else:
                return_answer = deploy.food_information(text) + "<br><br>추천드린 식단에 대한 레시피/재료 확인을 원하시면 음식명을 입력해주시고, 원치 않으시면 '종료'를 입력해주세요."
    else:
        return_answer = "챗봇을 종료합니다. 언제든 식단이 필요하시면 다시 찾아주세요."
        
    return return_answer


if __name__ == '__main__':
    app.run(debug=True)

# def get_Chat_response(text, yes_count, no_count, else_count, else_count_1, end_count):

#     # 전역변수를 업데이트할 거라서 전역변수 선언
#     global dst
#     global nop
#     global dur
#     global one
#     global foi_like
#     global foi_dislike

#     #출력되어져야 하는 str
#     return_answer = ""
    
#     if ((yes_count ==0) & (else_count==1) & (no_count==0)):
#         inference_result = deploy.inference_to_words(text)
#         predicted_tag = inference_result[0]
#         token_list = inference_result[1]
#         foi_result = deploy.foi_preprocessing(predicted_tag, token_list,foi_like,foi_dislike)
#         foi_like = foi_result[0] # foi like list
#         foi_dislike = foi_result[1] # foi dislike list
#         dur = deploy.dur_preprocessing(predicted_tag, token_list) # 1일
#         dst = deploy.dst_preprocessing(predicted_tag, token_list) # 3끼
#         nop = deploy.nop_preprocessing(predicted_tag, token_list) # 1인
#         one = deploy.one_preprocessing(predicted_tag, token_list) # True / False
#         summary = deploy.user_input_askback(dur, dst, nop, one, foi_like, foi_dislike)
#         return_answer = summary+"<br><br>"
#         return_answer += deploy.multiple_meal_generator(one, dur, dst, foi_like, foi_dislike)
#         return_answer += "<br>추천드린 식단 마음에 드시나요?"
#     elif ((yes_count ==0) & (no_count>1) & ((else_count+no_count)%2==1)):
#         return_answer = "마음에 들지 않으시다면 식단에 반영해야 할 내용 입력해주시면 다시 반영하여 추천드리겠습니다."
#     elif ((yes_count ==0) & (no_count>1) &((else_count+no_count)%2==0)):
#         inference_result = deploy.inference_to_words(text)
#         predicted_tag = inference_result[0]
#         token_list = inference_result[1]
#         foi_result = deploy.foi_preprocessing(predicted_tag, token_list,foi_like,foi_dislike)
#         foi_like = foi_result[0] # foi like list
#         foi_dislike = foi_result[1] # foi dislike list
#         dur = deploy.dur_preprocessing(predicted_tag, token_list) # 1일
#         dst = deploy.dst_preprocessing(predicted_tag, token_list) # 3끼
#         nop = deploy.nop_preprocessing(predicted_tag, token_list) # 1인
#         one = deploy.one_preprocessing(predicted_tag, token_list) # True / False
#         summary = deploy.user_input_askback(dur, dst, nop, one, foi_like, foi_dislike)
#         return_answer = summary+"<br><br>"
#         return_answer += deploy.multiple_meal_generator(one, dur, dst, foi_like, foi_dislike)
#         return_answer += "<br>추천드린 식단 마음에 드시나요?"
#     elif ((yes_count == 1) & (else_count_1==0)) :
#         return_answer = "추천드린 식단에 대한 레시피/재료 확인을 원하시면 음식명을 입력해주시고, 원치 않으시면 '종료'를 입력해주세요."
#     elif ((yes_count == 1) & (else_count_1>0)):
#         return_answer = deploy.food_information(text)
#     elif end_count>0:
#         return_answer = "챗봇을 종료합니다. 언제든 식단이 필요하시면 다시 찾아주세요."
#     else:
#         return_answer = "챗봇을 종료합니다. 언제든 식단이 필요하시면 다시 찾아주세요."
#     return return_answer