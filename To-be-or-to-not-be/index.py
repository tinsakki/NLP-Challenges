import re
words = ['am','are','were','was','is','been','being','be']
response = int(input())
data = str(input())
data = data.replace(",","")
data = data.replace(".", "")
data = data.lower()
result = re.split('\n|  ',data)
prediction_list = ' '.join(result)
prediction_list = prediction_list.split(" ")
for i in range(len(prediction_list)):
    if prediction_list[i] == '----':
        if prediction_list[i-1] == 'i':
            print('am')
            continue
        elif prediction_list[i-1] in ['is','are','was','were']:
            print('being')
            continue
        elif prediction_list[i-1] in ['had','has','have']:
            print('been')
            continue
        elif prediction_list[i-1] in ["won't", "can't", "can", "not",'might',"could",'may',"couldn't",'would',"will","wouldn't","shall"]:
            print("be")
            continue
        elif prediction_list[i-1].endswith('s'):
            print('were')
            continue
        elif prediction_list[i-1].endswith('s') and prediction_list[i+1].endswith('ed'):
            print('were')
            continue
        elif not prediction_list[i-1].endswith('s') and prediction_list[i+1].endswith('ed'):
            print('was')
            continue
        elif (prediction_list[i-1].endswith('s') or not prediction_list[i-1].endswith('s')) and prediction_list[i+1].endswith('ing'):
            a = prediction_list[i-20:i]
            if any(True for u in a if u.endswith('ed')):
                if prediction_list[i-1].endswith('s') and prediction_list[i+1].endswith('ing'):
                    print('were')
                    continue
                elif not prediction_list[i-1].endswith('s') and prediction_list[i+1].endswith('ing'):
                    print('was')
                    continue
                continue
            else:
                if prediction_list[i-1].endswith('s') and prediction_list[i+1].endswith('ing'):
                    print('are')
                    continue
                elif not prediction_list[i-1].endswith('s') and prediction_list[i+1].endswith('ing'):
                    print('is')
                    continue
        elif (not prediction_list[i-1].endswith('s') and not prediction_list[i-1].endswith('s') and not prediction_list[i+1].endswith('ing') and not prediction_list[i+1].endswith('ed')):
            a = prediction_list[i - 20:i]
            if any(True for u in a if u.endswith('ed')):
                print('was')
                continue
            else:
                print('is')
                continue