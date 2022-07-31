import requests

API_URL = "https://api-inference.huggingface.co/models/bert-base-chinese"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


def multimask_predictions(txt, outputs, check_vocabs):
    all_outputs = {}
    for i, i_output in enumerate(outputs):
        for output in i_output:
            pred_vocab = output['token_str']
            if pred_vocab in check_vocabs:
                all_outputs[i] = pred_vocab
                break
    corrected_txt = ''
    pred_cnt = 0
    for word in txt:
        if word in check_vocabs:
            if pred_cnt in all_outputs.keys():
                word = all_outputs[pred_cnt]
            pred_cnt += 1
        corrected_txt += word
    return corrected_txt

def main(check_vocabs):
    txt = input("打中文: ")  # user input
    txt_mask = txt
    for word in check_vocabs:
        txt_mask = txt_mask.replace(word, '[MASK]')

    output = query({ "inputs": txt_mask, })
    if txt_mask.count('[MASK]') <= 1:  # only one word is detected
        corrected_txt = output[0]['sequence'].replace(" ", "")
    else:  # more than one word are masked
        corrected_txt = multimask_predictions(txt, output, check_vocabs)
    print(txt + " -> " + corrected_txt)
    return

if __name__ == '__main__':
    zai_zai = ['在', '再']
    main(zai_zai)
