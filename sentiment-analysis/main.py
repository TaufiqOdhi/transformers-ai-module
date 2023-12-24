from transformers import pipeline, set_seed
import requests

classifier = pipeline("text-classification", model='distilbert-base-uncased-finetuned-sst-2-english', revision='af0f99b')
# classifier = pipeline("text-classification", model = "Souvikcmsa/BERT_sentiment_analysis")


def sentiment_analysis(text):
    result = classifier(text)[0]
    return result['label'], result['score']


def text_generation(prompt):
    model = pipeline('text-generation', model='gpt2')
    # set_seed(42)
    return model(prompt)


def question_answer(context: str, question: str):
    model = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')
    return model(question=question, context=context)



def translate(text, lang_source="id"):
    lang_target = "en"
    translate_api = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={lang_source}&tl={lang_target}&dt=t&q={text}"
    response = requests.get(translate_api)
    if response.status_code == 200:
        return response.json()[0][0][0]
    else:
        return "Error"
    


if __name__ == "__main__":
    # print(sentiment_analysis(text="I am mad !"))
    # print(sentiment_analysis(text="I am happy righ now !"))
    # print(text_generation("Hello, can you list days name in one week ?"))

    context = r"""

        Extractive Question Answering is the task of extracting an answer from a text given a question. An example     of a

        question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune

        a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.

    """
    question = "What is a good example of a question answering dataset?"
    print(question_answer(context=context, question=question))
