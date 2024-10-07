from transformers import pipeline

# classifier = pipeline("zero-shot-classification")
# classifier_result = classifier(
#     "This is a course about the Transformers library",
#     candidate_labels=["education", "politics", "business"],
# )
#
# print(f"The classifier result is: {classifier_result}")

# text-generation
# generator = pipeline("text-generation", model="distilbert/distilgpt2")
# generator_result = generator("In this course, we will teach you how to", max_length=15, num_return_sequences=2)
#
# print(f"The generator result is: {generator_result}")

# fill mask
# unmasker = pipeline("fill-mask")
# mask_result = unmasker("This course will teach you all about <mask> models.", top_k=2)
#
# print(f"The mask result is: {mask_result}")

# Named Entity Recognition (ner)
# ner = pipeline("ner", grouped_entities=True)
# ner_result = ner("He's called Kwame ,and he's a software engineer at ABC Consult in Ghana.")
# print(f"The NER result is: {ner_result}")

# Question Answering
# question_answerer = pipeline("question-answering")
# answer = question_answerer(
#     question="Where does Kwame work?",
#     context="He's called Kwame ,and he's a software engineer at ABC Consult in Ghana.",
# )
#
# print(f"The answer is: {answer}")

# translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-tw")
# translated_text = translator("My name is Kofi. I am 8 years old")

translator = pipeline("translation", model="facebook/nllb-200-distilled-600M")
translated_text = translator("My name is Kofi. I am 8 years old", src_lang="eng_Latn", tgt_lang="ewe_Latn")

print(f"The translated text is: {translated_text}")