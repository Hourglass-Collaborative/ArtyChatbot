from collections import Counter
from responses import responses, blank_spot, welcome_message
from helpers import preprocess, compare_overlap, pos_tag, extract_nouns, compute_similarity, preprocess_tfidf
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json

exit_commands = ("goodbye", "good bye", "exit", "stop", "cya", "talk later", "done", "that's enough")

def generate_response(prompt_input):
    for command in exit_commands:
        if command in prompt_input:
            return "Goodbye"


    best_response = find_intent_match(prompt_input, responses, algorithm='bow')

    cleaned_response = remove_tags(best_response)
    
    return cleaned_response

def remove_tags(text):
    pattern = re.compile(r'\#\w+')
    
    # re.sub() function replaces the pattern with an empty string.
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text

def find_intent_match(prompt_input, responses, algorithm='bow'):
    if algorithm == 'bow':
        #IMPLEMENT BOW
        ## print(f'Prompt Input: {prompt_input}')
        bow_prompt = Counter(preprocess(prompt_input))
        ##print(f'Prompt BOW: {bow_prompt}')
        processed_responses = [Counter(preprocess(response)) for response in responses]
        similarity_list = [compare_overlap(response, bow_prompt) for response in processed_responses]
        ##print(similarity_list)
        # print(f'Similarity List: {similarity_list}')
        response_index = similarity_list.index(max(similarity_list))
        # print(f'Response Index: {response_index}')

        return responses[response_index]


    elif algorithm == 'tfidf':
        #IMPLEMENT TF-IDF
        documents = [response for response in responses] + [prompt_input]
        processed_documents = [preprocess_tfidf(doc) for doc in documents]
        vectorizer = TfidfVectorizer()
        tfidf_vectors = vectorizer.fit_transform(processed_documents)
        cosine_similarities = cosine_similarity(tfidf_vectors[-1], tfidf_vectors)
        print(cosine_similarities)
        similar_response_index = cosine_similarities.argsort()[0][-2]
        best_response = documents[similar_response_index]

        return best_response


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)