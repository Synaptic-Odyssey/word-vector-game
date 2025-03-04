
#from gensim.models import KeyedVectors
import spacy
import random


#Export this functionality with FastAPI or Flask to create a website
class word_game:
    
    def __init__(self):
        
        self.nlp_model = spacy.load("en_core_web_md")
        #requires download...using spaCy for now
        #self.word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
        
    def input(self):
        
        self.word_guess = input()
    
    '''
        the challenge is finding two words that add up to one word in a way that makes sense.
        Not just for addition of 2 words, both addition and subtraction of multiple words.
        (could do something similar to Wordle, start with a 2 word operation then go up to 3-4 words)
        there is no fixed answer, it just has to be close to the two words
    '''
    
    '''
    for simplicity, could randomly generate 2 words, then check if the guess is close to the summed vector,
    or match with a list of words that is close to the vector
    After randomization could check if the words that exist close to the summed vector even make sense 
    --> this is time consuming and seems dumb
    '''
    def check_random (self, guess, threshold):
        
        #is_alpha makes sure no special letters
        
        meaningful_words = [word.text for word in self.nlp.model.vocab if word.has_vector and word.is_alpha and word.is_lower]
        
        rand1 = int(random.random()*len(meaningful_words))
        
        while True:
            rand2 = int(random.random()*len(meaningful_words))
            if rand1 != rand2:
                break
            
        word1 = meaningful_words[rand1]
        word2 = meaningful_words[rand2]
        
        #doc is token container --> doc[0] is the token for the first word
        doc1 = self.nlp_model(word1)
        word1_vector = doc1[0].vector
        
        doc2 = self.nlp_model(word2)
        word2_vector = doc2[0].vector
        
        sum_vector = word1_vector + word2_vector
        
        doc3 = self.nlp_model(guess)
        guess_vector = doc3[0].vector
        
        similarity = sum_vector.similarity(guess_vector)
        
        #issue! the user won't be able to see word1 and word2 LOL
        
        if similarity >= threshold:
            print(f" CORRECT! {word1} + {word2} = {guess}")
        else:
            print("INCORRECT")