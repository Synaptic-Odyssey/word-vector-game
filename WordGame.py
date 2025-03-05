
#from gensim.models import KeyedVectors
import spacy
import random
from numpy.linalg import norm

def main():
    word_game = WordGame()
    word_game.check_random(0.6)

#Export this functionality with FastAPI or Flask to create a website
class WordGame:
    
    def __init__(self):
        
        self.nlp_model = spacy.load("en_core_web_md")
        #requires download...using spaCy for now
        #self.word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    
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
    def check_random (self, threshold):
        
        #is_alpha makes sure no special letters
        
        print("initiated")
        
        meaningful_words = [word.text for word in self.nlp_model.vocab 
                            if word.has_vector 
                            and len(word.text) > 2
                            and word.is_alpha
                            # and word.is_lower                           
                            #and word.prob >= -15
                            ]

        
        #why is this word array so short??? Is Spacy just a horrible model? The ENTIRE length is 764 for
        #some reason?? it feels like these are all purposefully never used words? is it an issue with the
        #model or my code is accessing it incorrectly?
        
        #issue is the length is zero
        print(f"finished processing {len(meaningful_words)}")
        
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
        
        
        #can set the operation itself as a parameter so one method can handle everything
        sum_vector = word1_vector + word2_vector
        
        
        guess = input(f"{word1} + {word2} = ? \n")
        
        
        doc3 = self.nlp_model(guess)
        guess_vector = doc3[0].vector
        
        #cosine similarity
        similarity = (sum_vector @ guess_vector)/(norm(sum_vector)*norm(guess_vector))
        
        #could change this into a while loop for them to keep guess
        if similarity >= threshold:
            print(f" CORRECT! {word1} + {word2} = {guess}!!!")
        else:
            print("INCORRECT")
            

if __name__ == "__main__":
    main()