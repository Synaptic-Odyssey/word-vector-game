
#from gensim.models import KeyedVectors
import spacy
import random
import wordfreq
from numpy.linalg import norm

def main():
    word_game = WordGame()
    word_game.simple_addition(0.70)


#Export this functionality with FastAPI or Flask to create a website

#TODO: SpaCy might be too inaccurate in terms of word vectors, but codepad might not fit it.
#TODO: rewrite in gensim (more accurate larger dataset) and do it in google collab --> my priority is ensuring that this is accurate
class WordGame:
    
    def __init__(self):
        
        #TODO: can en_core_web_lg be installed on codepad 
        self.nlp_model = spacy.load("en_core_web_lg")
        self.meaningful_words = self.load_meaningful_words(5000)
        
        #print(f"finished processing {len(self.meaningful_words)}")

        #requires download...using spaCy for now
        #self.word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    
    '''
        the challenge is finding two words that add up to one word in a way that makes sense.
        Not just for addition of 2 words, both addition and subtraction of multiple words.
        Do a daily time limit, users must solve as many as possible.
        there is no fixed answer it just has to be close to the two words --> this might not work well
        
        Need to check if 2 words added together to make a 3rd "makes sense", might need a NLP model for this...
    '''
    
    #TODO: also need to check if the word the user enter is invalid
    def simple_addition (self, threshold):

                        
        rand1 = int(random.random()*len(self.meaningful_words)-1)
        
        word1 = self.meaningful_words[rand1]

        
        #TODO: set rand2 to a word vector value close to rand1
        while True:
            rand2 = int(random.random()*len(self.meaningful_words)-1)
            if rand1 != rand2:
                break
            
        word2 = self.meaningful_words[rand2]
        
        #doc is token container --> doc[0] is the token for the first word
        doc1 = self.nlp_model(word1)
        word1_vector = doc1[0].vector
        
        doc2 = self.nlp_model(word2)
        word2_vector = doc2[0].vector
        
        
        #can set the operation itself as a parameter so one method can handle everything
        sum_vector = word1_vector + word2_vector
        
        similarity = 0
        
        
        while True:
            
            guess = input(f"{word1} + {word2} = ? \n")
            doc3 = self.nlp_model(guess)
            guess_vector = doc3[0].vector

            #cosine similarity
            similarity = (sum_vector @ guess_vector)/(norm(sum_vector)*norm(guess_vector))

            if similarity > threshold and guess != word1 and guess != word2:
                
                print(f" \n CORRECT! {word1} + {word2} = {guess}!!!")
                break
            
            print("Incorrect, try again! \n")
            
            
            
    def load_meaningful_words(self, common_count):
                
        common_words = wordfreq.top_n_list("en", common_count)
                
        #self.nlp_model.vocab is only for words encountered during training, which is very little
        #isalpha makes sure no special letters        

        meaningful_words = [word for word in common_words
                            if self.nlp_model(word)[0].has_vector 
                            and len(word) > 2
                            and word.isalpha()
                            # and word.is_lower                           
                            #and word.prob >= -15
                            ]
        
        return meaningful_words    



if __name__ == "__main__":
    main()