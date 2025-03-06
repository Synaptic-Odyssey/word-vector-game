
#from gensim.models import KeyedVectors
import spacy
import random
import wordfreq
from numpy.linalg import norm

def main():
    
    iterations = int(input("How many rounds would you like to play? \n"))
    
    word_game = WordGame()
    points = 0
    
    #TODO: maybe only give people 5 guesses
    
    for i in range(iterations):
        
        print(f" \n Onto round {i+1}! \n")
        outcome = word_game.simple_addition(0.52)
        
        if outcome:
            points += 1
        
    print(f"\n Congrats! You've won {points} out of {iterations} rounds!")


'''
Export this functionality with FastAPI or Flask to create a website
When I do so, ensure each operation of words makes sense for game satisfaction.
At the very least, keep the I give up operation
The variation (meaning word equations will be different for each person) ensures you can't cheat
Which gives a nice level of satisfaction
the objective is highest score in say 3 minutes (daily reset)
A little scuffed but I won't check if the operation makes sense.
As for other operations, if they are too complex it also wouldn't make sense. Adding 3 words makes more
sense than subtracting. But it would still work. In fact I should make everything one method, have the 
number of inputs and specific operations as the parameter.
'''

'''
TODO: other operations for words, penalization for guessing, timer
TODO: for actual website new word combination is too slow. Optimize + preload all of them so they don't
    take up the user's timer. Preloading is possible if number of rounds is fixed. But it isn't, number of 
    rounds is dependent on time. Could preload like 10 and have threads run in the background while the
    user is playing.
TODO: have website built before tuesday?
'''

class WordGame:
    
    def __init__(self):
        
        #TODO: can en_core_web_lg be installed on codepad? 
        self.nlp_model = spacy.load("en_core_web_lg")
        self.meaningful_words = self.load_meaningful_words(5000)
        
        #print(f"finished processing {len(self.meaningful_words)}")

        #requires download...using spaCy for now
        #self.word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

    
    def simple_addition (self, threshold):
                        
        rand1 = int(random.random()*len(self.meaningful_words)-1)
        
        word1 = self.meaningful_words[rand1]
        
        '''
        essentially the nlp_model takes in a word and converts it to a doc object with tokens,
        which in turn have word bectors. doc[0].vector is the vector of the first word
        '''
        doc1 = self.nlp_model(word1)
        word1_vector = doc1[0].vector

        #making sure the words added together are somewhat similar
        
        close_words = []
        
        for word in self.meaningful_words:
            
            temp_doc = self.nlp_model(word)
            temp_vector = temp_doc[0].vector
            
            if self.cosine_similarity(word1_vector, temp_vector) > 0.37:
                
                close_words.append(word)
        
        #print(f"close words length: {len(close_words)} \n")
        #potential issue is that close_words could be empty, but I'm betting that's not going to happen
        
        
        while True:
            rand2 = int(random.random()*len(close_words)-1)
            if rand1 != rand2 and word1 != close_words[rand2]:
                break
            
        word2 = close_words[rand2]
        
        doc2 = self.nlp_model(word2)
        word2_vector = doc2[0].vector
        
        
        #can set the operation itself as a parameter so one method can handle everything
        sum_vector = word1_vector + word2_vector
        
        similarity = 0
        
        
        while True:
            
            guess = input(f"{word1} + {word2} = ? \n")  

            if guess == "I give up":
                
                return False
                #break
            
            doc3 = self.nlp_model(guess)
            
            if doc3[0].has_vector:
                guess_vector = doc3[0].vector

                similarity = self.cosine_similarity(sum_vector, guess_vector)
                
            else:
                print("not a valid word \n")
            
            #don't need an else because similarity won't be updated to a usable value

            if similarity > threshold and guess != word1 and guess != word2:
                
                print(f" \n CORRECT! {word1} + {word2} = {guess}!!!")
                
                return True
                #break
            
            print("Incorrect, try again! \n")
            
            
#TODO: Programming club challenge! Create a method for subtracting words!

            
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



    def cosine_similarity(self, vec1, vec2):
        
        similarity = (vec1 @ vec2)/(norm(vec1)*norm(vec2))
        
        return similarity



if __name__ == "__main__":
    main()