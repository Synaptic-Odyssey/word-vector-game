
#from gensim.models import KeyedVectors
import spacy
import random
import wordfreq
import numpy as np
from numpy.linalg import norm

def main():
    
    iterations = int(input("How many rounds would you like to play? \n"))
    
    print("processing words...")
    word_game = WordGame()
    points = 0
        
    for i in range(iterations):
        
        print(f" \n Onto round {i+1}! \n")
        
        outcome = word_game.simple_addition(0.55, 5)
        
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

Or I could preload 10 and see how fast people are able to complete them. The issue is that sometimes
people will get an odd combination that will seem unfair. However this reduces the incentive to just
give up on harder combinations in favor for super easy combinations, furthermore it will add a tangible
penalty to guessing instead of the I can just move on mentality.

However the feeling of time running out is also a really good motivator!
What makes Wordle better however is the shared sense of struggle as everyone has the same word

ISSUE: No way of showing "correct answer" unless maybe iterating through freq words until there is a 
maximum similarity
'''

'''
TODO: other operations for words, timer
TODO: Increase accuracy! maybe use a larger model like gensim. Problems + annoying = nuisance being incorrect
    will definitely make the experience bothersome.
TODO: have website built before tuesday?
TODO: one of the problems is that the solution is always one of the inputs
'''

class WordGame:
    
    def __init__(self):
        
        #TODO: can en_core_web_lg be installed on codepad? 
        self.nlp_model = spacy.load("en_core_web_lg")
        self.meaningful_words = self.load_meaningful_words(5000)
        self.word_vectors = np.array([self.nlp_model(word)[0].vector for word in self.meaningful_words])
        self.vector_norms = np.linalg.norm(self.word_vectors, axis=1)
        
        #print(f"finished processing {len(self.meaningful_words)}")

        #requires download...using spaCy for now
        #self.word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

    
    
    def simple_addition (self, threshold, num_guesses):
                        
        rand1 = int(random.random()*len(self.meaningful_words)-1)
        
        word1 = self.meaningful_words[rand1]
        
        '''
        essentially the nlp_model takes in a word and converts it to a doc object with tokens,
        which in turn have word bectors. doc[0].vector is the vector of the first word
        '''
        doc1 = self.nlp_model(word1)
        word1_vector = doc1[0].vector


        #making sure the words added together are somewhat similar   
        
        dot_products = np.dot(self.word_vectors, word1_vector)
        norms = self.vector_norms * np.linalg.norm(word1_vector)
        cosine_similarities = dot_products/norms
        
        #(cosine_similarities > 0.37) creates own boolean mask, likewise for (cosine_similarities < 0.78)
        close_words = [self.meaningful_words[i] for i in np.where((cosine_similarities > 0.37) & (cosine_similarities < 0.85))[0]]
        
                
        while True:
            rand2 = int(random.random()*len(close_words)-1)
            if rand1 != rand2 and word1 != close_words[rand2]:
                break
            
        word2 = close_words[rand2]
        
        doc2 = self.nlp_model(word2)
        word2_vector = doc2[0].vector
        
        
        #can set the operation itself as a parameter so one method can handle everything
        sum_vector = word1_vector + word2_vector
        
        similarity, sim1, sim2 = 0
        
        
        
        while True:
            
            guess = input(f"{word1} + {word2} = ? \n")  

            if guess == "I give up":
                
                print("round terminated \n")
                print("Correct answer was: " + str(self.find_answer(sum_vector, word1, word2)) + "\n")
                return False
            
            if num_guesses == 0:
                
                print("Correct answer was: " + str(self.find_answer(sum_vector, word1, word2)) + "\n")
                print(" Out of guesses! \n")
                return False
            
            doc3 = self.nlp_model(guess)
            
            if doc3[0].has_vector:
                guess_vector = doc3[0].vector

                similarity = self.cosine_similarity(sum_vector, guess_vector)
                #makes sure not just using a synonym of an input
                sim1 = self.cosine_similarity(guess_vector, word1_vector)
                sim2 = self.cosine_similarity(guess_vector, word2_vector)
                
            else:
                print("not a valid word \n")
            
            #don't need an else because similarity won't be updated to a usable value
            
            
            if similarity > threshold and guess != word1 and guess != word2 and sim1 < 0.85 and sim2 < 0.85:
                
                print(f" \n CORRECT! {word1} + {word2} = {guess}!!!")
                
                return True
                #break
            
            num_guesses -= 1
            
            print("Incorrect, try again! \n")
            
            
#TODO: Programming club challenge! Create a method for subtracting words!

#TODO: save this as a file so I can load these on directly in the future           
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


    #TODO: cannot be the cosine product of the word itself
    def find_answer(self, result_vector, word1, word2):      
        
        print("computing answer")
        
        #this is a band aid solution --> takes too long!
        word_vectors = np.array([self.nlp_model(word)[0].vector for word in self.meaningful_words 
                                if word != word1 and word != word2])

        dot_products = np.dot(word_vectors, result_vector)
        norms = np.linalg.norm(word_vectors, axis=1) * np.linalg.norm(result_vector)
        cosine_similarities = dot_products/norms
        
        return self.meaningful_words[np.argmax(cosine_similarities)]




    def cosine_similarity(self, vec1, vec2):
        
        similarity = (vec1 @ vec2)/(norm(vec1)*norm(vec2))
        
        return similarity



if __name__ == "__main__":
    main()