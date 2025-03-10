
import spacy
import random
import wordfreq
import numpy as np
from numpy.linalg import norm
import pickle

def main():
    
    iterations = int(input("How many rounds would you like to play? \n"))
    
    word_game = WordGame()
    points = 0
        
    for i in range(iterations):
        
        print(f" \n Onto round {i+1}! \n")
        
        #threshold is 0.42 b/c cosine similarity is between -1 to 1
        outcome = word_game.simple_addition(0.4, 5)
        
        if outcome:
            points += 1
        
    print(f"\n Congrats! You've won {points} out of {iterations} rounds!")


'''
Export this functionality with FastAPI or Flask to create a website
When I do so, ensure each operation of words makes sense for game satisfaction.
The variation (meaning word equations will be different for each person) ensures you can't cheat
Which gives a nice level of satisfaction
the objective is highest score in say 3 minutes (daily reset)
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
TODO: one way to ensure that the operations make sense is to ensure that there are at 
least 2 words with above 60% accuracy or smthg like that, might be slow.
'''

class WordGame:
    
    def __init__(self):
        
        self.nlp_model = spacy.load("en_core_web_lg")
        self.meaningful_words = self.open_meaningful_words()
        
        self.word_vectors = np.array([self.nlp_model(word)[0].vector for word in self.meaningful_words])
        self.vector_norms = np.linalg.norm(self.word_vectors, axis=1)

    
    #Cosine similarity is between -1 and 1
    def simple_addition (self, threshold, num_guesses):
                        
        rand1 = int(random.random()*len(self.meaningful_words)-1)
        
        word1 = self.meaningful_words[rand1]
        
        '''
        essentially the nlp_model takes in a word and converts it to a doc object with tokens,
        which in turn have word bectors. doc[0].vector is the vector of the first word
        '''
        doc1 = self.nlp_model(str(word1))
        word1_vector = doc1[0].vector


        #making sure the words added together are somewhat similar   
        
        dot_products = np.dot(self.word_vectors, word1_vector)
        norms = self.vector_norms * np.linalg.norm(word1_vector)
        cosine_similarities = dot_products/norms
        
        #(cosine_similarities > 0.3) creates own boolean mask, likewise for (cosine_similarities < 0.8)
        close_words = [self.meaningful_words[i] for i in np.where((cosine_similarities > 0.2) & (cosine_similarities < 0.78))[0]]
        
                
        while True:
            rand2 = int(random.random()*len(close_words)-1)
            if rand1 != rand2 and word1 != close_words[rand2]:
                break
            
        word2 = close_words[rand2]
        
        doc2 = self.nlp_model(str(word2))
        word2_vector = doc2[0].vector
        
        
        #can set the operation itself as a parameter so one method can handle everything
        sum_vector = word1_vector + word2_vector
        
        similarity = 0
        sim1 = 0
        sim2 = 0
        
        
        while True:
            
            guess = input(f"{word1} + {word2} = ? \n")  

            if guess == "I give up":
            
                print("round terminated \n")
                answer, similarity_score = self.find_answer(sum_vector, word1, word2)
                print(f"Answer with highest similarity was: '{str(answer)}' with a similarity score of {similarity_score}% \n")
                return False
            
            if num_guesses == 0:
                
                answer, similarity_score = self.find_answer(sum_vector, word1, word2)
                print(f"Answer with highest similarity was: '{str(answer)}' with a similarity score of {similarity_score}% \n")
                print(" Out of guesses! \n")
                return False
            
            doc3 = self.nlp_model(guess)
            
            if doc3[0].has_vector:
                guess_vector = doc3[0].vector

                similarity = self.cosine_similarity(sum_vector, guess_vector)
                #makes sure not just using a synonym of an input
                sim1 = self.cosine_similarity(guess_vector, word1_vector)
                sim2 = self.cosine_similarity(guess_vector, word2_vector)
                
                if similarity > threshold and guess != word1 and guess != word2 and sim1 < 0.75 and sim2 < 0.75:
                
                    print(f" \n CORRECT! {word1} + {word2} = {guess}!!!")
                    print(f"similarity score: {int(similarity*100)}%")
                    
                    return True
                
            else:
                print("not a valid word \n")
                
            
            num_guesses -= 1
            
            incorrect_similarity = self.cosine_similarity(sum_vector, guess_vector)
            print(f"Similarity was below threshold: {int(incorrect_similarity*100)}% \n")
            print("Incorrect, try again! \n")
            
            
#TODO: Programming club challenge! Create a method for subtracting words!


    def load_meaningful_words(self, common_count = 5000):
                
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
    
    
    def save_meaningful_words(self, meaningful_words, filename = "meaningful_words.pk1"):
        
        with open(filename, "wb") as file:
            pickle.dump(meaningful_words, file)
    
            
    def open_meaningful_words(self, filename = "meaningful_words.pk1"):
        
        try:
            with open(filename, "rb") as file:
                print("Opening file... \n")
                return pickle.load(file)
            
        except FileNotFoundError:
            
            print("File not found. Processing words...")
            meaningful_words = self.load_meaningful_words()
            self.save_meaningful_words(meaningful_words, filename)
            
            return meaningful_words


    #TODO: Programming club --> print top 3 answers since top answer is usually synonym of an input
    #TODO: alternatively you could find the most similar answers, compare similarity with inputs and reject if it is too high
    def find_answer(self, result_vector, word1, word2):      
        
        dot_products = np.dot(self.word_vectors, result_vector)
        norms = self.vector_norms * np.linalg.norm(result_vector)
        mask = np.isin(self.meaningful_words, list([word1, word2]))
        cosine_similarities = np.where(mask, -np.inf, dot_products/norms)
        max_index = np.argmax(cosine_similarities)
        similarity_score = int(cosine_similarities[max_index]*100)
        
        return self.meaningful_words[max_index], similarity_score
        


    def cosine_similarity(self, vec1, vec2):
        
        similarity = (vec1 @ vec2)/(norm(vec1)*norm(vec2))
        
        return similarity



if __name__ == "__main__":
    main()



'''
the indices should be the same as meaningful_words, meaning the vector of a word and its 
text share the same indice between both arrays. Will delete these indices permanently for each
run time because repeating words is no fun as well. Should be restored after loading in file again.
Ensure that this permanent deletion causes no issues in the simple addition method.
'''

#code that doesn't work for find answer (the current code shouldn't work but it does for some reason)
#This code was meant to fix the indices being off by 2 but it doesn't work (doesn't acctually delete word1 & word2)

        # i1 = np.nonzero(self.meaningful_words == word1)[0]
        # i2 = np.nonzero(self.meaningful_words == word2)[0]

        # # i1 = np.atleast_1d(np.where(self.meaningful_words == word1)[0])
        # # i2 = np.atleast_1d(np.where(self.meaningful_words == word2)[0])
        
        # deletion = np.concatenate((i1, i2))
        
        # self.meaningful_words = np.delete(self.meaningful_words, deletion)
        # self.word_vectors = np.delete(self.word_vectors, deletion, axis = 0)
        # self.vector_norms = np.delete(self.vector_norms, deletion)
        
        # dot_products = np.dot(self.word_vectors, result_vector)
        # norms = self.vector_norms * np.linalg.norm(result_vector)
        # cosine_similarities = dot_products/norms
                
        # # sorted_similarities = np.sort(cosine_similarities)
        
        # max_index = np.argmax(cosine_similarities)    
            
        # similarity_score = int(cosine_similarities[max_index]*100)
        
        # return self.meaningful_words[max_index], similarity_score

    
#requires download...using spaCy for now
#from gensim.models import KeyedVectors
#self.word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)