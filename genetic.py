import random
import time
import model as M

import printlog as printlog
import datetime
import pdb

date = datetime.date.today()

class GeneticAlgorithm(object):
    
    def __init__(self, genetics):
        self.genetics = genetics
        pass

    def run(self):

        population = self.genetics.initial()
        print("population ",population)
        counter = 0
        while True:
            self.genetics.target = population[counter]
            fits_pops = [(self.genetics.fitness(ch),  ch) for ch in population]
            if self.genetics.check_stop(fits_pops): 
                break
            population = self.next(fits_pops)
            counter += 1
            pass
        return population

    def next(self, fits):
        parents_generator = self.genetics.parents(fits)
        size = len(fits)
        nexts = []
        while len(nexts) < size:
            parents = next(parents_generator)
            parent1, parent2 = parents
            father = parent1
            mother = parent2
            pdb.set_trace()
            cross = random.random() < self.genetics.probability_crossover()
            print("cross "+str(cross))
            children = self.genetics.uniform_crossover(parents) if cross else parents
            for ch in children:
                mutate = random.random() < self.genetics.probability_mutation()
                nexts.append(self.genetics.mutation(ch) if mutate else ch)
                pass
            pass
            print("Next Gen A " + str(nexts))
            if(self.genetics.init_parents):
                
                print("Father " + str(father))
                print("Mother " + str(mother))
                nexts.append(father)
                nexts.append(mother)
                self.genetics.init_parents = False
                print("Next Gen with parent" + str(nexts))

            print("Next Gen B " + str(nexts))
        self.genetics.init_parents = True
        return nexts[0:size]
    pass

class GeneticFunctions(object):
    def probability_crossover(self):
        r"""returns rate of occur crossover(0.0-1.0)"""
        return 1.0

    def probability_mutation(self):
        r"""returns rate of occur mutation(0.0-1.0)"""
        return 0.0

    def initial(self):
        r"""returns list of initial population
        """
        return []

    def fitness(self, chromosome):
        r"""returns domain fitness value of chromosome
        """
        return len(chromosome)

    def check_stop(self, fits_populations):
        r"""stop run if returns True
        - fits_populations: list of (fitness_value, chromosome)
        """
        return False

    def parents(self, fits_populations):
        r"""generator of selected parents
        """
        gen = iter(sorted(fits_populations))
        while True:
            f1, ch1 = next(gen)
            f2, ch2 = next(gen)
            yield (ch1, ch2)
            pass
        return

    def crossover(self, parents):
        r"""breed children
        """
        return parents

    def mutation(self, chromosome):
        r"""mutate chromosome
        """
        return chromosome
    pass

if __name__ == "__main__":


    """
    example: Mapped guess prepared Text
    """
    all_population = {}
    class evolve(GeneticFunctions):
        def __init__(self, target_text,
                     limit=10, size=10,
                     prob_crossover=0.9, prob_mutation=0.2):
            self.counter = 0
            self.init_parents = True
            self.limit = limit
            self.size = size
            self.prob_crossover = prob_crossover
            self.prob_mutation = prob_mutation
            pass

        # GeneticFunctions interface impls
        def probability_crossover(self):
            return self.prob_crossover

        def probability_mutation(self):
            return self.prob_mutation

        def initial(self):
            return [self.random_chromo() for j in range(self.size)]

        def fitness(self, chromo):
            # larger is better, matched == 0
            # return -sum(abs(c - t) for c, t in zip(chromo, self.target))
            PATH_TO_IMAGES = "../images_resized/"
            WEIGHT_DECAY = 1e-4
            print(chromo)
            NUM_LAYERS = chromo[0]
            # FREEZE_LAYERS = int(round(self.target[1]*NUM_LAYERS/100))
            FREEZE_LAYERS = chromo[1]
            LEARNING_RATE = 10**-chromo[2]
            DROP_RATE = chromo[3]/10
            # print("LEARNING_RATE ",LEARNING_RATE)
            # message = ("LEARNING_RATE ",LEARNING_RATE)
            # printlog.printlog(message)
            # print("NUM_LAYERS ",NUM_LAYERS)
            # print("FREEZE_LAYERS ",FREEZE_LAYERS)
            # print("DROP_RATE ",DROP_RATE)
            NUM_OF_EPOCHS = 1
            fitness = 0
            key = '-'.join(str(e) for e in chromo)
            try:
                fitness = all_population[key]
            except:
                preds, aucs, epoch_loss = M.train_cnn(PATH_TO_IMAGES, LEARNING_RATE, WEIGHT_DECAY, NUM_LAYERS, FREEZE_LAYERS, DROP_RATE, chromo, NUM_OF_EPOCHS)
                fitness = -1 * epoch_loss
                all_population[key] = fitness
            # preds, aucs, epoch_loss = M.train_cnn(PATH_TO_IMAGES, self.target[2], WEIGHT_DECAY, self.target[0], self.target[1], self.target[3])
            return fitness

        def check_stop(self, fits_populations):

            self.counter += 1
            # if self.counter % 10 == 0:
            print("fitness and population : ",fits_populations)
            best_match = list(sorted(fits_populations))[-1][1]
            fits = [f for f, ch in fits_populations]
            best = max(fits)
            worst = min(fits)
            ave = sum(fits) / len(fits)
            print(
                "[G %3d] score=(%4f, %4f, %4f): %r" %
                (self.counter, best, ave, worst,
                 self.fitness))

            fits_populations.sort(key=lambda tup: tup[0], reverse = True)
            fittest_fitness = fits_populations[0][0] * -1
            fittest_chromosom = fits_populations[0][1]

            fi= open("results/best_population.txt","a+")
            fi.write(f"{self.counter},{fittest_fitness},{fittest_chromosom}\n")
            fi.close()

            # pass
            return self.counter >= self.limit

        def parents(self, fits_populations):

            while True:
                father = self.tournament(fits_populations)
                mother = self.tournament(fits_populations)
                yield (father, mother)
                pass
            pass

        def crossover(self, parents):

            father, mother = parents
            index1 = random.randint(1, len(self.target) - 2)
            index2 = random.randint(1, len(self.target) - 2)
            if index1 > index2: index1, index2 = index2, index1
            child1 = father[:index1] + mother[index1:index2] + father[index2:]
            child2 = mother[:index1] + father[index1:index2] + mother[index2:]
            return (child1, child2)

        def uniform_crossover(self,parents):

            father, mother = parents
            child1 = []
            child2 = []
            for x in range(len(father)):
                prob = random.uniform(0.0, 1.0)
                print("prob crossover "+str(prob))
                if prob > 0.5:
                    child1.append(mother[x])
                    child2.append(father[x])
                else:
                    child1.append(father[x])
                    child2.append(mother[x])
            return (child1,child2)

            

        def mutation(self, chromosome):

            # index = random.randint(0, len(self.target) - 1)
            # vary = random.randint(-5, 5)
            # mutated = list(chromosome)
            # mutated[index] += vary
            mutated = chromosome
            for cell in range(len(chromosome)):
                prob = random.uniform(0.0, 1.0)
                if prob > 0.5:
                    if cell == 0:
                        vary = random.randint(-5, 5)
                        mutated[cell] += vary
                        if mutated[cell] > 58 or mutated[cell] < 6:
                            mutated[cell] -= vary
                    elif cell == 1:
                        vary = random.randint(-5, 5)
                        old_data = mutated[cell]
                        mutated[cell] += vary
                        if mutated[cell] > mutated[0]:
                            mutated[cell] = old_data
                        if mutated[cell] > 18:
                            mutated[cell] = old_data
                        if mutated[cell] < 0:
                            mutated[cell] = old_data
                    elif cell == 2:
                        vary = random.randint(-1, 1)
                        mutated[cell] += vary
                        if mutated[cell] > 6 or mutated[cell] < 1:
                            mutated[cell] -= vary
                    elif cell == 3:
                        vary = random.randint(-1, 1)
                        mutated[cell] += vary
                        if mutated[cell] > 9 or mutated[cell] < 1:
                            mutated[cell] -= vary
            return mutated

        # internals
        def tournament(self, fits_populations):
            alicef, alice = self.select_random(fits_populations)
            bobf, bob = self.select_random(fits_populations)
            return alice if alicef > bobf else bob

        def select_random(self, fits_populations):
            return fits_populations[random.randint(0, len(fits_populations)-1)]

        def text2chromo(self, text):
            return [ord(ch) for ch in text]

        def chromo2text(self, chromo):
            return "".join(chr(max(1, min(ch, 255))) for ch in chromo)

        def random_chromo(self):
            used_layers = random.sample(range(7, 58), 1)
            # unfreeze_layers = random.sample(range(1, 3), 1)
            # freeze_layers = used_layers[0] - unfreeze_layers[0]
            freeze_layers_random = random.sample(range(0, 18), 1)
            freeze_layers = freeze_layers_random[0]
            learning_rate = random.sample(range(1, 6), 1)
            drop_rate = random.sample(range(1,9),1)

            chromo = [used_layers[0],freeze_layers,learning_rate[0],drop_rate[0]]
            return chromo
            # return [random.randint(1, 255) for i in range(len(self.target))]
        def roulete_wheels(choices):
            max = sum(choices.values())
            pick = random.uniform(0, max)
            current = 0
            for key, value in choices.items():
                current += value
                if current > pick:
                    return fits_populations[key]
        pass
    # GeneticAlgorithm(GuessText("Hello World!")).run()
    loss = 0.5
    
    GeneticAlgorithm(evolve(loss)).run()

pass