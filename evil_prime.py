#Shephard 'MagicianGirl' phoenixAiden 19/04/2019
""" The Evil Prime!
In this task you are required to write a basic python script that checks all the numbers in the set of [0 -> 99999] for “Evil Primes”.
 “Evil Primes” in this task are numbers which are prime numbers and contain the digits “666” in them. """
import math


#Find Prime number
def find_prime():
    primes = []
    evil_primes = []
    for n in range(2, 99999): # Started range from 2 because 1 is a Unit
        max_d = math.floor(math.sqrt(n))
        for d in range(2, 1 + max_d): # Range exclusive since every number is devisible by its self,(2 -> sqrt(n)) reduces number of divisors for efficiency 
            if n % d == 0:
                break
        else:
            primes.append(n) 
    return [ev for ev in primes if "666" in str(ev)]

print(find_prime()) #Print the evil primes.
   
   
