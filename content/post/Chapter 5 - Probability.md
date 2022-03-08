+++
date = "2018-08-22T09:12:36+01:00"
description = "Think Stats: Chapter 5"
draft = false
tags = ["Think Stats", "Probability", "Monty Hall Problem", "Binomial Distribution", "Bayes's Theorem"]
title = "Think Stats: Chapter 5"
topics = ["Think Stats"]

+++


The belief in which probability is expressed in terms of frequencies is called as <b>frequentism</b>. Frequentists believe that if there is no set of identical trials, there is no probability.

An alternative is <b>Bayesianism</b>, which defines probability as a degree of belief that an event will occur. By this definition, the notion of probability can be applied in almost any cicumstance. One drawback of Bayesian probability is that it depends on person's state of knowledge. People with different information might have different degree of belief about the same event. This is the reason that many people think that Bayesian probability is more subjective than frequentists.

### 5.1 Rules of Probability

Conditional probability is defined as:
$$P(A|B) = \frac{P(A \wedge B)}{P(B)}$$

<b>Exercise 5.1:</b> If I roll two dice and the total is 8, what is the chance that one of the dice is a 6?

<b>Solution:</b> A total of 8 can be received in 5 ways: (2,6),(3,5),(4,4),(5,3),(6,2). Out of these, there are two cases for which 1 dice is 6 and hence the probability is 2/5 = 0.4

<b>Exercise 5.2:</b> If I roll 100 dice, what is the chance of getting all sixes? What is the chance of getting no sixes?

<b>Solution:</b> Chance of getting all sixes is (1/6)^100 and chance of getting no sixes is (5/6)^100.

<b>Exercise 5.3:</b> The following questions are adapted from Mlodinow, The Drunkard’s Walk
 - If a family has two children, what is the chance that they have two girls?

Two children can either be (Boy, Boy), (Boy, Girl) or (Girl, Girl) and hence the probability is 0.33


 - If a family has two children and we know that at least one of them is a girl, what is the chance that they have two girls?

The probability is 0.5 as out of two outcomes (Boy, Girl) and (Girl, Girl), one is favourable.

 - If a family has two children and we know that the older one is a girl, what is the chance that they have two girls?

The probability is 0.5 as out of two outcomes (Girl, Boy) and (Girl, Girl), one is favourable.

 - If a family has two children and we know that at least one of them is a girl named Florida, what is the chance that they have two girls?

Same as above, as the name of the girl being Florida does not affect the outcome.

### 5.2 Monty Hall Problem

The problem statement is: Out of 3 doors, one has a car. Contestant has to choose one of the three doors as the guess  behind which the car is present. The presenter then opens one of the two remaining doors (the one which does not have the car) and gives the contestant the option to switch his choice. The problem is, should the contestant “stick” or “switch” or does it make no difference?

Intutively it seems that it makes no difference but switching doubles the chance of winning. There are three possible scenarios which are: Car is behind Door A or B or C. Suppose the contestant chooses Door A. So his chance of winnig is 1/3. The chance of the car being behind Door B or C is 2/3 and hence after switching the chance of contestant guessing the correct door increases to 2/3.

<b>Exercise 5.4:</b> Write a program that simulates the Monty Hall problem and use it to estimate the probability of winning if you stick and if you switch.


```python
import numpy as np
trials = 1000
number_of_wins_stick = 0
number_of_wins_switch = 0
for i in range(trials):
    winning_door = np.random.randint(3)
    guess = np.random.randint(3)
    if guess == winning_door:
        number_of_wins_stick += 1
    else:
        number_of_wins_switch += 1
print("Probability of winning when sticks: " + str(number_of_wins_stick/trials))
print("Probability of winning when switches: " + str(number_of_wins_switch/trials))
```

    Probability of winning when sticks: 0.338
    Probability of winning when switches: 0.662


<b>Exercise 5.5:</b> To understand the Monty Hall problem, it is important to realize that by deciding which door to open, Monty is giving you information. To see why this matters, imagine the case where Monty doesn’t know where the prizes are, so he chooses Door B or C at random.


```python
import random
trials = 1000
number_of_wins_stick = 0
number_of_wins_switch = 0
for i in range(trials):
    doors = [0, 1, 2]
    winning_door = np.random.randint(3)
    guess = np.random.randint(3)
    doors.remove(guess)
    result_switch = random.choice(doors) # As one door is removed randomly
    if guess == winning_door:
        number_of_wins_stick += 1
    elif result_switch == winning_door:
        number_of_wins_switch += 1
print("Probability of winning when sticks: " + str(number_of_wins_stick/trials))
print("Probability of winning when switches: " + str(number_of_wins_switch/trials))
```

    Probability of winning when sticks: 0.335
    Probability of winning when switches: 0.348


<b>Exercise 5.7</b> If you go to a dance where partners are paired up randomly, what percentage of opposite sex couples will you see where the woman is taller than the man? The distribution of height is roughly normal with mean = 178cm and variance = 59.4cm for men and mean = 163cm and variance = 52.8cm for women.


```python
from scipy.stats import norm
import math
trials = 10000
mean_men = 163
std_men = math.sqrt(59.4)
mean_women = 163
std_women = math.sqrt(52.8)
favourable_outcomes = 0
for i in range(trials):
    men = norm.rvs(loc=mean_men, scale=std_men, size=1, random_state=None)[0]
    women = norm.rvs(loc=mean_women, scale=std_women, size=1, random_state=None)[0]
    if women > men:
        favourable_outcomes += 1
print("Perccentage couples in which women is taller than the men is: " +str((favourable_outcomes*100)/trials) + "%")
```

    Perccentage couples in which women is taller than the men is: 49.98%


<b>Exercise 5.8:</b> If I roll 2 dice, what is the chance of rolling at least one six?

<b>Solution: </b> Total number of outcomes is 36 when we roll two dice and out of those there are 25 outcomes of getting no 6. Hence favourable outcomes are (36-25) = 11 and probability is 11/36.

### 5.5 Binomial Distribution

<b>Binomial Distribution</b> has PMF defined as (where n is the number of trials, p is the probability of success and k is the number of success):
$$PMF(k) = \binom{n}{k}p^{k}(1-p)^{n-k}$$

<b>Exercise 5.10:</b> If you flip a coin 100 times, you expect about 50 heads, but what is the probability of getting exactly 50 heads?


```python
import math

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

probability = nCr(100, 50) * math.pow(0.5, 50) * math.pow(0.5, 50)
print("Probability is: " +str(probability))
```

    Probability is: 0.07958923738717877


### 5.6 Streaks and hot spots

In real scenarios of random processes, there is no such thing as streak. A related phenomenon is the clustering illusion, which is the tendency to see clusters in random spatial pattern.

<b>Exercise 5.11:</b> If there are 10 players in a basketball game and each one takes 15 shots during the course of the game, and each shot has a 50% probability of going in, what is the probability that you will see, in a given game, at least one player who hits 10 winning shots in a row? If you watch a season of 82 games, what are the chances you will see at least one streak of 10 hits or misses?


```python
from random import shuffle

trials = 10000
favourable_outcome = 0
players = 10

for x in range(trials):
    for p in range(players):
        shots = np.random.randint(2, size=15) # As every shot has 50% probability of going in
        # Find winning streak
        winning_streak = False
        for i in range(len(shots)-10):
            window = shots[i:i+10]
            if sum(window) == 10:
                winning_streak = True
            if winning_streak:
                break
        if winning_streak:
            favourable_outcome += 1
            break

print("Probability of seeing 10 winning shots in a game: " +str(favourable_outcome/trials))

seasons = 1000
favourable_outcome = 0
players = 10
games = 82

for x in range(seasons):
    for g in range(games):
        streak = False
        for p in range(players):
            shots = np.random.randint(2, size=15) # As every shot has 50% probability of going in
            # Find streaks of hits and misses
            for i in range(len(shots)-10):
                window = shots[i:i+10]
                if sum(window) == 10 or sum(window) == 0:
                    streak = True
                if streak:
                    break
            if streak:
                    break
        if streak:
            favourable_outcome += 1
            break

print("Probability of seeing a streak in a season is: " +str(favourable_outcome/seasons))
```

    Probability of seeing 10 winning shots in a game: 0.0285
    Probability of seeing a streak in a season is: 0.994


### 5.7 Bayes's Theorem

Bayes's Theorem states that:
$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$$

Suppose H is a hypothesis and E an event, then from Bayes's Theorem:

$$P(H|E) = P(H)\frac{P(E|H)}{P(E)}$$

This equation says that the probability of hypothesis H after the occurrence of event E is the product of the probability of H before the occurrence of event E and the ratio of the probability of occurrence of the event given the hypothesis has occurred and the total probability of the occurrence of event. This gives a knowledge about how the <b>probability of a hypothesis get updated overtime, usually in the light of new event/evidence.</b> Here, P(H) is called the <b>prior probability</b>, P(H|E) the <b>posterior probability</b>, P(E|H) the <b>likelihood of the event</b> and P(E) is the <b>normalizing constant</b>.

<b>Exercise 5.15:</b> Suppose there are two full bowls of cookies. Bowl 1 has 10 chocolate chip and 30 plain cookies, while Bowl 2 has 20 of each. Our friend Fred picks a bowl at random, and then picks a cookie at random. The cookie turns out to be a plain one. How probable is it that Fred picked it out of Bowl 1?

<b>Solution: </b>We have to find P(Bowl 1 | Plain), which can be evaluated as:
$$P(Bowl 1| Plain) = P(Bowl 1)\frac{P(Plain|Bowl 1)}{P(Plain)}$$

Here, P(Bowl 1) = 0.5, P(Plain | Bowl 1) = 30/40 = 0.75 and P(Plain) = 50/80 = 0.625

Hence, P(Bowl 1 | Plain) = 0.6
