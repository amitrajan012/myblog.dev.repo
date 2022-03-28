+++
date = "2018-11-15T12:03:41+01:00"
description = "Commonly used Distributions"
draft = false
tags = ["Probability Distribution", "Bernoulli Distribution", "Binomial Distribution", "Poisson Distribution", "Multinomial Distribution"]
title = "Commonly used Distributions (Part 1)"
topics = ["Probability Distribution"]
+++

</br>
#### The Bernoulli Distribution :

<b>Bernoulli trial</b> is an experiment that can result in two outcomes: <b>success</b> (with probability $p$) and <b>failure</b> (with probability $1-p$). A Bernoulli random variable $X$ can be represented as $X \sim Bernoulli(p)$. It's mean $\mu_X$ and variance $\sigma_X^2$ can be computed as:

$$\mu_X = 0 \times (1-p) + 1 \times p = p$$

$$\sigma_X^2 = (0-p)^2(1-p) + (1-p)^2p = p(1-p)$$

</br>
#### The Binomial Distribution :

When a set of $n$ <b>independent Bernoulli trials</b> are conducted, each with a success probability of $p$, a random variable $X$ which is equal to the number of success in these trials is said to have the <b>binomial distribution</b> with parameters $n$ and $p$ and is represented as $X \sim Bin(n, p)$. <b>Probability mass function</b> of a binomial distribution can be computed as:

$$P(X=x) = (number \ of \ arrangements \ of \ x \ successes \ in \ n \ trials ) \times p^x(1-p)^{n-x}$$

The total number of arrangements of $x$ successes in $n$ trials is given by $n \choose x$. Hence, the PMF can be given as:

$$p(x) = P(X=x) = {n \choose x }p^x(1-p)^{n-x} = \frac{n!}{x!(n-x)!} p^x(1-p)^{n-x}$$

Let the random variables representing the $n$ Bernoulli trials are $Y_1, Y_2, ..., Y_n$. Each of these take a value of either 0 (for failure) or 1 (for success). Hence the binomial RV $X$ representing the number of successes in $n$ trials can be represented as the sum of these $n$ Bernoulli RVs, i.e. $X = Y_1 + Y_2 + ... + Y_n$. The mean and variance of the binomial random variable can be calculated by applying the notion of it's being the linear combination of $n$ independent Bernoulli RVs. They are given as:

$$\mu_X = np$$

$$\sigma_X^2 = np(1-p)$$

The <b>success probability</b> associated with a Bernoulli trial can be estimated as:

$$\widehat{p} = \frac{number \ of \ successes}{number \ of \ trials} = \frac{X}{n}$$

The <b>bias</b> and <b>uncertainty</b> associated with the estimate of the probability can be computed as:

$$Bias = \mu _{\widehat{p}} - p = \mu _{\frac{X}{n}} - p = \frac{\mu_X}{n} - p = \frac{np}{n} - p = 0$$

$$\sigma _{\widehat{p}} = \sigma _{\frac{X}{n}} = \frac{\sigma_X}{n} = \frac{\sqrt{np(1-p)}}{n} = \sqrt{\frac{p(1-p)}{n}}$$

</br>
#### The Poisson Distribution :

The <b>Poisson distribution</b> is an approximation of the binomial distribution when $n$ is large and $p$ is small. When $n$ is large and $p$ is small, the probability mass funaction of the binomial distribution depends on <b>mean</b> $np$, rather than $n$ and $p$. Let $\lambda = np$, the <b>PMF</b> of a binomial distribution is given as:

$$P(X=x) = \frac{n!}{x!(n-x)!} p^x(1-p)^{n-x}$$

Substituting the value of $p$ as $\frac{\lambda}{n}$, we get

$$P(X=x) = \frac{n!}{x!(n-x)!} \bigg(\frac{\lambda}{n}\bigg)^x \bigg(1-\frac{\lambda}{n}\bigg)^{n-x}$$

To get a PMF of Poisson distribution, we need to evaluate the limit of the above quantity as $n \to \infty$. Evaluating the limit and pulling the constants out, we get

$$lim _{n \to \infty}P(X=x) = lim _{n \to \infty} \frac{n!}{x!(n-x)!} \bigg(\frac{\lambda}{n}\bigg)^x \bigg(1-\frac{\lambda}{n}\bigg)^{n-x} = \bigg( \frac{\lambda^x}{x!}\bigg) lim _{n \to \infty} \frac{n!}{(n-x)!} \bigg(\frac{1}{n^x}\bigg)\bigg(1-\frac{\lambda}{n}\bigg)^{n} \bigg(1-\frac{\lambda}{n}\bigg)^{-x}$$

The first two terms in the limit can be evaluated as:

$$lim _{n \to \infty} \frac{n!}{(n-x)!} \bigg(\frac{1}{n^x}\bigg) = lim _{n \to \infty} \frac{n(n-1)(n-2)...(n-x+1)}{n^x} = 1$$

For the evaluation of the limit of the remaining two terms, we need to use the property $e = lim _{k \to \infty} \bigg( 1+\frac{1}{k} \bigg)^k$. Substituting $k = \frac{-n}{\lambda}$ in the third term, we get

$$lim _{n \to \infty} \bigg(1-\frac{\lambda}{n}\bigg)^{n} = lim _{k \to \infty} \bigg(1 + \frac{1}{k}\bigg)^{-k\lambda} = e^{-\lambda}$$

The limit of the fourth term simply evaluates to 1, as $n \to \infty$, $\frac{\lambda}{n} \to 0$. Hence,

$$lim _{n \to \infty}P(X=x) = \bigg( \frac{\lambda^x}{x!}\bigg) e^{-\lambda}$$

Hence, the probability mass funaction of a Poisson distribution is given as:

$$p(x) = P(X=x) = e^{-\lambda} \frac{\lambda^x}{x!}$$

The <b>mean</b> and <b>variance</b> of the Poisson distribution is $\lambda$. An intutive explanation for this is as follows: As the Poisson distreibution is a binomial distribution when the value of $n$ is sufficiently large and $p$ sufficiently small. The mean of the Poisson distribution is same as that of the binomial distribution , which is $np = \lambda$. The variance of binomial distribution is $np(1-p)$ and as $p$ is small, $(1-p) \to 1$, and hence the variance of the Poisson distribution is $np = \lambda$. i.e. For a Poisson distribution,

$$\mu_X = np = \lambda$$

$$\sigma_X^2 = np = \lambda$$

</br>
#### The Multinomial Distribution :

A generalization of the Bernoulli trial is the <b>multinomial trial</b>, which is the process that can result in any of the $k$ outcomes, where $k \geq 2$. If $n$ independent multinomial trials, each with the same $k$ possible outcomes, having the probabilities $p_1, p_2, ..., p_k$ is conducted, and for each outcome $i$, let $X_i$ denotes the number of trials that result in that outcome, the collection $X_1, X_2, ..., X_k$ is said to have a <b>multinomial distribution</b> with the parameters $n, p_1, p_2, ..., p_k$, and is written as $X_1, X_2, ..., X_k \sim MN(n, p_1, p_2, ..., p_k)$. The probability mass funaction of the multinomial trial is given as:

$$p(x_1, x_2, ..., x_k) = P(X_1 = x_1, X_2 = x_2, ..., X_k = x_k) = \frac{n!}{x_1!x_2!...x_k!} p_1^{x_1} p_2^{x_2} ... p_k^{x_k}$$

</br>
#### Reference :

https://www.mheducation.com/highered/product/statistics-engineers-scientists-navidi/M0073401331.html

https://medium.com/@andrew.chamberlain/deriving-the-poisson-distribution-from-the-binomial-distribution-840cc1668239
