## Markov Chains

We have a set of stated, $\mathcal{S}$ = {${s_1, s_2, ..., s_m} $}. A Markov process starts in one of these states and moves successively from one state to another. Each move is called a _step_ . If the chain is currently in state $s_i$, then it moves to state $s_j$ at the next step with a probability denoted by $p_{ij}$, and this probability does not depend upon which states the chain was in before the current state.These probabilities $p_{ij}$ are called **transition probabilities**. The process can remain in the state it is in, and this occurs with probability $p_{ii}$ . Initially a probability is defined by specifying a particular state as the starting state.



Say, we stay in *Patna* (Capital of the State of Bihar, India), and at my home we usually eat either *Roti* ($\text{Ro}$), Rice ($\text{Ri}$), *Parantha* ($\text{Pa}$)) or Bread ($\text{Br}$) for dinner. There are also some days when we eat one particular dinner on two or three or more consecutive days.

Now, we define transition probabilities of using a **Transition Matrix** which is given as, 
$$
\text{T}=\begin{array}{c c} &
\begin{array}{c c c} \text{Ri} & \text{Ro} & \text{Pa} & \text{Br} \\
\end{array} 
\\
\begin{array}{c c c}
\text{Ri} \\
\text{Ro} \\
\text{Pa} \\
\text{Br}\\
\end{array} 
&
\left[
\begin{array}{c c c}
\frac{1}{2}      & \frac{1}{4} & \frac{1}{4} &  0 \\
    0      & \frac{1}{4} & \frac{1}{4} &  \frac{1}{2} \\
    0      & \frac{1}{2} & \frac{1}{4} &  \frac{1}{4} \\
    \frac{1}{2}      & 0 & \frac{1}{4} &  \frac{1}{4} \\
\end{array}
\right] 
\end{array}
$$
From the above matrix, it can be seen that $P_{(Rice \implies Rice)}=\frac{1}{2}$ and similarly, $P_{(Roti \implies Parantha)}=\frac{1}{4}$. 

In the similar lines of the above concept, a **Transition Diagram** can also be drawn.

Let us dig further into the concept now. Imagine today we eat *Rice* at home and two days from today we eat *Bread*. This is written as $P^{2}_{(Rice \implies Bread)}$. We see that if we ate *Rice* today then the event of us eating *Bread* two days from today is a disjoint union of the following, 

1. We ate *Rice* tomorrow and Bread day after.
2. We ate something else tomorrow and Bread day after.
3. We ate Bread tomorrow and Bread day after.

Therefore, we can write $P^{2}_{(Rice \implies Bread)}$ as , 
$$
P^{2}_{(Rice \implies Bread)} = [P_{(Rice \implies Rice)}]\cdot[P_{(Rice \implies Bread)}]+[P_{(Rice \implies Roti)}]\cdot[P_{(Roti \implies Bread)}]+ \\ [P_{(Rice \implies Parantha)}]\cdot[P_{(Parantha \implies Bread)}]+[P_{(Rice \implies Bread)}]\cdot[P_{(Bread
\implies Bread)}]
$$
The has pointed towards a more generalized concept of **Dot product of two vectors**.

Considering there are $r$ states in the Markov chain,
$$
P^{2}_{ij}=\sum_{k=1}^{r}P_{ik}P_{kj}
$$
This study was for $n=2$ , meaning we are yet to define this in a more generalized manner. Obviously, looking at the present scenario of high-end computing software packages available, this attempt seems to be a waste of efforts, but then knowing the math is always good fun! 

Before doing that, we will again go into the basic definition as defined above and try to make an **iterative** approach into the problem. 

We have assumed that our meal starts at state *Rice*, so, for us, 
$$
\text{p}_0=\begin{bmatrix}
1 \\
    0  \\
    0 \\
    0\\
\end{bmatrix}
$$
$\text{p}_0$ is the state of our system at $0^{th}$ state *or* the beginning of the system.

Similarly, if we want to find $\text{p}_1$ we will do it as, 
$$
\text{p}_1 = \text{T} \cdot \text{p}_0
$$
In the similar lines, 
$$
\boxed{\text{p}_5 = \text{T} \cdot \text{p}_4 = \text{T}^{2} \cdot \text{p}_3 = \text{T}^3 \cdot \text{p}_2 = \text{T}^4 \cdot \text{p}_1 = \text{T}^{5} \cdot \text{p}_0}
$$
This means, for generalization’s sake, 
$$
\text{p}_n = \text{T}^{n}\cdot \text{p}_0
$$
Using some very basic **Linear Algebra** methods to compute $\text{T}^n$,

Recall : $\text{T}=\mathcal{u}\cdot D \cdot\mathcal{u}^{-1}$

Here, $D$ is a Diagonal matrix and $\mathcal{u}$ is a matrix whose columns correspond to the Eigen-vectors of $\text{T}$.

**I leave the computation to your own practice.**

There for, 
$$
\text{T}^n = \mathcal{u} \cdot D^n . \mathcal{u}^{-1}
$$
Since, $D$ is a diagonal matrix, which is obviously of form, 
$$
D = \begin{bmatrix}
a & 0 & 0 & 0 \\ 0 & b & 0 & 0 \\ 0 & 0 & c & 0 \\ 0 & 0 & 0 & d
\end{bmatrix}
$$
Therefore, $D^n$ will be,
$$
D = \begin{bmatrix}
a^n & 0 & 0 & 0 \\ 0 & b^n & 0 & 0 \\ 0 & 0 & c^n & 0 \\ 0 & 0 & 0 & d^n
\end{bmatrix}
$$
Hence, we can easily compute $T^n$.

The Python implementation of this concept is relatively simple once we understand the math behind it. 

All the best with that.

Cheers!

Resources:

1. [Matrix Diagonalization](https://www.youtube.com/watch?v=Sf91gDhVZWU)

2. [Eigen Values and Eigen Vectors](https://www.youtube.com/watch?v=G4N8vJpf7hM&vl=en)

3. [Markov Chains](https://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf)

4. [Markov Chains](http://web.math.ku.dk/noter/filer/stoknoter.pdf)
