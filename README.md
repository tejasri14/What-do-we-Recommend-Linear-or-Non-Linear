# What-do-we-Recommend-Linear-or-Non-Linear

In this work, we compare the performance of the linear and non-linear recommendation systems. We know that deep learning networks can take a lot of time and compute power, so we wanted to do a study to find out if they are really worth it. We investigate how successful each of these methods are in learning the inner product representation of the user and item matrices. With the results of these experiments, we show why it is difficult for a deep learning network to learn the inner product representation and why linear models like the Matrix Factorization are more suitable for the task of recommendation.

# SVD: Singular Value Decomposition

In this section we see how SVD is used as a collaborative filtering approach in recommender systems. SVD breaks down the Rating Matrix R, which maps the ratings given by each user for the movies in different columns, into three matrices P, Σ , and Q.. In the rest of the paper, we assume P is the user embedding and Q is the item embedding.

R = P ΣQT (1)

The P matrix represents the relationship between users and latent factors, Σ is a diagonal matrix that represents the strength of each latent factor and Q matrix represents the similarity between the movies and the latent factors. The latent factors here represent certain characteristics of the movies, for example genre, actors in the movie, length of the movie and so on. The SVD algorithm decreases the dimension of the rating matrix R by mapping the users and movies into a k- dimensional subspace. The diagonal matrix Σ is reduced to Σ by retaining the k largest singular values. The matrices P and Q are also reduced to Pk and Qk to estimate the rating matrix Rk as,

Rk=PkΣkQTk (2)

We then find the square root of the reduced diagonal matrix and multiply that with the reduced user and movie matrix to obtain the latent space for the users and the movies respectively. The rating prediction is then done by multiplying the latent space of the user with that of the movie.
the probability that a user u belong to a certain group of users represented by l who like the same set of movies and the probability that the set of users in the group l like the movie i. And each probability vector Pu is updated according to the rating of the respective users. Similarly the item vector Qi is also updated according to the ratings given for that particular item. 

# Matrix Factorization

One limitation of the SVD method is that it requires all the undefined NaN values in the rating matrix to be defined. So, as the sparsity of the rating matrix increases, the SVD methods gives undefined values for the matrices relating the users and items with the latent factors. To circumvent this issue, we use the method of Matrix Factorization (MF) which decomposes the rating matrix into two matrices P and Q which represent the association between the user and the latent features and the association between the movies and the latent features respectively. These matrices are mapped into a k-dimensional feature space such that the rating matrix R is approximated as the inner product of these matrices.

 The rating prediction for a user u on an item i is then modelled as the product of a row in the user feature matrix pu with the row of the item feature matrix qi as,
 
rˆu,i =qiTpu

These latent vectors pu and qi are learnt by minimizing the cost function which is computed by the summation squared error terms for the known set of ratings and the regularizing term.

# Non-Negative Matrix Factorization

Non-negative Matrix Factorization is another type of matrix factorization covered in this study. This method is imple- mented in an attempt to make the P and Q matrices more interpretable, since the values of these matrices in the matrix factorization method were arbitrary and had a mixture of negative and positive values. In this method, the values of P and Q are set to be positive values constrained to [0,1]. This gives the values of P and Q a probabilistic interpretation. It can be inferred that the latent factors represent a group of users with the same taste and the values of Pu,l and Ql,i represent the probability that a user u belong to a certain group of users represented by l who like the same set of movies and the probability that the set of users in the group l like the movie i. The cost function J for this algorithm is calculated as the summation of the square of the Euclidian distance between the given rating and the predicted rating plus the regularizing terms.

