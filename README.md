# What-do-we-Recommend-Linear-or-Non-Linear

In this work, we compare the performance of the linear and non-linear recommendation systems. We know that deep learning networks can take a lot of time and compute power, so we wanted to do a study to find out if they are really worth it. We investigate how successful each of these methods are in learning the inner product representation of the user and item matrices. With the results of these experiments, we show why it is difficult for a deep learning network to learn the inner product representation and why linear models like the Matrix Factorization are more suitable for the task of recommendation.

## SVD: Singular Value Decomposition

In this section we see how SVD is used as a collaborative filtering approach in recommender systems. SVD breaks down the Rating Matrix R, which maps the ratings given by each user for the movies in different columns, into three matrices P, Σ , and Q.. In the rest of the paper, we assume P is the user embedding and Q is the item embedding.

R = P ΣQT

The P matrix represents the relationship between users and latent factors, Σ is a diagonal matrix that represents the strength of each latent factor and Q matrix represents the similarity between the movies and the latent factors. The latent factors here represent certain characteristics of the movies, for example genre, actors in the movie, length of the movie and so on. The SVD algorithm decreases the dimension of the rating matrix R by mapping the users and movies into a k- dimensional subspace. The diagonal matrix Σ is reduced to Σ by retaining the k largest singular values. The matrices P and Q are also reduced to Pk and Qk to estimate the rating matrix Rk as,

Rk = PkΣkQTk

We then find the square root of the reduced diagonal matrix and multiply that with the reduced user and movie matrix to obtain the latent space for the users and the movies respectively. The rating prediction is then done by multiplying the latent space of the user with that of the movie.
the probability that a user u belong to a certain group of users represented by l who like the same set of movies and the probability that the set of users in the group l like the movie i. And each probability vector Pu is updated according to the rating of the respective users. Similarly the item vector Qi is also updated according to the ratings given for that particular item. 

# Matrix Factorization

One limitation of the SVD method is that it requires all the undefined NaN values in the rating matrix to be defined. So, as the sparsity of the rating matrix increases, the SVD methods gives undefined values for the matrices relating the users and items with the latent factors. To circumvent this issue, we use the method of Matrix Factorization (MF) which decomposes the rating matrix into two matrices P and Q which represent the association between the user and the latent features and the association between the movies and the latent features respectively. These matrices are mapped into a k-dimensional feature space such that the rating matrix R is approximated as the inner product of these matrices.

 The rating prediction for a user u on an item i is then modelled as the product of a row in the user feature matrix pu with the row of the item feature matrix qi as,
 
rˆu,i = qiTpu

These latent vectors pu and qi are learnt by minimizing the cost function which is computed by the summation squared error terms for the known set of ratings and the regularizing term.

## Non-Negative Matrix Factorization

Non-negative Matrix Factorization is another type of matrix factorization covered in this study. This method is imple- mented in an attempt to make the P and Q matrices more interpretable, since the values of these matrices in the matrix factorization method were arbitrary and had a mixture of negative and positive values. In this method, the values of P and Q are set to be positive values constrained to [0,1]. This gives the values of P and Q a probabilistic interpretation. It can be inferred that the latent factors represent a group of users with the same taste and the values of Pu,l and Ql,i represent the probability that a user u belong to a certain group of users represented by l who like the same set of movies and the probability that the set of users in the group l like the movie i. The cost function J for this algorithm is calculated as the summation of the square of the Euclidian distance between the given rating and the predicted rating plus the regularizing terms.


## EXPERIMENTS

To compare the performance of linear and non-linear, we choose a method from each type. For linear models, we experiment with the Matrix Factorization (MF) method and for the non-linear models we experiment with Neural Collabo- rative Filtering (NCF) method. We start with varying the input embedding size of the user and item vectors as experimented in. This experiment give us a insight about how the model is dealing with a denser representation of data. The assumption here is that the user and item matrices become denser as the embedding size increases as now the embedding gives a larger output for the same user compared to a smaller embedding size. This leads to a denser user and item vectors. We train the Matrix Factorization method by minimizing L2 regularization and stochastic gradient descent. We also gradually increase the number of users as input, to assess how the methods do with varying amount of input data.

### A. Varying Embedding Size

To assess the performance of the methods with varying embedding size, we use the Hit Ratio method on the top 10 recommendations of the models. We also compute the number training parameters in NCF with varying embedding size. We do this to analyze how many training parameters the NCF model has to learn with varying embedding size. The motivation behind this experiment is to see how the models are performing when the embedding size is increased. When the embedding size is increased, the model gets a denser input, but this also means that the model has to work with more data and training parameters and that might affect it’s performance. 

### B. Varying input data

For the second part of the experiments, we gradually varied the number of users as inputs to both MF and NCF models. We calculated the test RMSE [10] for both the models and plotted the graph that shows the difference in test RMSE for varying input data.

## Results 
![plot](./images/RMSE%20vs.%20no.%20of%20users.png)

![plot](./images/RMSE%20vs.%20users%20for%20varying%20embedding%20dimensions.png)

![plot](./images/RMSE%20vs.%20varying%20embedding%20dimensions.png)

![plot](./images/Trainable%20Prameters%20vs.%20Embedding%20Dimensions.png)
