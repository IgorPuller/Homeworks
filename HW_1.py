#!/usr/bin/env python
# coding: utf-8

# # Introduction to Machine Learning via Nearest Neighbours
# 

# ## Part 1 - Implement k-Nearest Neighbours (kNN) - 30 points
# 
# ### Note:
# This exercise closely follows the post by Scott Fortmann-Roe about bias-variance tradeoff (see references below). It is recommended that you think about each of the questions before referring to that essay.
# 
# In this exercise you will get familiar with a **non-parapmetric** learning algorithm called k-Nearest Neighbours (kNN), and will implement it. You will then analyse the bias-variance tradeoff and try to come up with the optimal kNN classifier for the given data.
# 
# For this exercise we will use hypothetical and artificial generated data.
# ### Nearest Neighbours
# The kNN algorithm is simple - given a labeled sample set data, and a new sample, predict the label of this sample by using majority vote (or averaging) over the labels of the k-nearest neighbour of the new data in the sample set.
# 
# For this task, assume each data point is an n-dimensional point in $\mathbb{R}^n$, and each label is either 0 or 1.
# 
# Implement a class called KNNClassifier and two methods:
# - `fit`: should recieve the training data (an array of shape [n_samples, n_features]) and their labels (array of shape [n_samples]). 
# - `predict`: should recieve a set of data (an array of shape [n_samples, n_features]) and **return** their predicted labels (array of shape [n_samples]).
# 
# Use simple Euclidean distance to measure the distance between two points. In case two points in the training have the same distance from a given point to predict, the nearest would be the one appearing first in the training set. 
# 
# Use majority vote between all kNN of a point to predict its label. In case the vote of all kNN is tied, you may predict whichever label you wish.
# 
# You may look up [sklearn.neighbors.KNeighborsClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) for reference.
# 
# * Bonus for nurdes: use kd-tree for efficiency, see [nearest neighbour search in wiki](https://en.wikipedia.org/wiki/Nearest_neighbor_search)
# 
# * Bonus for lazy nurdes: use [scipy's kd implementation](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.spatial.KDTree.html)

# In[2]:


class kNNClassifier:
  def __init__(self, n_neighbors):
    self.n_neighbors = n_neighbors
  def fit(self, X, y):
    self.X = X
    self.y = y
  def predict(self, X):
    distance = ((X-np.expand_dims(self.X, axis=1))**2)
    distance = np.sqrt(np.sum(distance, axis=2))
    n_row, n_col = distance.shape
    nn = np.array([self.y[np.argsort(distance[:,col])[:self.n_neighbors]] for col in range(n_col)])
    def most_common(array):
        val, counts = np.unique(array, return_counts=True)  
        return val[np.argmax(counts)]
    pred_y = np.array([most_common(nn[row,:]) for row in range(n_col)])
    return pred_y
      


# ## Part 2 - Learn and evaluate kNN algorithm on artificial data
# 
# kNN is a **non-parametric** in the sense that no inner parameter of the model is learned by the sample training set (or maybe you could say that the number of parameters increases with the size of the sample set). However, the number of neighbours **k is considered a hyper-parameter**, and choosing the optimal value for it, is choosing the balance between bias and variance as discussed in class.
# 
# 
# 

# ### An applied example: voter party registration
# 
# In this example, each voter is described by a vector of two features $(x_0, x_1)$, where $x_0$ describes how wealthy that voter is, and $x_1$ describes how religious the voter is. Label $y=1$ represents a Republican voter, and $y=-1$ represents a Democrat voter.
# 
# Use the given function `generate_data(m)` to create m samples with m labels. The labels are created using the following function:
# 
# $$
# y = \text{sign}\left(x_1 - 0.1\times((x_0-5)^3-x_0^2+(x_0 − 6)^2+80)\right)
# $$
# 
# and then a small fraction of the labels (chosen randomly, up to 10%) are flipped to represent unknown effect of features which are not measured. The sign of 0 is defined as 1 for this case.
# 
# 
# Below is an example of generating 500 samples, and plotting them.

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

def true_boundary_voting_pred(wealth, religiousness):
  return religiousness-0.1*((wealth-5)**3-wealth**2+(wealth-6)**2+80)

def generate_data(m, seed=None):
  # if seed is not None, this function will always generate the same data
  np.random.seed(seed) 
  
  X = np.random.uniform(low=0.0, high=10.0, size=(m,2))
  y = np.sign(true_boundary_voting_pred(X[:,0], X[:,1]))
  y[y==0] = 1
  samples_to_flip = np.random.randint(0,m//10)
  flip_ind = np.random.choice(m, samples_to_flip, replace=False)
  y[flip_ind] = -y[flip_ind]
  return X, y

def plot_labeled_data(X, y, no_titles=False):
  republicans = (y==1)
  democrats = (y==-1)
  plt.scatter(X[republicans,0], X[republicans,1], c='r')
  plt.scatter(X[democrats,0], X[democrats,1], c='b')
  if not no_titles:
    plt.xlabel('Wealth')
    plt.ylabel('Religiousness')
    plt.title('Red circles represent Republicans, Blues Democrats')
    
  plt.xlim([0, 10]);
  plt.ylim([0, 10]);
  plt.plot(np.linspace(0,10,1000), -true_boundary_voting_pred(np.linspace(0,10,1000), np.zeros(1000)), linewidth=2, c='k');


# In[4]:


# Play this several times to see different sampling sets
X, y = generate_data(m=500)
plot_labeled_data(X, y)


# 
# ### 1. Analyse the properties of kNN - 30 points
# Generate multiple sample data sets of size 500 (use the given function above), and plot the decision plane for increasing values of k (e.g.: 1, 3, 5, 11, 21, 51, 99).
# 
# The decision plane should cover the range [0,10] for both axes, coloring the patches that would be classified as Republicans or Democrats in two colors. It should look something like this:
# 
# ![decision plane](https://doc-14-14-docs.googleusercontent.com/docs/securesc/flg80o8vb463a3nd3i6da8hemig5me1b/hvii8ll4dscju8o17vuo2aab9aei8hgr/1543334400000/11934753179242311747/03422859225809857490/1chmyojft_R6ftfBhoPZuGN9AykyUS-cw?e=view&nonce=4i4j8lssjk6kc&user=03422859225809857490&hash=k9ogjg94ssot1vocu8uoeg4okkeekg6f)
# 
# https://drive.google.com/file/d/1chmyojft_R6ftfBhoPZuGN9AykyUS-cw/view?usp=sharing
# 
# Answer the following questions:
# - How is the decision plain affected from changes in the sample set, as a function of k?
# - Can you describe when do you underfit or overfit the data? 
# - How does the complexity of this model depends on k?
# 
# * Bonus for nurdes:
# Use interactive slider for k to see the effect [see [interact](https://ipywidgets.readthedocs.io/en/stable/examples/Using%20Interact.html)]

# In[8]:


from ipywidgets import interact

X, y = generate_data(m=500)

def decision_plane(k):
    knn_voters = kNNClassifier(k)
    knn_voters.fit(X, y)

    xy_coordinate = np.linspace(0, 10, 500)
    xx, yy = np.meshgrid(xy_coordinate, xy_coordinate)
    X2 = np.c_[xx.ravel(), yy.ravel()]
    y_pred = knn_voters.predict(X2)

    Z = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')

    plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')
   
    plt.xlabel('Wealth')
    plt.ylabel('Religiousness')
    plt.title('Red circles represent Republicans, Blues Democrats, k=%i'%k)
    
    plt.xlim([0, 10]);
    plt.ylim([0, 10]);
    plt.plot(np.linspace(0,10,1000), -true_boundary_voting_pred(np.linspace(0,10,1000), np.zeros(1000)), linewidth=2, c='k', linestyle='--');

interactive_plot = interact(decision_plane, k=widgets.IntSlider(min=1, max=99, step=1, value=3))
interactive_plot


# Changes in the sample set, such as alterations in sample density, outliers, and class imbalance, can impact the decision in a k-nearest neighbors (KNN) algorithm.
# 
# Underfitting is when the error on the sample set and on the test are large, overfitting is when the error on the sample set is small and on the test is large. (k=1 - overfitting, k=100 - underfitting)
# 
# The complexity of the k-nearest neighbors (KNN) model is inversely related to the value of k, where smaller k values result in a more complex model prone to overfitting, capturing fine details in the data, while larger k values yield a simpler model with a smoother decision boundary that generalizes better but may overlook certain nuances in the training data.ata.

# Text goes here...

# ### 2. Finding the optimal k - 15 points
# Sample a single sample set of size 1000 and divide it randomly to train (0.6) / validation (0.2) / test (0.2) sets. Plot the train vs validation error for several k values, and choose the best k. Where do you underfit/overfit the data? Finally, estimate the generalization error of your chosen classifier using the test set. What would happen if you optimize directly on test? is the optimal k the same?
# 

# In[11]:


X, y = generate_data(m=1000, seed=42)

from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

k_values = range(1, 99)

def pred_errors(X_train, y_train, X_valid, y_valid, k_values):
    train_errors = []
    valid_errors = []

    knn = kNNClassifier(1)
    knn.fit(X_train, y_train)

    train_len = len(y_train)
    valid_len = len(y_valid)

    for k in k_values:
        knn.n_neighbors = k
        train_pred = knn.predict(X_train)
        valid_pred = knn.predict(X_valid)
        train_pred_true = np.sum(np.not_equal(train_pred, y_train))
        valid_pred_true = np.sum(np.not_equal(valid_pred, y_valid))
        train_errors.append(train_pred_true/train_len)
        valid_errors.append(valid_pred_true/valid_len)
    return train_errors, valid_errors

train_errors, valid_errors = pred_errors(X_train, y_train, X_valid, y_valid, k_values)
plt.plot(k_values, train_errors, label='Train Error')
plt.plot(k_values, valid_errors, label='Validation Error')
plt.xlabel('k')
plt.ylabel('Accuracy error')
plt.title('Train vs Validation Error for Different k values')
plt.legend()
plt.show()

optimal_k = k_values[valid_errors.index(min(valid_errors))]
print(f"Optimal k is : {optimal_k}", )


# In[12]:


knn = kNNClassifier(optimal_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
len_test = len(y_test)
test_error = (len_test - np.sum(y_pred == y_test)) / len_test
print(f"Test error is: {test_error}")


# In[13]:


train_errors, test_errors = pred_errors(X_train, y_train, X_test, y_test, k_values)

optimal_k = k_values[test_errors.index(min(test_errors))]
print(f"Optimal k is : {optimal_k}", )


# In[14]:


knn = kNNClassifier(optimal_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
len_test = len(y_test)
test_error = (len_test - np.sum(y_pred == y_test)) / len_test
print(f"Test error is: {test_error}")


# Underfitting occurs with a large k in KNN, overfitting with a small k.
# 
# If we select k on the test sample, we can make a model in which the error on it is minimal, but on others it makes a much larger error (similar to retraining).
# 
# Optimal k is different on valid and test samples.

# ### 3. Using cross validation - 25 points
# This time, put the test data aside (0.2 of the data), and apply 5-fold CV on the remaining data to evaluate the performance of each k value. 
# What is the size of the validation and train set now, in each CV iteration?
# Did your final model selection change? Explain why.

# In[17]:


X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=0)

X_folds = []

for _, test_index in kf.split(X_temp):
    X_folds.append(X_temp[test_index]) 

y_folds = []

for _, test_index in kf.split(y_temp):
        y_folds.append(y_temp[test_index]) 

k_values_model = []
errors = []
for i in range(5):
    X_train = np.concatenate([X_folds[ind] for ind in range(5) if ind!=i])
    y_train = np.concatenate([y_folds[ind] for ind in range(5) if ind!=i])
    X_valid = X_folds[i]
    y_valid = y_folds[i]
    train_errors, valid_errors = pred_errors(X_train, y_train, X_valid, y_valid, k_values)
    errors.append(np.mean(valid_errors))
    k_values_model.append(k_values[valid_errors.index(min(valid_errors))])
    
print(k_values_model)
print(errors)


# Size of the validation is 200 and size of train set is 800 in each CV iteration. On each iteration we got different k. It happens because training occurs on different data, which includes different numbers of outliers.

# ## References
# - http://scott.fortmann-roe.com/docs/BiasVariance.html
# - http://scott.fortmann-roe.com/docs/MeasuringError.html
# - http://scikit-learn.org/stable/modules/cross_validation.html
