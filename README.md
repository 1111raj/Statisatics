# Statisatics
Questions and Answers 
1. What is a vector in mathematics?
Ans: A Vector has both magnitude and as well as direction and is represented by an arrow overhead that defines its component in a coordinate system
2. How is a vector different from a scalar?
Ans: A vector has magnitude as well as direction while a scalar has magnitude only
eg: distance is a scalar quantity while displacement is a vector quantity
3. What are the different operations that can be performed on vectors?
Ans: we can perform Addition, Subtraction, Scalar Multiplication, Dot product(scalar Product), Cross Product(Vector Product), Magnitude(Norm), Normalization, Projection
4. How can vectors be multiplied by a scalar?
Ans: let’s suppose a vector V is multiplied by a scalar value K then VK is the multiplication of the vector if K > 0 preserving its direction if K < 0 reversing the direction of Vector
5. What is the magnitude of a vector?
Ans : The magnitude of the Vector is the length of the vector if v = (V1, V2, ….Vn)
•   ∣v∣=v12+v22+⋯+vn2
6. How can the direction of a vector be determined?
Ans: If the vector V = (V1, V2, ….Vn) then its direction can be given by
$$ \mathbf{u} = \frac{\|\mathbf{v}\|}{\mathbf{v}} = \left( \frac{\|\mathbf{v}\|}{v_1}, \frac{\|\mathbf{v}\|}{v_2}, \dots, \frac{\|\mathbf{v}\|}{v_n} \right)
$$
7. What is the difference between a square matrix and a rectangular matrix?
Ans: A square matrix is a matrix with the same number of rows and columns (n×n). For example, a 3×3 matrix is square.
A rectangular matrix is a matrix where the number of rows is different from the number of columns (m×n). For example, a 2×3 matrix is rectangular.
8. What is a basis in linear algebra?
Ans: A basis in linear algebra is a set of linearly independent vectors that span a vector space. In other words, any vector in the space can be uniquely expressed as a linear combination of the basis vectors. The number of vectors in the basis is called the dimension of the vector space.
9. What is a linear transformation in linear algebra?
Ans:
A linear transformation in linear algebra is a function between two vector spaces that preserves vector addition and scalar multiplication. If T is a linear transformation from vector space V to vector space W, then for any vectors u\and v in V and any scalar c:
 T(u+v)=T(u)+T(v)
T(cu)=cT(u) 
Linear transformations are used to map vectors from one space to another while maintaining the structure of the space.
10. What is an eigenvector in linear algebra?
Ans: An eigenvector of a square matrix A is a non-zero vector v that, when the matrix is applied to it, results in a scalar multiple of the vector. In other words, Av=λv, where λ lambda is a scalar called the eigenvalue corresponding to the eigenvector v.
Eigenvectors represent directions that are invariant under the linear transformation represented by A, meaning they only get stretched or compressed, but not rotated
11. What is the gradient in machine learning?
Ans: In machine learning, the gradient refers to the vector of partial derivatives of a function with respect to its parameters. It indicates the direction and rate of the fastest increase of the function.
12. What is backpropagation in machine learning?
Backpropagation is a supervised learning algorithm used to train artificial neural networks by adjusting the weights of the network. It works by computing the gradient of the loss function with respect to each weight in the network, using the chain rule of calculus, and then updating the weights to minimize the loss.
13. What is the concept of a derivative in calculus?
In calculus, a derivative represents the rate of change of a function with respect to one of its variables. It measures how the output of a function changes as the input changes. The derivative of a function f(x)
f(x) at a specific point gives the slope of the tangent line to the curve of the function at that point.
14. How are partial derivatives used in machine learning?
Partial derivatives are used in machine learning to compute the rate of change of a function with respect to one specific variable (or parameter), while holding all other variables constant. They are particularly important in the optimization process, especially when dealing with functions that depend on multiple variables, such as the loss function in machine learning models.
In machine learning, partial derivatives are used for:
	Gradient Descent: During the training of models, we often have a loss function that depends on many parameters (such as weights in a neural network). To minimize this loss, we compute the partial derivative of the loss function with respect to each parameter. This helps in determining the direction and magnitude of change for each parameter, allowing the model to adjust the weights to reduce the loss.
	Backpropagation: In neural networks, the loss is propagated backward through the network to update the weights. This involves computing partial derivatives of the loss function with respect to each weight and bias. The chain rule is used to combine partial derivatives from each layer of the network to calculate the gradients efficiently.
	Optimization: Partial derivatives help in the optimization process by giving insights into how sensitive the model’s performance is to each parameter, allowing for more precise adjustments to the parameters during training.
In short, partial derivatives are a fundamental part of the training process in machine learning algorithms that involve multi-variable functions, helping to efficiently update the model parameters and minimize the loss.
15. What is probability theory?
Probability theory is the branch of mathematics that deals with the analysis of random phenomena and events. It provides a framework for quantifying uncertainty and making predictions about the likelihood of various outcomes. In probability theory, events are assigned probabilities, which are values between 0 and 1, representing the likelihood that an event will occur.
concepts in probability theory include:
	Experiment: A process or action that results in one of several possible outcomes (e.g., rolling a die).
	Sample Space: The set of all possible outcomes of an experiment (e.g., for a die, the sample space is {1, 2, 3, 4, 5, 6}).
	Event: A specific outcome or a set of outcomes from the sample space (e.g., rolling an even number on a die).
	Probability: A measure of the likelihood of an event, calculated as the ratio of favorable outcomes to total outcomes in the sample space.
	Random Variable: A variable whose value is determined by the outcome of a random experiment.
	Conditional Probability: The probability of an event occurring given that another event has already occurred.
	Independence: Two events are independent if the occurrence of one does not affect the probability of the other.
	Expected Value: The long-term average or mean value of a random variable, representing the "center" of its distribution.
Probability theory underpins many fields, including statistics, machine learning, economics, and physics, helping to model uncertainty and make informed decisions based on data and random processes.
16. What are the primary components of probability theory?
The primary components of probability theory are:
	Sample Space (S): The set of all possible outcomes of an experiment.
	Events (E): A subset of the sample space, representing outcomes of interest.
	Probability (P): A measure that assigns a numerical value to the likelihood of an event, ranging from 0 (impossible event) to 1 (certain event).
	Random Variables: Functions that map outcomes from the sample space to real numbers.
	Probability Distribution: Describes how probabilities are distributed over the values of a random variable.
	Expectation (Expected Value): The long-term average value of repetitions of the experiment.
17. What is conditional probability, and how is it calculated?
Conditional probability is the probability of an event occurring given that another event has already occurred. It is calculated using the formula:
$$ P(A | B) = \frac{P(A \cap B)}{P(B)} $$
where P(A∣B) is the probability of event A given B, P(A∩B) is the probability of both events occurring, and P(B) is the probability of event B.
18. What is Bayes theorem, and how is it used?
Ans: Bayes' theorem describes the relationship between conditional probabilities and is given by:
$$ P(A | B) = \frac{P(B | A) \cdot P(A)}{P(B)} $$
where:
	P(A∣B) is the probability of event A given B.
	P(B∣A) is the probability of event B given A.
	P(A) and P(B) are the probabilities of events A and B, respectively.
Use: Bayes' theorem is used to update the probability of a hypothesis based on new evidence. It is widely applied in fields such as medical diagnosis, spam filtering, and machine learning.
19. What is a random variable, and how is it different from a regular variable?
Ans: A random variable is a variable that represents a numerical outcome of a random process or experiment. It can take different values, each with a certain probability.
Difference from a regular variable:
	A regular variable has a fixed value determined by the conditions or equations in a problem.
	A random variable, however, has values that are determined by chance and is associated with a probability distribution.
20. What is the law of large numbers, and how does it relate to probability theory?
The law of large numbers states that as the number of trials or repetitions of a random experiment increases, the average of the results will get closer to the expected value. In other words, the sample mean will converge to the true population mean.
Relation to probability theory: It supports the idea that probabilities are long-term averages, reinforcing the concept that empirical results become more predictable with more trials.
21. What is the central limit theorem, and how is it used?
The central limit theorem (CLT) states that the distribution of the sample mean of a large number of independent and identically distributed random variables approaches a normal distribution, regardless of the original distribution of the variables, as the sample size becomes large.
Use: The CLT is used in statistical inference, allowing us to make approximations about population parameters and perform hypothesis testing even when the population distribution is unknown.
22. What is the difference between discrete and continuous probability distributions?
Discrete Probabilitty distributions gives the countable number of distinct outcomes eg: Rolling a dice
Continuous probability Distribution: gives the infinite number of possible outcomes within a range eg: Time taken to complete the specific task , Height of the people
23. What are some common measures of central tendency, and how are they calculated?
Mean : The average of the dataset , calculated as sum of all the values divided by the number of values
Mean: The average of a dataset, calculated as the sum of all values divided by the number of values.
Median : sort the data in ascending or descending order if the number of values are odd then median is middle values and if the the number of values is even then average of the middle values
Mode: the value that appear most frequently in the dataset. dataset may have one mode more than one mode and no mode if all the values are unique
24. What is the purpose of using percentiles and quartiles in data summarization?
Ans : To know the outliers in the data distribution and compare the dataset s effectively.
25. How do you detect and treat outliers in a dataset?
Outliers in the datset can be removed with help of statistical analysis and visualization
statistical Methods: Z-score , if the z score is greater than 3 or less than 3 are consider as outliers
IQR(inter quartile range) : lower boundry = Q1 - 1.5 x IQR , upper boundry= Q3 + 1.5 x IQR, if any datapoint lies below the lower boundry or above the upper boundry are consider to be outlier
Visualization Method
Box plot: if the points are outside the wishker then considered to be outliers
scatter plot : if the points are far from the cluster then consider to be the outliers
26. What is the covariance of a joint probability distribution?
covariance gives the linear relationship between two variables
cov(x,y) = E[(X - E(x))(Y - E(y))] , E(X) and E(Y) are the expected values of X and Y
positive covariance : with increase in X, Y increases
negative covariance : with the increase in X , Y decreases Vice-Versa
Zero covariance : no linear relationship between X and Y
26. What is sampling in statistics, and why is it important?
sampling is the technique of getting the the subset of the entire population and drawing conclusion about the entire population
Efficiency: it allows us to data analysis when it is impractical or too costly to study the entire population,
Speed: Data collection and analysis is more faster with the sampling
Accuracy  : accurate population parameters can get with the sampling technique.
27. What are the different sampling methods commonly used in statistical inference?
	Random sampling
	stratified sampling
	systematic sampling
	cluster sampling
	convenience sampling
	snowball sampling.
28. What is the central limit theorem, and why is it important in statistical inference?
Central limit theorem(CLT) states that the samling distrisbution of the sample mean of the large number of independent and identically distributed random variables approaches a normal distribution
Importance : 
	Approximation : even when the population distribution is unknown or non-normal use to make inferences about the population mean 
	Hypothesis Testing : justifies the parametric tests such as z-test, t-test, which assume the normality of the distribution
	Gives the confidence intervals for the population parameters,
29. What is the p-value in hypothesis testing?
P-value gives the probability of obtaining a test statistics at least as extreme as the one observed given that null hypothesis (H0) is true . it quantifies the evidence against the null hypothesis
P < 0.05 : Low p value (null hypothesis should be rejected)
P > 0.05 : High p value (null hypothesis should not be rejected)s
30. What is confidence interval estimation?
Confidence interval estimation is a method to estimate the range of values within which a population parameter(like the mean or proportion) is likely to fall, with a certain level of confidence,
 
31. What are Type I and Type II errors in hypothesis testing?
Type -I error : Rejecting a true null Hypothesis (False Positive)
Type – II error : Rejecting a false null hypothesis (False Negative)
32. What is the difference between correlation and causation?
Correlation : if the two variables having relationship meaning they tend to move together it can be positive, negative, zero eg: in summer the sales of ice-cream increases it is not affecting but have impact on the sales of the ice-cream
Causation : one variable directly cause a change in other variables eg : smoking cause directly lung cancer that implies that smoking leads harmful health effect
33. How is a confidence interval defined in statistics?
A confidence interval in statistics is a range of values, derived from a sample, that is used to estimate an unknown population parameter (such as the population mean or proportion). The interval is associated with a confidence level, which represents the probability that the interval contains the true parameter value.
34. What does the confidence level represent in a confidence interval?
•  The confidence level (e.g., 95%) indicates the likelihood that the interval will contain the true population parameter if the sampling process is repeated many times.
•  A wider confidence interval means more uncertainty, while a narrower interval indicates more precision.
35. What is hypothesis testing in statistics?
It is the method to draw the conclusion about a population based on sample data. Hypothesis is assumption about a population parameter and then checking weather the observed data supports or contradict our assumption 
36. What is the purpose of a null hypothesis in hypothesis testing?
It is the basline for the comparision against the alternative hypothesis where by default we assume there is no effect, no relationship and no difference between the variables in the population being studied.
37. What is the difference between a one-tailed and a two-tailed test?
•  One-tailed test: Tests for an effect in one direction (greater than or less than).
•  Two-tailed test: Tests for an effect in either direction (different from a specified value).
38. What is the geometric interpretation of the dot product?
Geometric interpretation of the dot product in terms of their magnitude and angle between them
A⋅B=∣A∣∣B∣cos(θ)

Magnitude and Direction : It gives the magnitude of the two vectors and the cosine of the angle between the vectors it gives you the information about how much of one vector lies in the direction of other vector.
Angle between two vectors: if the dot product  > 0 , angle is acute(less than 90°)
if the dot product < 0, angle between the vectors is obtuse(greater than 90°)
if the dot product = 0, both vectors are perpendicular(equal to 90°)
Projection : dot product gives the information of the projection of one vector on another
 
39. What is the geometric interpretation of the cross-product?
∣A×B∣=∣A∣∣B∣sin(θ)
The magnitude of ∣A×B∣ gives the area of the parallelogram formed by the two vectors
The direction of the cross product is perpendicular to the direction of the plane containing the two vectors,
40. How are optimization algorithms with calculus used in training deep learning models?
Specially calculus(derivatives and gradients) allows optimization algorithms like gradient descent is used to update the parameters of deep learning models during the training.
Minimising the loss function, back-propagation helps to compute the gradient in deep networks.
41. What are observational and experimental data in statistics?
Observational Data: When the researcher observes the data and record the data without manipulating and controlling the variables. Takes the outcomes in the natural settings. Majorly used for the identification of the pattern, relationship and correlations between the variables.
Experimental Data : When the researcher observes the data and do some manipulation on one variable to see the effect on other variables and their effect on their dependent variables(outcomes)
42. What is the left-skewed distribution and the right-skewed distribution?
A left-skewed distribution has a longer tail on the left side (the lower end) of the distribution.
The mean is typically less than the median because the tail pulls the mean to the left.
•  Most of the data is concentrated on the right (higher) side of the distribution.The left side of the graph is stretched out, with the majority of the data clustered to the right.
 
43. What is Bessel’s correction?
calculation of the sample variance and standard deviation in order to account for the bias that arises when estimating population parameters from a sample.
This correction is important when working with small sample sizes, as it helps provide a more accurate estimate of the population variance. For large sample sizes, the difference between using nnn and n−1n - 1n−1 becomes negligible.
44. What is kurtosis?
Kurtosis is a statistical measure that describes the shape of a distribution's tail and the peak. It indicates how heavy or light the tails of a distribution are compared to a normal distribution, and how peaked the distribution is.
Kurtosis is useful for understanding the potential for outliers and the overall shape of the data's distribution, which can affect the choice 	of statistical models or methods used.
45. What is the probability of throwing two fair dice when the sum is 5 and 8?
When throwing two dice, each die has 6 faces, so the total number of possible outcomes is: 
 6 x 6 = 36
To get a sum of 5, the following pairs of dice rolls are possible:(1, 4), (2, 3), (3, 2), (4, 1)
So, there are 4 favorable outcomes for a sum of 5.
To get a sum of 8, the following pairs of dice rolls are possible: (2, 6), (3, 5), (4, 4), (5, 3),  (6, 2)
So, there are 5 favorable outcomes for a sum of 8.
Since the events "sum is 5" and "sum is 8" are mutually exclusive (they cannot happen at the same time), we simply add the favorable outcomes: 4 + 5 = 9
The probability of throwing a sum of 5 or 8 is the ratio of favorable outcomes to the total number of outcomes: 9 / 36 = 1/4 
The probability of throwing two fair dice and getting a sum of 5 or 8 is 1/4  or 0.25,
46. What is the difference between Descriptive and Inferential Statistics?
Aspect	Descriptive Statistics	Inferential Statistics
Purpose	Summarize and describe data	Make inferences about a population based on sample data
Scope	Describes a dataset in its entirety	Generalizes from a sample to a population
Methods	Mean, median, mode, standard deviation, charts, tables	Hypothesis testing, confidence intervals, regression, etc.
Outcome	Provides simple summaries and visualizations	Provides predictions, estimates, and conclusions about the population
Example	"The average test score of 100 students is 75."	"We can estimate the average test score of all students in the school based on the sample."

47. Imagine that Jeremy took part in an examination. The test has a mean score of 160, and it has a standard deviation of 15. If Jeremy’s z-score is 1.20, what would be his score on the test?
z= x−μ / σ
z = z-score = 1.20
x = score of Jeremy = ?? (need to find)
μ = mean score = 160
σ = standard deviation = 15
x = 1.20 x 15 + 160 = 178
Jeremy’s score on the test is 178.
48. In an observation, there is a high correlation between the time a person sleeps and the amount of productive work he does. What can be inferred from this?
From a high correlation between sleep and productivity, we can infer that there is an association between the two. However, without further analysis, we cannot conclude that sleep directly causes an increase in productivity or vice versa.
49. What is the meaning of degrees of freedom (DF) in statistics?
Degrees of freedom (DF) are a fundamental concept in statistics used to describe the number of independent pieces of information available to estimate parameters or perform tests. They are essential for determining the correct statistical methods and calculating test statistics accurately.
50. If there is a 30 percent probability that you will see a supercar in any 20-minute time interval, what is the proba¬bility that you see at least one supercar in the period of an hour (60 minutes)?
p = 30% = 0.30 (probability of supercar in 20 minutes)
p’ = 1 – p = 1 – 0.30 = 0.7 (probability of no supercar in 20 minutes)
p= 0.7 x 0.7 x 0.7 =0.343 (Probability of supercar in 60 minutes)
p = 1 – 0.343 = 0.657(At least one supercar in 60 minutes)
The probability of seeing at least one supercar in a 60-minute period is 0.657 or 65.7%.
