Download link :https://programming.engineering/product/cs440-ece448-mp01-probability/

# CS440-ECE448-MP01-Probability
CS440/ECE448 MP01: Probability
The first thing you need to do is to download this file: mp01.zip. It has the following

content:

submitted.py



Your homework. Edit, and then submit to Gradescope.

mp01_notebook.ipynb



This is a Jupyter notebook to help you debug. You

can completely ignore it if you want, although you might find that it gives you useful instructions.

grade.py



Once your homework seems to be working, you can test it by typing

python grade.py tests/tests_visible.py

, which will run the tests in .

tests/test_visible.py



: This file contains about half of the unit tests that

Gradescope will run in order to grade your homework. If you can get a perfect score on these tests, then you should also get a perfect score on the additional hidden tests that Gradescope uses.

solution.json



This file contains the solutions for the visible test cases, in

JSON format. If the instructions are confusing you, please look at this file, to see if it can help to clear up your confusion.

data



This directory contains the data.

reader.py



This is an auxiliary program that you can use to read the data.

requirements.txt



This tells you which python packages you need to have

grade.py

installed, in order to run . You can install all of those packages by

pip install -r requirements.txt pip3 install -r


typing or

requirements.txt

.

mp01_notebook.ipynb

This file ( ) will walk you through the whole MP, giving you instructions and debugging tips as you go.

Table of Contents

Reading the data

Joint and Conditional Distributions

Mean, Variance and Covariance

Expected Value of a Function of an RV

Grade Your Homework

Reading the data

There are two types of data: visible data (provided to you), and hidden data (available only to the autograder on Gradescope). If you get your code working for the visible

2/1/24, 7:04 PM mp01_notebook

data, it should also work for the hidden data.

The visible dataset consist of 500 emails, a subset of the Enron-Spam dataset

provided by Ion Androutsopoulos. MP02 will use a larger portion of the same dataset.

In order to help you load the data, we provide you with a utility function called

reader.py

To use it, you will need to install nltk. It should be possible for you to

do this by running the following code block:


2/1/24, 7:04 PM mp01_notebook

Help on module reader:

NAME

reader – This file is responsible for providing functions for reading the files

FUNCTIONS

loadDir(dirname, stemming, lower_case, use_tqdm=True)

Loads the files in the folder and returns a

list of lists of words from the text in each file.

Parameters:

name (str): the directory containing the data

stemming (bool): if True, use NLTK’s stemmer to remove suffixes

lower_case (bool): if True, convert letters to lowercase

use_tqdm (bool, default:True): if True, use tqdm to show status ba

r

Output:

texts (list of lists): texts[m][n] is the n’th word in the m’th em

ail

count (int): number of files loaded

loadFile(filename, stemming, lower_case)

Load a file, and returns a list of words.

Parameters:

filename (str): the directory containing the data

stemming (bool): if True, use NLTK’s stemmer to remove suffixes

lower_case (bool): if True, convert letters to lowercase

Output:

x (list): x[n] is the n’th word in the file

DATA

bad_words = {‘aed’, ‘eed’, ‘oed’}

porter_stemmer = <PorterStemmer>

tokenizer = RegexpTokenizer(pattern=’\\w+’, gaps=False, disc…ty=Tru

e…

FILE

/Users/jhasegaw/Dropbox/mark/teaching/ece448/ece448labs/spring24/mp01/ src/reader.py

2/1/24, 7:04 PM mp01_notebook

The first file contained the following words: [‘Subject’, ‘done’, ‘new’, ‘sitara’, ‘desk’, ‘request’, ‘ref’, ‘cc’, ‘20000813’, ‘carey’, ‘per’, ‘sco tt’, ‘s’, ‘request’, ‘below’, ‘the’, ‘following’, ‘business’, ‘unit’, ‘ak a’, ‘desk’, ‘id’, ‘portfolio’, ‘was’, ‘added’, ‘to’, ‘global’, ‘productio n’, ‘and’, ‘unify’, ‘development’, ‘test’, ‘production’, ‘and’, ‘stage’, ‘please’, ‘copy’, ‘to’, ‘the’, ‘other’, ‘global’, ‘environments’, ‘thank s’, ‘dick’, ‘x’, ‘3’, ‘1489’, ‘updated’, ‘in’, ‘global’, ‘production’, ‘en vironment’, ‘gcc’, ‘code’, ‘desc’, ‘p’, ‘ent’, ‘subenti’, ‘data’, ‘_’, ‘c d’, ‘ap’, ‘data’, ‘_’, ‘desc’, ‘code’, ‘_’, ‘id’, ‘a’, ‘sit’, ‘deskid’, ‘i mcl’, ‘a’, ‘ena’, ‘im’, ‘cleburne’, ‘9273’, ‘from’, ‘scott’, ‘mills’, ‘0 8′, ’30’, ‘2000’, ’08’, ’27’, ‘am’, ‘to’, ‘samuel’, ‘schott’, ‘hou’, ‘ec t’, ‘ect’, ‘richard’, ‘elwood’, ‘hou’, ‘ect’, ‘ect’, ‘debbie’, ‘r’, ‘brack ett’, ‘hou’, ‘ect’, ‘ect’, ‘judy’, ‘rose’, ‘hou’, ‘ect’, ‘ect’, ‘vanessa’, ‘schulte’, ‘corp’, ‘enron’, ‘enron’, ‘david’, ‘baumbach’, ‘hou’, ‘ect’, ‘e ct’, ‘daren’, ‘j’, ‘farmer’, ‘hou’, ‘ect’, ‘ect’, ‘dave’, ‘nommensen’, ‘ho u’, ‘ect’, ‘ect’, ‘donna’, ‘greif’, ‘hou’, ‘ect’, ‘ect’, ‘shawna’, ‘johnso n’, ‘corp’, ‘enron’, ‘enron’, ‘russ’, ‘severson’, ‘hou’, ‘ect’, ‘ect’, ‘c c’, ‘subject’, ‘new’, ‘sitara’, ‘desk’, ‘request’, ‘this’, ‘needs’, ‘to’, ‘be’, ‘available’, ‘in’, ‘production’, ‘by’, ‘early’, ‘afternoon’, ‘sorr y’, ‘for’, ‘the’, ‘short’, ‘notice’, ‘srm’, ‘x’, ‘33548’]

Joint, Conditional, and Marginal Distributions

In this week’s MP, we will work with the following two random variables:

X1 = the number of times that word1 occurs in a text



X = the number of times that word2 occurs in a text

2

… where you can specify word1 and word2 as parameters of the function. In this

section, we will compute the joint, conditional, and marginal distributions of X1 and X2. These will be estimated, from the available data, using the following formulas,

where N(X1 = x1, X2 = x2) is the number of texts in the dataset that contain x1 instances of word1, and x2 instances of word2:

Joint distribution:

N(X1 = x1, X2 = x2)



P(X1 = x1, X2 = x2) = ∑x1 ∑x2 N(X1 = x1, X2 = x2)

Marginal distributions:

P(X1 = x1) = ∑ P(X1 = x1, X2 = x2)

x2

P(X2 = x2) = ∑ P(X1 = x1, X2 = x2)

x1

Conditional distribution:

P(X2 = x2|X1 = x1) = P(X1 = x1, X2 = x2)

P(X1 = x1)

submitted.py

At this point, we’ll load the file .

2/1/24, 7:04 PM mp01_notebook

Help on function marginal_distribution_of_word_counts in module submitted:

marginal_distribution_of_word_counts(texts, word0)

Parameters:

texts (list of lists) – a list of texts; each text is a list of words word0 (str) – the word that you want to count

Output:

Pmarginal (numpy array of length cX0) – Pmarginal[x0] = P(X0=x0), wher

e

X0 is the number of times that word0 occurs in a document cX0-1 is the largest value of X0 observed in the provided texts

marginal_distribution_of_word_counts

Edit so that it does the task

specified in its docstring. When you get the code working, you can count the number of times that the word “company” occurs in any given document once, twice, thrice, etc. It turns out that only 2.4% of texts contain the word “company” just once, 0.2% contain it twice, 0.2% contain it four times; 97.2% don’t contain it at all.


2/1/24, 7:04 PM mp01_notebook


It looks like the mean vector will be pretty close to μ = [0, 0]. Let’s find out.


In [20]: importlib.reload(submitted)

help(submitted.mean_vector)

Help on function mean_vector in module submitted:

mean_vector(Pjoint)

Parameters:

Pjoint (numpy array, shape=(cX0,cX1)) – Pjoint[x0,x1] = P(X0=x0, X1=x

1)

Outputs:

mu (numpy array, length 2) – the mean of the vector [X0, X1]


In [21]: importlib.reload(submitted)

mu = submitted.mean_vector(Pa_the)

print(mu)

[1.364 4.432]

That’s a bit of a surprise – the mean of X1 is higher than the mean of X0! That result

wasn’t obvious in the figure, unless you noticed that the maximum value of X1 is 58, while the maximum value of X0 is only 19.

Now let’s try to find the matrix of variances and covariances.

2/1/24, 7:04 PM mp01_notebook


In [22]: importlib.reload(submitted)

help(submitted.covariance_matrix)

Help on function covariance_matrix in module submitted:

covariance_matrix(Pjoint, mu)

Parameters:

Pjoint (numpy array, shape=(cX0,cX1)) – Pjoint[x0,x1] = P(X0=x0, X1=x

1)

mu (numpy array, length 2) – the mean of the vector [X0, X1]

Outputs:

Sigma (numpy array, shape=(2,2)) – matrix of variance and covariances of [X0,X1]


In [23]: importlib.reload(submitted)

Sigma = submitted.covariance_matrix(Pa_the, mu)

print(Sigma)

[[ 4.891504 9.244752]

[ 9.244752 41.601376]]

A few things to notice:

The variance of X1 is larger than the variance of X0. This is because X1 varies



over a larger range than X0, with nonzero probabilities all the way.



The covariance of X0 and X1 is positive, meaning that a large value of X0 tends

to co-occur with a large value of X1. Probably, this just means that long texts


a the

have larger counts of both the words and .

Function of Random Variables is a Random

Variable

Finally, let’s calculate a new random variable by taking a function of the random

variables X0 and X1. Any function of random variables is a random variable, and its distribution is

P(f(X0, X1) = z) = ∑ P(X0 = x0, X1 = x1)

x0,x1:f(x0,x1)=z

Let’s read the docstring:


2/1/24, 7:04 PM mp01_notebook

Help on function distribution_of_a_function in module submitted:

distribution_of_a_function(Pjoint, f)

Parameters:

Pjoint (numpy array, shape=(cX0,cX1)) – Pjoint[x0,x1] = P(X0=x0, X1=x

1)

f (function) – f should be a function that takes two real-valued inputs, x0 and x1. The output, z=f(x0,x1),

may be any hashable value (number, string, or even a tuple).

Output:

Pfunc (Counter) – Pfunc[z] = P(Z=z)

Pfunc should be a collections.defaultdict or collections.Counter, so that previously unobserved values of z have a default setting of Pfunc[z]=0.

here


You can read about defaultdict and Counter data types

<https://docs.python.org/3/library/collections.html>

_. Basically, they

are just dictionaries with a default value for any previously unseen keys.

Let’s create a new random variable whose value is a string, rather than being a number. Here is the function:


2/1/24, 7:04 PM mp01_notebook

ax.set_xlabel(‘Instance value $z=f(x_0,x_1)$’)


ax.set_ylabel(‘$P(f(X_0,X_1)=z)$’)

ax.set_title(‘Probability Mass Function of a Function of Two Random Varia

Out[27]: Text(0.5, 1.0, ‘Probability Mass Function of a Function of Two Random Va riables’)


Grade your homework

If you’ve reached this point, and all of the above sections work, then you’re ready to try grading your homework! Before you submit it to Gradescope, try grading it on your own machine. This will run some visible test cases (which you can read in

tests/test_visible.py

), and compare the results to the solutions (which you

solution.json

can read in ).

The exclamation point (!) tells python to run the following as a shell command.

Obviously you don’t need to run the code this way — this usage is here just to remind you that you can also, if you wish, run this command in a terminal window.


2/1/24, 7:04 PM mp01_notebook

EE…………

====================================================================== ERROR: test_extra (test_extra.TestStep)

———————————————————————-

Traceback (most recent call last):

File “/Users/jhasegaw/Dropbox/mark/teaching/ece448/ece448labs/spring24/m p01/src/tests/test_extra.py”, line 16, in test_extra

hyp_p, hyp = extra.estimate_geometric(Pa)

File “/Users/jhasegaw/Dropbox/mark/teaching/ece448/ece448labs/spring24/m p01/src/extra.py”, line 13, in estimate_geometric

raise RuntimeError(“You need to write this”)

RuntimeError: You need to write this

====================================================================== ERROR: test_extra (test_extra_hidden.TestStep)

———————————————————————-

Traceback (most recent call last):

File “/Users/jhasegaw/Dropbox/mark/teaching/ece448/ece448labs/spring24/m p01/src/tests/test_extra_hidden.py”, line 16, in test_extra

hyp_p, hyp = extra.estimate_geometric(Pa)

File “/Users/jhasegaw/Dropbox/mark/teaching/ece448/ece448labs/spring24/m p01/src/extra.py”, line 13, in estimate_geometric

raise RuntimeError(“You need to write this”)

RuntimeError: You need to write this

———————————————————————-

Ran 14 tests in 0.431s

FAILED (errors=2)

If your code is working, then as shown above, the only error you get should be from

test_extra.py


the extra credit part ( ).

If you got any other ‘E’ marks, it means that your code generated some runtime errors, and you need to debug those.

If you got any ‘F’ marks, it means that your code ran without errors, but that it

solutions.json

generated results that are different from the solutions in . Try debugging those differences.

If neither of those things happened, and your result was a series of dots except for

test_extra.py

the one error associated with , then your code works perfectly.

If you’re not sure, you can try running grade.py with the -j option. This will produce a JSON results file, in which you should get a score of 50%.


2/1/24, 7:04 PM mp01_notebook

{

“tests”: [

{

“name”: “test_extra (test_extra.TestStep)”,

“score”: 0.0,

“max_score”: 5,

“status”: “failed”,

“output”: “Test Failed: You need to write this\n”

},

{

“name”: “test_extra (test_extra_hidden.TestStep)”,

“score”: 0.0,

“max_score”: 5,

“status”: “failed”,

“output”: “Test Failed: You need to write this\n”

},

{

“name”: “test_cond (test_hidden.TestStep)”,

“score”: 8,

“max_score”: 8,

“status”: “passed”

},

{

“name”: “test_covariance (test_hidden.TestStep)”,

“score”: 8,

“max_score”: 8,

“status”: “passed”

},

{

“name”: “test_distribution_of_function (test_hidden.TestSte

p)”,

“score”: 8,

“max_score”: 8,

“status”: “passed”

},

{

“name”: “test_joint (test_hidden.TestStep)”,

“score”: 9,

“max_score”: 9,

“status”: “passed”

},

{

“name”: “test_marginal (test_hidden.TestStep)”,

“score”: 9,

“max_score”: 9,

“status”: “passed”

},

{

“name”: “test_mean (test_hidden.TestStep)”,

“score”: 8,

“max_score”: 8,

“status”: “passed”

},

{

“name”: “test_cond (test_visible.TestStep)”,

“score”: 8,

“max_score”: 8,

“status”: “passed”

},

{

2/1/24, 7:04 PM mp01_notebook

“name”: “test_covariance (test_visible.TestStep)”,

“score”: 8,

“max_score”: 8,

“status”: “passed”

},

{

“name”: “test_distribution_of_function (test_visible.TestSte

p)”,

“score”: 8,

“max_score”: 8,

“status”: “passed”

},

{

“name”: “test_joint (test_visible.TestStep)”,

“score”: 9,

“max_score”: 9,

“status”: “passed”

},

{

“name”: “test_marginal (test_visible.TestStep)”,

“score”: 9,

“max_score”: 9,

“status”: “passed”

},

{

“name”: “test_mean (test_visible.TestStep)”,

“score”: 8,

“max_score”: 8,

“status”: “passed”

}

],

“leaderboard”: [],

“visibility”: “visible”,

“execution_time”: “0.23”,

“score”: 100.0

}

submitted.py

Now you should try uploading to Gradescope.

Gradescope will run the same visible tests that you just ran on your own machine,

plus some additional hidden tests. It’s possible that your code passes all the visible

tests, but fails the hidden tests. If that happens, then it probably means that you

hard-coded a number into your function definition, instead of using the input

parameter that you were supposed to use. Debug by running your function with a

variety of different input parameters, and see if you can get it to respond correctly in

all cases.

Once your code works perfectly on Gradescope, with no errors, then you are done with the MP. Congratulations!

Extra Credit

On many of the machine problems (not all), extra credit of up to 10% will be available for doing a problem that goes a little bit beyond the material we’ve covered in lecture.

2/1/24, 7:04 PM mp01_notebook

On MP01, for extra credit, let’s model the frequency of a word as a geometric random variable. A geometric random variable, Y , is one whose pmf is given by

P(Y = y) = p(1 − p)y for all non-negative integer values of y, where p is a parameter called the “success probability” or the “stopping probability.”

In order to model an observed random variable (X) using a geometric random

variable (Y ), the easiest way to estimate the model is by calculating E[X], then

choosing the parameter p so that E[Y ] = E[X]. The mean of a geometric random

variable is E[Y ] =

1−p

.

p


For extra credit, try estimating the parameter p that matches the observed mean of a

extra.py

non-negative integer random variable. The template code is in , which

estimate_geometric

has just one function for you to complete, the function :


In [68]: import extra, importlib

importlib.reload(extra)

help(extra.estimate_geometric)

Help on function estimate_geometric in module extra:

estimate_geometric(PX)

@param:

PX (numpy array of length cX): PX[x] = P(X=x), the observed probabilit y mass function

@return:

p (scalar): the parameter of a matching geometric random variable

PY (numpy array of length cX): PY[x] = P(Y=y), the first cX values of the pmf of a

geometric random variable such that E[Y]=E[X].

When you have the code working, you can test it by finding a geometric distribution

model for the number of occurrences of the word “a”:


In [70]: importlib.reload(extra)

p, PY = extra.estimate_geometric(Pa)

print(‘p=’,p)

print(‘The first five entries in the model pmf are’,PY[:5])

p= 0.4230118443316413

The first five entries in the model pmf are [0.42301184 0.24407282 0.14082 713 0.08125559 0.04688351]


In [71]: import matplotlib.pyplot as plt

fig, axs = plt.subplots(2,1,figsize=(14,4))

axs[0].bar(np.arange(len(Pa)), Pa)

axs[0].set_title(‘Observed probability mass function’)

axs[0].set_ylabel(‘$P(X=x)’)

axs[1].bar(np.arange(len(PY)), PY)

axs[1].set_title(‘Geometric distribution model’)

axs[1].set_ylabel(‘$P(Y=x)$’)

axs[1].set_xlabel(‘Instance value, $x$, of the number of occurrences of t fig.tight_layout()

2/1/24, 7:04 PM mp01_notebook


grade.py

You can test your extra credit by running again.


In [72]: !python grade.py

…….

———————————————————————-

Ran 7 tests in 0.102s

OK

extra.py

When that works, try uploading your file to Gradescope, under the

MP01 Extra Credit


heading .


In [ ]: 
