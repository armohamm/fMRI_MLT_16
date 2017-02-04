## Project title :
 Predicting the word from brain activity  
 
##Problem Description : 
Trying out different predictive models that will predict words corresponding to the fMRI scan observed when a person reads a noun from given set of words .    
We are going to use data from an experiment where 300 different subjects were given a word from a set 
of 60 words along with a corresponding line diagram. Each word is associated with 218 human defined 
attributes. fMRI scan which was recorded of all 300 subjects will be used as training data . We are going 
to learn models which will predict a word (among two candidate words)  given fMRI scan of entirely new 
subject.Our test data consists of 60 cases.

##Data extraction :
Basically the subject(person) is shown an image and asked to think about the properties of the object ,
the image is shoen to the subject for around 3 sec and try to clear the min( given 8sec) before other image is ahown

Images are acquired in X\*Y\*Z dimension (voxel image , voxel are basically the parts of the brain that gets activated when the subjects thinks of a word .After this image was processed by some method and we finally have vortex values for each images shown
one more thing, each eg here was taken as the average of the image during 4s span while teh subject was shown the image few sec earlier.


##Classification :

1. **nearest neighbour**
	finding the training example that has least euledean distance
	variation : taking mean of the all the class and then finding distance from the class mean
	    finding k nearest and the value which has majority (considering k = odd)
	    this variat minimize the noise from prev one as when 2 class are relative at same dist
	    but there's also an issue with this as when a class has two clusters which are far from each other(dissimilar )
	    than the dist from mean will not give correct class

2. **discriminative and generative model**
    In the former, the goal is to directly learn to predict from the training data; typically this entails
    learning a prediction function with a given parametric form by setting its parameters. In the latter,
    what is learned is essentially a statistical model that could generate an example belonging to a particular
    class.

    GNB, LR, LDA (feature selection or dimen red.)

3. **feature selection**
    scoring/filtering and wrapper methods
    The former involves ranking the features by a given criterion–each feature is scored by itself, a bit like a
    univariate test of a voxel–and selecting the best in the ranking. The latter consists broadly of picking new
    features by how much impact they have on the classifier given the features already selected (or, in reverse,
    considering all the features to begin with and removing features while performance increases).

##Report
`CS771Midtermprojectreport.pdf` contains detailed description of the project done

##Implementation :
 **using regression model :**
    so we had N*D data and each N had a value .
    we mapped each value to the feature matrix ( given separately for each value of Y . so we now have
    Y value as a N\*218(y_feature mat) instead of N*1 . Now using regression we find weight vector for each column
    in y feature matrix , in this we will have weght vector for each column in y_feature matrix ,i.e,
    weight matrix wiill be of size D*218.

    ```ye bas regression jaisa h , regression me apan ko ek N*D mat diya hota tha and a Y_val each N k corresponding ,
    uske liye hum ek weight vector nikalte the jo kisi bhi test data ka value bta de i.e 1\*D (weight) x D*1 (test data
    with D feature ) , jabki yaha apne pass wahi Y 218 h(mtlb ek data k correspong 218 value h ) .ye 218 values ye btate
    h ki wo object us feature k kitna pass h . ek weight vector ye predict kr rha h ki ek test data us feature k
    kitna paas h and ye cheez uska value nikal kr pta kr rha h (for eg .if 0 no relation if 1 - totally related )

    toh hme ab har test data k corrrespong 218 valus aa gye . ab ye check krenge k jo do word h uske ye kitna paas h .
    wo euclidean dist nikal k kr lenge.

    LASSO : L1 regularisation
    Ridge : L2 regularisation
    Elastic Net : both regularisation is used thus it will```


###Reference : 
http://www.sciencedirect.com/science/article/pii/S1053811908012263
https://www.analyticsvidhya.com/blog/2016/01/complete-tutorial-ridge-lasso-regression-python/#four
