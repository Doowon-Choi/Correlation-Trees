# Correlation Trees

This repository contains Python code for building correlation trees. The idea of correlation tree is that the correlation of two variables of our interest can
be different depending on the conditions/interactions of other covariates. Correlation trees automatically identify subgroups with different correlations and conditions of covariates that lead to such difference. Correlation trees are developed not for *prediction*, but for *explanatory tool*, like clustering with specific respect to correlation measures. The current version provides main effect test alone for split variable selection. Details are found in the reference below.

## Train correlation trees
Before constructing correlation trees, the variables in the dataset should be carefully designed.   

The first and second column of dataset must be two variables of interest for correlation. The rest columns are used split variables alone, which may be involved
in different correlation between the two variables (i.e., the first and second column).

For a categorical variable in dataset, please convert it as numeric expression in the function "*Find_split_var_by_partialCor*".   
In the example in this repository, two categorical variables, Edu and Gender, are available and they are converted like the following way.
```python
Edu_mapper= {0:'Other', 1:'College', 2:'Master', 3:'Ph.D.'}
Edu_ordvar=df3["Edu"].replace(Edu_mapper)
df3["Edu"]=list(pd.factorize(Edu_ordvar))[0]     
Gender_mapper= {0:'Female', 1:'Male'}
Gender_ordvar=df3["Gender"].replace(Gender_mapper)
df3["Gender"]=list(pd.factorize(Gender_ordvar))[0]
```
The following function createTree is used to construct correlation tree.   
```python
model3_1 = createTree(df, df2, 6, corLeaf, ops=(0.01, 10))
```   
In the above, 'df' is the dataframe of the original dataset (i.e., imported data by pandas), df2 is the numpy array of the dataframe.

To control tree size, the third argument and the last argument, ops, are used like stopping condtions. The third argument controls the number of executing 'createTree' function
(i.e., how many splits are tried). The ops sets the threshold by a tuple whose first element is the difference in correlation before and after split (i.e., improvement in 
the objective functions) and second element is the minimum sample size in each node. The larger the third argument is and the smaller ops is, the larger/deeper tree is.  

This implemtnation uses Python dictionary to store the resulting tree.

## Objective functions
Unlike standard decision trees that use error as a split criterion, correlation trees do not have such metric since correlation trees are not models for prediction.
Based on various practical needs, three different objective functions are suggested.   

<img src="https://user-images.githubusercontent.com/69023373/89252883-a96b6600-d5e0-11ea-8463-b391bb1b74fb.PNG" width="50%">   

*Type 1* objective function aims to obtain multiple subgroups with strong correlations.   
*Type 2* objective function aims to identify the subgroup with the strongest correlation.   
*Type 3* objective function aims to identify the most distinguishable subgroups each other.   

## Types of correlation measures
Correlation trees are able to handle linear correlation measure (e.g., Pearson's correlation) and non-linear correlation measure (e.g., Spearman's rank correlation).

## Example of constructed correlation tree   
The following figure is a constructed linear correlation tree with Type 2 objective function in brain-behavior dataset. Subgroups with positive correlation mean subjects willing
to take a risk, while those with negative correlation mean subjects inclined to avoid a risk. We can easily find that different people have different tendency depending on age bracket and gender.

<img src="https://user-images.githubusercontent.com/69023373/89252932-c6079e00-d5e0-11ea-86a4-8b85b00d0cde.PNG" width="50%">   

## Version
- python 3.6.3
- numpy 1.16.4
- scipy 1.3.1
- pandas 0.20.3
- sklearn 0.21.1

## Further improvement in split variable selection
To construct a rigorous hypothesis test for selecting split variable, we would come up with a score-based test for bivariate normal distribution. The idea is to capture any
systematic fluctuation in correlation with respect to other covariates. Details will be described in my dissertation.   

## References
Choi, D., Li, L., Liu, H., & Zeng, L. (2020). A recursive partitioning approach for subgroup identification in brain–behaviour correlation analysis. *Pattern Analysis and Applications*, 23(1), 161-177.
