# About 

You will be submitting an implementation of the class `AbstractSolution3` for automated grading;
that spec is included and documented in the file `abstract_solution_3.py`. See also file `solution3.py` for an example of how to implement your submission according to that spec. The latter simply includes the bare bones implementation, without all the data exploration stuff, from the base notebook provided to you originally. 

## Self-testing prior to submission

from an anaconda terminal or terminal where your python environment is available, run:

```
python tester.py
```

## Breaking Changes:

+ __Data Shape__<br>
The shape of the return value of `build_verbs_data` has been modified for more code safety; update your code accordingly. Basically we only changed its return value from a tuple to a dict (but reach out to the [python forum](https://opal.openu.ac.il/mod/ouilforum/view.php?f=225720) in case you need help in adapting your code to this change).
<br><br>

+ __Input Data Cleanup ☢️__ <br>
In the original notebook provided, we had also provided and used fused words from the conllu data, in addition to using the decomposed forms of the same fused words. For example, due to the way that the Hebrew treebank was built, the data building function `build_verbs_data` would provide your code with both the word "והגישו" and the decomposition of that word into separated tokens "ו" and "הגישו". In this revision, that is no longer the case. This means that the training data you get from this function is now also _smaller_, and has a little less tokens in the *non_verbs* category than before. Verify that your training algorithm still performs well, on this more correct makeup of input data. 
<br><br>

+ __Feature Selection Pollution ☢️__<br>
In the [original exercise notebook](https://goo.gl/rQAj8S), ngram features for question 3 were built from the entire dataset. This can be methodologically problematic; the test driver we use here, provides only a train set portion of the data to your training function, which avoids this. It also somewhat helps you to avoid overfitting, because you don't know which randomized train split your training function will get (which is a good thing!)
