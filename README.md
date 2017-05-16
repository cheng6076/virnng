# Generative Constituency Parser with (linear-time) Discriminative Recognition Algorithm

The code will be released before conference. 

## Dependencies
* Dynet (2.0)
* Numpy

## Instructions
* Training (supervised\_enc, supervised\_dec, unsupervised)
* Testing (lm, parsing)
* Focused training depending on the final objective: if parsing is the goal, focus on maximizing q(a|x) and p(a|x); if lm is the goal, focus on maximizing p(x)

## Citation
```
@InProceedings{cheng2017virnng, 
  author = {Cheng, Jianpeng and Lopez, Adam and Lapata, Mirella}, 
  title = {A Generative Parser with a Discriminative Recognition Algorithm}, 
  booktitle = {Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Short Papers)}, 
  year = {2017}, 
  address = {Vancouver, Canada}, 
  publisher = {Association for Computational Linguistics} 
 }
```
## Contact
jianpeng.cheng@ed.ac.uk
