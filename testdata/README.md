## Note About Datasets Used in Evaluation
Enron is run using the `enron-eval` dataset

Real Estate is run using the `real-estate-eval` dataset

For the easy and hard code generation evaluations, we needed to create a range of dataset sizes based on the `real-estate-eval` dataset. Thus, I created `real-estate-eval-5`, `real-estate-eval-10`, ..., `real-estate-eval-30`. Note that `real-estate-eval-15` should be equivalent to `real-estate-eval`.

Groundtruth labels are stored in the `groundtruth` folder. The Enron and Real Estate groundtruth files match the evaluation directories of the same name. The codegen groundtruths are slightly different than the criteria used for Real Estate, so they have their own set of labels in e.g. `codegen-easy-eval-[5,30].csv` which map(s) to `real-estate-eval-[5,30]`.  
