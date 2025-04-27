# Drug-Inhibitor-Cancer-Treatment

In this preliminary project phase, we will incorporate the other two repos as subtrees, so that the code is fully on our side however commit history is preserved.

This will be the case with TranSynergy and ... .

For Transynergy we did (in case you want to ad a subtree yourself):
```
git remote add drug_combination_repo https://github.com/qiaoliuhub/drug_combination.git
git fetch drug_combination_repo
git subtree add --prefix=external/drug_combination drug_combination_repo main
```