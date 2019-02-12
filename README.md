# Norm Conflict Classification

This repository contains the code used in the experiments of the paper entitled "Classification of Contractual Conflicts via Learning of Semantic Representations" accepted in the International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2019) by João Paulo Aires, Roger Granada, Juarez Monteiro, Rodrigo C Barros and Felipe Meneguzzi.


---
## Abstract

Contracts are the main medium through which parties formalize their trade relations, be they the exchange of goods or the specification of mutual obligations. While electronic contracts allow automated processes to verify their correctness, most agreements in the real world are still written in natural language, which need substantial human revision effort to eliminate possible conflicting statements in long and complex contracts. In this paper, we formalize a typology of conflict types between clauses suitable for machine learning and develop techniques to review contracts by learning to identify and classify such conflicts, facilitating the task of contract revision. We evaluate the effectiveness of our techniques using a manually annotated contract conflict corpus with results close to the current state-of-the-art for conflict identification, while introducing a more complex classification task of such conflicts for which our method surpasses the state-of-the art method.

---
## Conflict Identification and Classification

We develop two approaches for norm conflict identification, *i.e.*, classify unseen pairs of norms as conflict or non-conflict, and two further approaches to classify the *type* of conflict that occurs between norm pairs (*deontic-modality*, *deontic-structure*, *deontic-object*, or *object-conditional*). Before identifying or classifying norms, we transform each norm written in natural language within a contract into a vector representation using [sent2Vec](https://github.com/epfml/sent2vec). 



[image]: docs/distance.svg "Approaches using binary classification (a) and (b) and multiclass classification (c) and (d)."
![Alt text][image]


---
## How to cite

When citing our work in academic papers, please use this BibTeX entry:

```
@inproceedings{AiresEtAl2018aamas,
  author    = {Aires, Jo\~{a}o Paulo and Granada, Roger and Monteiro, Juarez and Barros, Rodrigo C and Meneguzzi, Felipe},
  title     = {Classification of Contractual Conflicts via Learning of Semantic Representations},
  booktitle = {Proceedings of the 18th International Conference on Autonomous Agents and Multiagent Systems},
  series    = {AAMAS 2019},
  location  = {Montreal, Canada},
  month     = {May},
  year      = {2019},
  publisher = {International Foundation for Autonomous Agents and Multiagent Systems}
}
```
