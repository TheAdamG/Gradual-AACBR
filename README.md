# Gradual Abstract Argumentation for Case-Based Reasoning (Gradual AA-CBR)

**Gradual AA-CBR** is a neuro-symbolic AI library implemented in PyTorch that models case-based reasoning as a debate between data points. Each labeled instance acts as an *argument* advocating for its classification, engaging in structured interactions to determine the outcome of new, unlabeled instances. The framework extends Abstract Argumentation-based Case-Based Reasoning (AA-CBR) with trainable, differentiable mechanisms, allowing the argumentative structure to be learned via gradient descent, similar to how neural networks function.

## 📦 Installation

Install from source. Run the following from the root directory:

```bash
pip install .
```

## 🚀 Usage

Example usage can be found in the [`examples/`](../examples/) directory. These examples demonstrate how to construct debates, train models, and make predictions using the Gradual AA-CBR framework.

## ✨ Key Features

- 🧠 **Neuro-symbolic reasoning**: Combines formal argumentation with neural learning.
- 🗣️ **Case-based debates**: Each data point argues for its label in a structured debate.
- 🔁 **Differentiable argumentation semantics**: Argumentation structure is learned using backpropagation.
- ⚙️ **Modular PyTorch components**: Easily integrable into other ML pipelines.
- 📚 **Example-driven design**: Pre-built examples guide users through usage and training.

## 📂 Project Structure

```
Gradual-AA-CBR/
├── src/
│   └── deeparguing/
│       └── ...
├── examples/
│   └── ...
├── data/
│   └── ...
└── README.md
```



## 📚 References
Gradual-AA-CBR is a result of research carried out by the [Computational Logic and Argumentation group](https://clarg.doc.ic.ac.uk/), at Imperial College London. This repository is based on the following publications:

[1]: https://arxiv.org/abs/2505.15742 (Adam Gould, Francesca Toni: Neuro-Argumentative Learning with Case-Based Reasoning. arXiv)
> **Adam Gould, Francesca Toni**: *Neuro-Argumentative Learning with Case-Based Reasoning*. arXiv
([text](https://arxiv.org/abs/2505.15742), [bib](https://arxiv.org/bibtex/2505.15742))

[2]: https://dl.acm.org/doi/10.5555/3032027.3032100 (Kristijonas Cyras, Ken Satoh, Francesca Toni: Abstract Argumentation for Case-Based Reasoning. KR 2016: 549-552)
> **Kristijonas Cyras, Ken Satoh, Francesca Toni**: *Abstract Argumentation for Case-Based Reasoning*. KR 2016: 549-552
([text](https://dl.acm.org/doi/10.5555/3032027.3032100), [bib](https://dblp.org/rec/conf/kr/CyrasST16.html?view=bibtex))

[3]: https://doi.org/10.3233/FAIA200377 (Oana Cocarascu, Andria Stylianou, Kristijonas Čyras and Francesca Toni: Data-Empowered Argumentation for Dialectically Explainable Predictions. ECAI 2020)
>**Oana Cocarascu, Andria Stylianou, Kristijonas Čyras and Francesca Toni**: *Data-Empowered Argumentation for Dialectically Explainable Predictions*. ECAI 2020
([text](https://doi.org/10.3233/FAIA200377), [bib](https://dblp.org/rec/conf/ecai/CocarascuSCT20.html?view=bibtex))

[4]: https://doi.org/10.1609/aaai.v35i7.16801 (Nico Potyka)
>**Nico Potyka**: *Interpreting Neural Networks as Quantitative Argumentation Frameworks*. AAAI 2021
([text](https://ojs.aaai.org/index.php/AAAI/article/view/16801/16608),   [bib](https://dblp.org/rec/conf/aaai/Potyka21.html?view=bibtex)

## 📚 Data Sources

- Iris: [link](https://archive.ics.uci.edu/dataset/53/iris)
- Mushroom: [link](https://archive.ics.uci.edu/dataset/73/mushroom)
- Glioma Grading Clinical and Mutation Features: [link](https://archive.ics.uci.edu/dataset/759/glioma+grading+clinical+and+mutation+features+dataset)
- Breast Cancer Wisconsin (Diagnostic): [link](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
