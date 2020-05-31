import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brminer",
    version="0.0.1",
    author="Juan Carlos Velazco Rossell, Miguel Angel Medina Pérez and Octavio Loyola Gonzáles",
    author_email="miguelmedinaperez@gmail.com",
    description="BaggingRandomMiner is an ensemble of weak one-class classifiers based on dissimilarities. In the training phase, every weak classifier is built using Bagging and computing a threshold of dissimilarities. In the classification phase, the classification scores of the weak classifiers are averaged, and every weak classifier computes its score based on the dissimilarity to the nearest neighbor and the threshold computed in the training phase. -J. Benito Camiña, M. A. Medina-Pérez, R. Monroy, O. Loyola-González, L. A. Pereyra-Villanueva, L. C. González-Gurrola, \"Bagging-RandomMiner: A one-class classifier for file access-based masquerade detection,\" Machine Vision and Applications, vol. 30, no. 5, pp. 959-974, 2019. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/octavioloyola/BRM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=2.7',
)
