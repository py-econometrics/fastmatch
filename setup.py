from setuptools import setup

extras_require = {
    "gpu": ["faiss-gpu"],
    "docs": ["pdoc"],
}

setup(
    name="fastmatch",
    version="0.1",
    description="Fast matching estimators for causal inference",
    url="http://github.com/apoorvalal/fastmatch",
    author="Apoorva Lal",
    author_email="lal.apoorva@gmail.com",
    license="MIT",
    install_requires=["numpy", "faiss-cpu", "scikit-learn", "pdoc"],
    extras_require=extras_require,
    packages=["fastmatch"],
    zip_safe=False,
)
