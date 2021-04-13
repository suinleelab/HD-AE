import setuptools

setuptools.setup(
    name="hd_ae",
    version="0.0.2",
    author="Ethan Weinberger",
    author_email="ewein@cs.washington.edu",
    description="For producing pretrained embedding models of scRNA-seq data.",
    long_description="""
        HD-AE (Hilbert-Schmidt Deconfounded Autoencoder is a deep learning model
        traine to produce integrated embedidngs of scRNA-seq data. Pretrained HD-AE
        models can be used to embed new batches of data at test time, without needing
        to retrain the model.
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/suinleelab/hd_ae",
    packages=['hd_ae'],
    install_requires=[
        'torch',
        'pytorch-lightning',
        'scanpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
)