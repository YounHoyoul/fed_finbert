from setuptools import setup, find_packages

setup(
    name="fednlp",
    version="0.1.0",
    author="Hoyoul Youn",
    author_email="yhy0215@gmail.com",
    url="https://github.com/YounHoyoul/fed_nlp.git",
    license="Apache",
    packages=['fednlp'],
    install_requires=[
        'torch==1.6.0',
        'torchvision==0.7.0',
        'transformers==2.11.0',
        'pytorch-pretrained-bert==0.6.2',
        'tensorflow==2.1.0',
        'ktrain==0.16.2',
        'scikit-learn==0.22.0',
        'wordcloud ',
        'spacy',
        'pysentiment2',
        'tqdm',
        'summa',
        'eli5',
        'lime',
        'textblob'
    ]
)

