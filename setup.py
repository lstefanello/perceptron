from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
   name='perceptron',
   description='A simple package to make simple neural networks',
   author='Lincoln Stefanello',
   packages=['perceptron'],  #same as name
   install_requires=['numpy'], #external packages as dependencies
)
