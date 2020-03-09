Code for paper "Training Binary Neural Networks using the Bayesian Learning Rule". https://arxiv.org/abs/2002.10778 

1. Synthetic data

synthetic_data.ipynb

2. Image classification

 BayesBiNN method: 
 
 MNIST: python main_mnist.py --model MLPBinaryConnect --optim BayesBiNN
 
 Cifar10: python main_cifar10.py --model MLPBinaryConnect --optim BayesBiNN
 
 Cifar100: python main_cifar100.py --model MLPBinaryConnect --optim BayesBiNN
 
 STE-Adam method: 
 
 MNIST: python main_mnist.py --model MLPBinaryConnect_STE --optim STE
 
 Cifar10: python main_cifar10.py --model MLPBinaryConnect_STE --optim STE
 
 Cifar100: python main_cifar100.py --model MLPBinaryConnect_STE --optim STE
 
 
 Full-Precision Adam: 
 
 MNIST: python main_mnist.py --model MLPBinaryConnect --optim Adam
 
 Cifar10: python main_cifar10.py --model MLPBinaryConnect --optim Adam
 
 Cifar100: python main_cifar100.py --model MLPBinaryConnect --optim Adam
 
3. Continual Learning

python main_permute_mnist_CL.py

