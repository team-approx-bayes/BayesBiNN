
Code for "Training Binary Neural Networks using the Bayesian Learning Rule".

1. Synthetic data
Run synthetic_data.ipynb

2. Real Image classification

 BayesBiNN method: 
 
 MNIST: python main_mnist.py --model MLPBinaryConnect --optim BayesBiNN
 
 Cifar10: python main_cifar10.py --model MLPBinaryConnect --optim BayesBiNN
 
 Cifar100: python main_cifar100.py --model MLPBinaryConnect --optim BayesBiNN
 
 STE-Adam method: 
 
 MNIST: python main_mnist.py --model MLPBinaryConnect_STE --optim STE
 
 Cifar10: python main_cifar10.py --model MLPBinaryConnect_STE --optim STE
 
 Cifar100: python main_cifar100.py --model MLPBinaryConnect_STE --optim STE
 
 
 
3. Continual Learning

python main_permute_mnist_CL.py

