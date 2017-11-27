import cnnLoader as cnn

cnnLoader = cnn.CNNLoader('')

model = cnnLoader.MNIST_CNN_Model()
print "MNIST_CNN_Model:"
cnnLoader.MNIST_test_model(model)
model = cnnLoader.MNIST_CNN_sigmoid_Model()
print "MNIST_CNN_sigmoid_Model:"
cnnLoader.MNIST_test_model(model)
model = cnnLoader.MNIST_CNN_adamax_Model()
print "MNIST_CNN_adamax_Model:"
cnnLoader.MNIST_test_model(model)
model = cnnLoader.MNIST_CNN_adamax_2_Model()
print "MNIST_CNN_adamax_2_Model:"
cnnLoader.MNIST_test_model(model)
model = cnnLoader.MNIST_CNN_adamax_3_Model()
print "MNIST_CNN_adamax_3_Model:"
cnnLoader.MNIST_test_model(model)
model = cnnLoader.MNIST_CNN_adamax_4_Model()
print "MNIST_CNN_adamax_4_Model:"
cnnLoader.MNIST_test_model(model)
model = cnnLoader.MNIST_CNN_adam_6_Model()
print "MNIST_CNN_adam_6_Model:"
cnnLoader.MNIST_test_model(model)
model = cnnLoader.MNIST_CNN_adam_16_Model()
print "MNIST_CNN_adam_16_Model:"
cnnLoader.MNIST_test_model(model)
model = cnnLoader.MNIST_CNN_adam_31_Model()
print "MNIST_CNN_adam_31_Model:"
cnnLoader.MNIST_test_model(model)
model = cnnLoader.MNIST_CNN_adam_1_Model()
print "MNIST_CNN_adam_1_Model:"
cnnLoader.MNIST_test_model(model)
model = cnnLoader.MNIST_CNN_adam_2_Model()
print "MNIST_CNN_adam_2_Model:"
cnnLoader.MNIST_test_model(model)
model = cnnLoader.MNIST_CNN_adam_2_Model()
print "MNIST_CNN_adam_3_Model:"
cnnLoader.MNIST_test_model(model)
model = cnnLoader.MNIST_CNN_big_adam_3_Model()
print "MNIST_CNN_big_adam_3_Model:"
cnnLoader.MNIST_test_model(model)