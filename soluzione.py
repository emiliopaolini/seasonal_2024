# model hyperparameters
epochs = 25 # how many times the network experiments the whole training set
batch_size = 64 #how many examples the network exmperiments at every step



dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset.dropna(inplace=True)

#y_train = to_categorical(y_train)
#print(y_train)

# Define your model here
# model hyperparameters

epochs = 25 # how many times the network experiments the whole training set
batch_size = 32 #how many examples the network exmperiments at every step


# MODEL 
hidden_units = 1024
output_units = 2
output_activation = 'softmax'
hidden_activation = 'relu'
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow import keras
network = models.Sequential()
network.add(layers.Dense(256, activation=hidden_activation, input_shape=(78,))) # we need to specify the input shape
#network.add(layers.Dropout(0.2))
#network.add(layers.Dense(1024, activation=hidden_activation))
network.add(layers.Dense(128, activation=hidden_activation))
#network.add(layers.Dense(100, activation='relu'))
network.add(layers.Dense(output_units, activation=output_activation))
network.summary() # pay attention to the number of parameters. where do they come from?


y_train = y_train.reshape(y_train.shape[0],1)

print(y_train.shape)
# compile model 
opt = keras.optimizers.Adam(learning_rate=0.001)
network.compile(optimizer=opt, # Adam is a good choice for the optimizer
                loss='sparse_categorical_crossentropy', # sparse categorical crossentropy 
                metrics=['accuracy']) # accuracy = number of correctly labeled samples / number of samples


# fit model here
history = network.fit(X_train, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size,
            shuffle = True)







## SECOND PART

!python3 -m pip install seaborn

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import seaborn as sns
labels = [0,1]

# test the models and report scores
print(np.unique(y_test,return_counts=True))

y_prob = network.predict(X_test) 
print(y_prob)
y_classes = np.argmax(y_prob, axis=1)


from sklearn.metrics import f1_score, accuracy_score

# Calculate the F1 score
f1 = f1_score(y_test, y_classes)
accuracy = accuracy_score(y_test, y_classes)


print(classification_report(y_test, y_classes))
print(confusion_matrix(y_test,y_classes))

# report metrics
# history.history is a dictionary
loss = history.history["loss"]
acc = history.history["accuracy"]
print('loss',loss)
print('accuracy',acc)
plt.figure()
plt.plot(loss,'^-b')
plt.title('Loss')
plt.show()
plt.figure()
plt.plot(acc,'^-b')
plt.title('Accuracy')
plt.show()
#ConfusionMatrixDisplay(confusion_matrix(y_test,y_classes), display_labels=labels).plot()

print("F1-score: ", f1)
print("Accuracy score: ", accuracy)









