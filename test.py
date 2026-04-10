

'''from numpy import loadtxt
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("model.weights.h5")

predictions = (model.predict(x_test) > 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions)
print("Test Accuracy: %.2f%%" % (accuracy * 100))'''
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Load dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x = dataset[:, 0:8]
y = dataset[:, 8]

# ✅ Normalize data (VERY IMPORTANT)
scaler = StandardScaler()
x = scaler.fit_transform(x)

# ✅ Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# ✅ Improved Neural Network
model = Sequential()
model.add(Dense(16, input_dim=8, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ✅ Train model (more epochs)
model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=1)

# ✅ Predictions
predictions = (model.predict(x_test) > 0.5).astype(int)

# ✅ Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Improved Test Accuracy: %.2f%%" % (accuracy * 100))

# ✅ Confusion Matrix (VER
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)
