from rich import print
import keras
from keras.layers import Input, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

data = pd.read_csv("iris.csv")

# Separate features (X) and labels (y)
X = data.drop(columns=["variety"])  # Assuming "variety" is the target column
y = data["variety"]

# Encode labels
y = OneHotEncoder(sparse_output=False).fit_transform(y.values.reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

inputs = Input(shape=(4,), name="input")
layer_x = Dense(6, activation="relu", name="dense_1")(inputs)
layer_x = Dense(4, activation="relu", name="dense_2")(layer_x)
outputs = Dense(3, activation="softmax", name="predictions")(layer_x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Fit the model
history = model.fit(X_train, y_train, epochs=80, validation_data=(X_test, y_test))

model.save("iris_model_hot.keras")

print("model.summary:")
print(model.summary())
print("-" * 30)

result = model.evaluate(X_test, y_test)

# slice test data to get the first x rows
X_test_subset = X_test
predictions = model.predict(X_test_subset)

data_variety = np.array([["Setosa"], ["Versicolor"], ["Virginica"]])
print(":heavy_minus_sign:" * 47)
# Iterate through the predictions and get the expected variety
for i in range(len(predictions)):
    variety = data.iloc[X_test_subset.index[i]]["variety"]
    indices = np.where(data_variety == variety)
    highest_index = np.argmax(predictions[i])
    if highest_index == indices[0][0]:
        checkmark = "[green] :heavy_check_mark:"
    else:
        checkmark = "[red] :double_exclamation_mark:"
    print(
        f"{checkmark} - Sample {i + 1:03d}: Predictions: {predictions[i]}, Expected: {indices[0][0]} - {variety}"
    )

print(":heavy_minus_sign:" * 47)
print(f"Test loss: {result[0]} / Test accuracy: {result[1]}")
