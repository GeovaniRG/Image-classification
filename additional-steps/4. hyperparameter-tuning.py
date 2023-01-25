from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

# Define the model
def create_model(optimizer='adam', hidden_size=32):
    model = Sequential()
    model.add(Dense(hidden_size, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Create the model wrapper
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32)

# Define the hyperparameters and possible values
param_grid = {'optimizer': ['adam', 'sgd'], 'hidden_size': [32, 64, 128]}

# Create the grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

# Fit the grid search
grid_result = grid.fit(X_train, y_train)

# Print the best parameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
