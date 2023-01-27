#//Geovani Rodriguez//#

from keras.models import Model
from keras.layers import Input, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import BaggingClassifier

# Define the base model
def create_model():
    inputs = Input(shape=(784,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create a keras wrapper
model = KerasClassifier(build_fn=create_model, epochs=20, batch_size=32)

# Create the ensemble
ensemble = BaggingClassifier(base_estimator=model, n_estimators=10, max_samples=0.8, max_features=0.8)

# Fit the ensemble on the training data
ensemble.fit(X_train, y_train)

# Evaluate the performance of the ensemble on the test data
ensemble_acc = ensemble.score(X_test, y_test)
print("Ensemble accuracy: {:.4f}".format(ensemble_acc))
