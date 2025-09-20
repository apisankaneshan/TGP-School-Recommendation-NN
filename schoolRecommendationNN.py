import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam

SCALER = MinMaxScaler()

def normalize_data(data):
    normazlied_data = SCALER.fit_transform(data) #outputs a numpy array,
    normalized_df = pd.DataFrame(normazlied_data, columns=numerical_cols) #convert back to pandas df
    return normalized_df

def apply_one_hot_encoding(data):
    return pd.get_dummies(data, columns=['school_id'], dtype=int) #apply one hot encoding (ohc)

numerical_cols = ['reputation', 'opportunities', 'safety', 'happiness', 'internet', 'location', 'facilities', 'clubs', 'food', 'social']

df = pd.read_csv("schoolReviews.csv", index_col=False)

numerical_df = df[numerical_cols]
school_id_df = df['school_id']


normalized_ndf = normalize_data(numerical_df)
normalized_df = normalized_ndf
normalized_df['school_id'] = school_id_df
ohc_df = apply_one_hot_encoding(normalized_df)

X = ohc_df.iloc[:, :10]
y = ohc_df.iloc[:, 10:]

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=15)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=73)

model = Sequential()
model.add(Input(shape=(10,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(12, activation='softmax'))
model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, epochs=50, batch_size=50, validation_data=(X_val, y_val))
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

model.save('TGP_School_Recommendation_NN.keras')

