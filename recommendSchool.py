import numpy as np
from keras.models import load_model

UNIVERSITIES = ['uoft', 'waterloo', 'mcmaster', 'ubc', 'queens', 'western', 'mcgill', 'ottawa', 'york', 'toronto-met', 'carleton', 'guelph']

def normalize_array(array):
    normalized_array = [((val - 1) / (4)) for val in array]
    return normalized_array

def apply_one_hot_encoding(array):
    for i, val in enumerate(UNIVERSITIES):
        one_hot_vector = [0]*len(UNIVERSITIES)
        one_hot_vector[i] = 1
        preferences = array+one_hot_vector
    return preferences

model = load_model('TGP_School_Recommendation_NN.keras')

# numerical_cols = ['reputation', 'opportunities', 'safety', 'happiness', 'internet', 'location', 'facilities', 'clubs', 'food', 'social']
my_preferences = [5, 5, 2, 3, 3, 2, 1, 5, 5, 5]

normalized_preferences = normalize_array(my_preferences)
preferences = np.array(normalized_preferences)
preferences = preferences.reshape(1, -1)

predicted_unis = model.predict(preferences)

predictions_with_names = list(zip(UNIVERSITIES, predicted_unis[0]))
predictions_with_names.sort(key=lambda x: x[1], reverse=True)
print("University Recommendations (Highest to Lowest Score):")
for uni, score in predictions_with_names:
    print(f"- {uni}: {score:.4f}")