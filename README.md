# Speech-Emotion-Recognition-Project
 Speech Emotion Recognition (SER) is a project that aims to detect and classify emotions conveyed in human speech. The goal is to develop a system that can accurately recognize and interpret the emotional content of spoken language.

The project involves the following steps:

Data Collection: A large dataset of audio recordings with labeled emotional categories is needed. These recordings can be obtained from various sources, such as public speech databases, online platforms, or custom recordings. In my project I have used Toronto Emotional Speech Set (TESS) dataset There are a set of 200 target words were spoken in the carrier phrase "Say the word _' by two actresses (aged 26 and 64 years) and recordings were made of the set portraying each of seven emotions (anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral). There are 2800 data points (audio files) in total.

Preprocessing: The audio data is preprocessed to extract relevant features. This may involve techniques such as converting the audio into a numerical representation (e.g., spectrograms or Mel-frequency cepstral coefficients), segmenting the recordings into smaller units, and removing noise or irrelevant parts. For which I have used librosa library to extract audio features from sample files.

Feature Extraction: From the preprocessed audio, various features are extracted to capture relevant information for emotion recognition. These features may include pitch, intensity, spectral features, and prosodic cues (e.g., rhythm, intonation). In my project I have used librosa library to extract audio features from sample files.

Emotion Classification: In my model I have trained and tested the machine learning model long short term memory(LSTM). Which is very efficient for SERS system purpose as it memory and it can save the data and learn and train further achieving higher accuracy.

Model Evaluation: The trained model is evaluated using a separate dataset to measure its performance in recognizing emotions accurately. Common evaluation metrics include accuracy, precision, recall, and F1 score. I visualized the data into the graphical form using matplotlib library and after some repeated testing using different values the average accuracy of the model is found to be 98%.

It's worth noting that SER is a challenging task due to the subjectivity and variability of emotions, as well as the potential influence of cultural and individual differences. Therefore, the project often requires careful design, feature engineering, and robust machine learning techniques to achieve reliable emotion recognition from speech.





