import streamlit as st
import pickle
# Text Pre-processing libraries
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Tensorflow imports to build the model.
import nltk
import tensorflow as tf
from keras.utils import pad_sequences

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')

###########
max_words = 5000
max_len = 100

classes = {
    0:"hate speech",
    1:"Offensive Language",
    2:"Neither"
}

###########
st.set_page_config(
    layout="wide", page_title="Hate Speech Detection", page_icon="❄️"
)



st.title("Hate Speech Text Classifier")


# We need to set up session state via st.session_state so that app interactions don't reset the app.

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False


#######################


st.write("")
st.markdown(
    """
Welcome to the Hate Speech Detection app! This tool leverages deep learning to identify hate speech in text inputs, helping to detect and prevent abusive or harmful language in online communications. Using an **LSTM-based model**, this app can classify input text as :orange[**"hate speech"**] , :orange[**"Offensive Language"**] or :orange[**"Neither"**] with high accuracy.

#### **Simply enter any text to analyze, and the model will provide a classification result.**

"""
)

st.write("")


with st.form(key="my_form"):

    MAX_TOKENS = 50

    new_line = "\n"

    pre_defined_text =  "@DomWorldPeace: Baseball season for the win. #Yankees This is where the love started"
       



    text = st.text_area(
        # Instructions
        "Enter any text to classify",
        # 'sample' variable that contains our text example.
        pre_defined_text,
        # The height
        height=200,
        # The tooltip displayed when the user hovers over the text area.
        help="",
        key="1",
    )

    ### text preprocessing
    text = text.lower()

    # Removing punctuations
    punctuations_list = string.punctuation
    text = text.translate(str.maketrans('', '', punctuations_list))
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    
    words = []
    
    
    for word in str(text).split():
        if word not in stop_words:

			# Lemmatizing the word
            lemmatized_word = lemmatizer.lemmatize(word)
            words.append(lemmatized_word)
    text = " ".join(words)


    # Tokenizer
    # Load the tokenizer
    import os
    
    with open('/mount/src/hate_speech_detection/hate_speech_detection/pickle/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    # Convert text to sequences
    text_seq = tokenizer.texts_to_sequences([text])

    # Pad sequences for uniform length
    text_padded = pad_sequences(text_seq, maxlen=max_len)
    model = tf.keras.models.load_model('/mount/src/hate_speech_detection/hate_speech_detection/pickle/model.keras')
    # Get prediction
    prediction = model.predict(text_padded)
    # Return the class with the highest probability
    predicted_class = prediction.argmax(axis=-1)[0]
    prediction_confidence = round(prediction.max(),2)
    print(prediction_confidence)


    submit_button = st.form_submit_button(label="Submit")

    ############ CONDITIONAL STATEMENTS ############

    if not submit_button and not st.session_state.valid_inputs_received:
        st.stop()

    elif submit_button and not text:
        st.warning("❄️ There is no text to classify")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button or st.session_state.valid_inputs_received:

        if submit_button:

            # The block of code below if for our session state.
            # This is used to store the user's inputs so that they can be used later in the app.

            st.session_state.valid_inputs_received = True

        ############ THE Model Prediction  ############


       
        st.success("✅ Done!")

        st.caption("")
        st.markdown("### Check the results!")
        st.markdown(f"### ==> The model predict the text as :")
        st.caption("")
        if predicted_class == 2:
            st.success(classes[predicted_class])
        else:
            st.warning(classes[predicted_class])
        st.markdown("### :blue[Confidence] = "+str(prediction_confidence))


