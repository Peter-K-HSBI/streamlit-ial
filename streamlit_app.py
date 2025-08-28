import streamlit as st
#import ial_gui
#import oop_al_picker as al_picker

#al_picker.main()

#ial_gui()

st.header("InterActive Learner")
st.text("Hello! This is the entrypoint page for the InterActive Learner. Here are some descriptions of the available modes.")

col1, col2 = st.columns(2)

col1.header("This is the first column.")
col1.text("I guess I will describe the first mode here.")

col2.header("This is the second column.")
col2.text("I might put a description of the competition mode here.")
