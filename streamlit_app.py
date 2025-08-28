import streamlit as st
#import ial_gui
#import oop_al_picker as al_picker

#al_picker.main()

#ial_gui()

# Title for the app
st.title("Streamlit App")

# Displaying the text
if 'button_clicked' not in st.session_state:
    st.session_state['button_clicked'] = False

def button_clicked():
    st.session_state['button_clicked'] = not st.session_state['button_clicked']

# Button
st.button("Click Me", on_click=button_clicked)

# Label updates on button click
if st.session_state['button_clicked']:
    st.text("Button Clicked!")
else:
    st.text("Hello, Streamlit!")
