import streamlit as st
import os
from app import create_chat, generate_mom


# Set OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    st.error("OpenAI API key is not set!")
    st.stop()

# Sidebar Guide
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    1. **Start the Conversation**
        * Type 'hi' to begin the interview
        * The bot will ask you essential questions about the meeting
    
    2. **During the Interview**
        * Answer each question clearly and concisely
        * Provide all relevant details requested
        * The bot will ask follow-up questions as needed
    
    3. **Completing the Interview**
        * The bot will confirm when all necessary information is gathered
        * You can add any additional information if needed
    
    4. **Generating Minutes**
        * Once the interview is complete, click 'Generate Minutes'
        * The bot will create a formatted Meeting Minutes document
    
    5. **End Session**
        * Type 'quit' to end the conversation
    """)

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = create_chat()
    st.session_state.messages = []
    st.session_state.interview_completed = False
    # Add welcome messages to session state
    welcome_msg = "Welcome to the Meeting Analysis Chatbot!"
    st.session_state.messages.extend([
        {"role": "assistant", "content": welcome_msg}
    ])

# Title
st.title("Meeting Minutes Bot")

# Chat container
chat_container = st.container()
with chat_container:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get bot response
        with st.chat_message("assistant"):
            response = st.session_state.conversation.predict(input=prompt)
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Check if interview is complete
        if not st.session_state.interview_completed:
            if "additional information" in response.lower() and "would like to add" in response.lower():
                st.session_state.interview_completed = True
                st.success("Interview completed! Click 'Generate Minutes' to create Meeting Minutes.")

# Generate MoM button
if st.session_state.interview_completed:
    if st.button("Generate Minutes"):
        with st.spinner("Generating Meeting Minutes..."):
            mom = generate_mom(st.session_state.conversation)
            st.markdown("---")
            st.markdown(mom)
