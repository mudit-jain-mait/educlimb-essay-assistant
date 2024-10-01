import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables and configure the API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_feedback_chain():
    """Setup the chain for generating feedback on the essay."""
    prompt_template = """
    You are a kind and supportive educational assistant for students.

    Class Level: {class_level}
    Topic: {topic}
    Essay:
    {essay}

    Please provide constructive feedback on the essay considering the class level and topic. Your feedback should:

    - Guide and motivate the student to accept their mistakes.
    - Point out specific areas where they can improve, providing relevant examples.
    - Explain any language rules or concepts related to their mistakes.
    - Provide a revised version of their essay, slightly modifying their writing style but keeping it appropriate for their class level.

    Be encouraging and focus on helping the student learn and grow.
    
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["class_level", "topic", "essay"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def main():
    st.set_page_config(page_title="EduClimb")
    st.title("EduClimb: Your Personal Writing Coach✏️")

    # Initialize feedback_provided in session_state if not already present
    if 'feedback_provided' not in st.session_state:
        st.session_state['feedback_provided'] = False

    # User inputs for class, topic, and essay
    class_level = st.selectbox("What Grade Are You In?", ["Class 1", "Class 2", "Class 3", "Class 4", "Class 5"], key='class_level')
    topic = st.text_input("What Would You Like to Write About?", key='topic')
    essay = st.text_area("Start Writing Your Essay", key='essay')

    # Process and analyze the essay when the button is clicked
    if st.button("See How I Did!") and not st.session_state['feedback_provided']:
        if essay and class_level and topic:
            with st.spinner("Processing your writing..."):
                # Generate feedback from the essay
                st.subheader("Here's What We Think")
                feedback_chain = get_feedback_chain()
                feedback = feedback_chain.run({
                    "class_level": class_level,
                    "topic": topic,
                    "essay": essay,
                })
                st.write(feedback)
                st.session_state['feedback_provided'] = True  # Set feedback_provided to True
        else:
            st.error("Please provide your class level, essay topic, and write your essay.")

    # Option to write another essay after feedback is provided
    if st.session_state['feedback_provided']:
        if st.button("Let's Write Again"):
            # Reset the session_state variables
            st.session_state['feedback_provided'] = False
            st.session_state['topic'] = ""
            st.session_state['essay'] = ""
            st.experimental_rerun()

if __name__ == "__main__":
    main()
