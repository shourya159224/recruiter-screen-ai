from job_metadata import JobMetaDataRetriever,ask_recruiter_questions
import streamlit as st

chat_model_name = "gpt-3.5-turbo"
text_embeddings_model = "text-embedding-3-small"
# file_path = "../sde/jd_docs/Interac-SDE.pdf"
# file_path ='../sde/jd_docs/TXT-Fidelity-SDE.txt'

def main():

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "questions" not in st.session_state:
        st.session_state.questions = []

    if "questions_started" not in st.session_state:
        st.session_state.questions_started = 0

    if "questions_done" not in st.session_state:
        st.session_state.questions_done = 0

    if "question_number" not in st.session_state:
        st.session_state.question_number = 0 

    if "len_seq" not in st.session_state:
        st.session_state.len_seq = None 

    def increment_count():
        if st.session_state.question_number <= st.session_state.len_seq:
            st.session_state.question_number += 1

        if st.session_state.question_number > st.session_state.len_seq:
            st.session_state.questions_done = 1

        return None


    with st.sidebar:
        openai_api_key = st.text_input(" Enter OpenAI API Key", key="chatbot_api_key", type="password")

    st.title("🤖 Ariana-AI recruitment screener")
    st.caption("A first round interview screener powered by OpenAI LLM. It uses RAG process to retrieve job metadata and ask questions based on that in order to screen candidates.")
    st.markdown(":red[Please make sure to paste your own Open AI API key in the sidebar.]")
    st_job_description = st.text_input("Paste the job description and then click on start button")

    if st.button("Start interview") and st.session_state.questions_started == 0:

        if not openai_api_key:
            st.write(":red[API KEY NOT FOUND!]")

        else:
            retrieval_job = JobMetaDataRetriever(
            embedding_model=text_embeddings_model, file_path=st_job_description,
            chat_model=chat_model_name, api_key = openai_api_key
            )
            
            jd_metadata = retrieval_job.retrieve_metadata()


            question_sequence = ask_recruiter_questions(jd_metadata)
            st.session_state.questions = question_sequence
            st.session_state.len_seq = len(question_sequence) - 1
            st.session_state.question_number = 0
            st.session_state.questions_started = 1
            bot_message = st.session_state.questions[st.session_state.question_number]
            st.session_state.messages.append({"role": "assistant", "content": bot_message})
            st.session_state.question_number += 1
            bot_message = st.session_state.questions[st.session_state.question_number]
            st.session_state.messages.append({"role": "assistant", "content": bot_message})


    if st.session_state.questions_started == 1:

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"]) 

        if st.session_state.questions_done != 1:
            if prompt := st.chat_input("Enter response",on_submit=increment_count):

                with st.chat_message("user"):
                    st.markdown(prompt)
                
                st.session_state.messages.append({"role": "user", "content": prompt})

                if st.session_state.questions_done != 1:
                    bot_message = st.session_state.questions[st.session_state.question_number]
                    with st.chat_message("assistant"):
                        st.markdown(bot_message)
                    st.session_state.messages.append({"role": "assistant", "content": bot_message})

             


if __name__ == "__main__":
    main()
