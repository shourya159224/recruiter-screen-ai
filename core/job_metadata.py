from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain.memory.buffer import ConversationBufferMemory
import os, shutil
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain

n_matching_documents = 3


class JobMetaDataRetriever:

    def __init__(self, embedding_model: str, file_path: str, chat_model: str, api_key: str):
        self.embedding_model = embedding_model
        self.file_path = file_path
        self.chat_model = chat_model
        self.model_api_key = api_key

    def retrieve_metadata(self) -> dict:
        """
        This function will process the job description based on file. Split the text and create
        embeddings and store in Chroma DB.
        It will then ask relevant questions to Chat GPT and get metadata for each job.
        this will then be used
        """

        # Returned json metadata definition
        jd_metadata = {"job_title": None, "company_name": None, "job_location": None}

        job_location_prompt = (
            PromptTemplate.from_template("Where is the job currently located?")
            + " Answer the question in the format of city,state,country."
        )

        prompt_company_name = (
            "What is the name of the company from this job description?"
        )
        prompt_job_title = "What is the exact job title from this job description?"

        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=0,
                is_separator_regex=False,
            )

        # choose loader based on the type of file
        if True:
            pages = text_splitter.create_documents([self.file_path])

        # Original piece of code was designed to handle pdf/txt files
        # Due to limitations of streamlit on file processing removing this piece of code.
        else: 
            if self.file_path[-4:].lower() == ".pdf":
                loader = PyPDFLoader(self.file_path)

            else:
                loader = TextLoader(self.file_path)

            pages = loader.load_and_split(text_splitter=text_splitter)

        # Embed documents
        # check if documents are already embedded, cleanup directory if that's the case
        if "chroma_db" in os.listdir("./"):
            shutil.rmtree("./chroma_db")

        embeddings_model = OpenAIEmbeddings(model=self.embedding_model, openai_api_key=self.model_api_key)
        db = Chroma(persist_directory="./chroma_db").from_documents(
            pages, embeddings_model
        )
        # db = Chroma().from_documents(
        #     pages, embeddings_model
        # )

        # # Prompt and find relevant piece of context
        docs = db.similarity_search(
            job_location_prompt.format(), k=n_matching_documents
        )

        # for doc in docs:
        #     print("\n >>>>>>>>>")
        #     print("\n {}".format(doc[1]))
        #     print(doc[0])
        #     print("\n <<<<<<<<<")

        model = ChatOpenAI(model_name=self.chat_model, openai_api_key=self.model_api_key)

        job_chat_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="You're and extremely efficient assistant and give out only precise answers."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )

        job_buffer_memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        llm_location_chain = self.generate_llm_chain(
            model, job_chat_template, job_buffer_memory
        )
        chain_company_name = self.generate_llm_chain(
            model, job_chat_template, job_buffer_memory
        )
        chain_job_title = self.generate_llm_chain(
            model, job_chat_template, job_buffer_memory
        )

        for doc_message in docs:
            llm_location_chain.memory.chat_memory.add_user_message(
                doc_message.page_content
            )

        job_location_question = (
            "Where is the job located? Only Answer in the format of City,Country."
        )
        jd_metadata["job_location"] = llm_location_chain.invoke(job_location_question)[
            "text"
        ]

        # RAG process for retrieving company name
        docs_company_name = db.similarity_search(
            prompt_company_name, k=n_matching_documents
        )

        for doc_message in docs_company_name:
            chain_company_name.memory.chat_memory.add_user_message(
                doc_message.page_content
            )

        job_company_name_question = (
            "Give me only the name of the company from the above job description."
        )
        jd_metadata["company_name"] = chain_company_name.invoke(
            job_company_name_question
        )["text"]

        # RAG process for retrieving job title

        docs_job_title = db.similarity_search(prompt_job_title, k=n_matching_documents)

        for doc_message in docs_job_title:
            chain_job_title.memory.chat_memory.add_user_message(
                doc_message.page_content
            )

        job_title_question = (
            "Give me the exact job title from the above job descriptions."
        )

        retrieved_job_title = chain_job_title.invoke(job_title_question)
        jd_metadata["job_title"] = retrieved_job_title["text"]

        return jd_metadata

    def generate_llm_chain(self, input_llm, input_prompt, input_memory):
        gen_llm_chain = LLMChain(
            llm=input_llm, prompt=input_prompt, memory=input_memory
        )
        return gen_llm_chain


def ask_recruiter_questions(job_metadata_dict):

    intro_prompt = PromptTemplate.from_template(
        "Hello my name is Ariana-AI. I am an AI powered technical recruiter."
        + "I will be asking you some basic sets of questions to see if you are a good"
        + " fit for the position of {job_position} at {job_company}"
    )

    work_ex_prompt = (
        "In some brief words please describe your background and experience "
    +"and if possible, please describe in detail,"
    +"the projects which you have worked upon"
    )

    company_interest_prompt = PromptTemplate.from_template(
        "What interests you in working for {job_company}?"
    )

    prompt_job_location = PromptTemplate.from_template(
        "Where are you currently based? Are you comfortable working out of {work_location}?"
    )

    prompt_salary_expectations = "What are your salary expectations for this role?"

    prompt_start_availability = "When are you available to start?"

    prompt_work_authorization = (
        "What is your work permit status for the place in which this job is located?"
    )

    prompt_additional_questions = "Do you have any additional questions that you would like me to pass to the hiring manager?"

    prompt_thank_you = (
        "Thanks for taking the time to chat with me. I have forwarded your responses to "
    + "the hiring manager. We will be in touch soon. PLEASE NOTE THAT THIS IS THE END OF THE CONVERSATION"
    )
    job_question_sequence = [
        intro_prompt.format(job_position = job_metadata_dict['job_title'], job_company=job_metadata_dict['company_name'])
        ,work_ex_prompt
        ,company_interest_prompt.format(job_company=job_metadata_dict['company_name'])
        ,prompt_job_location.format(work_location=job_metadata_dict['job_location'])
        ,prompt_salary_expectations
        ,prompt_start_availability
        ,prompt_work_authorization
        ,prompt_additional_questions
        ,prompt_thank_you
    ]

    return job_question_sequence
