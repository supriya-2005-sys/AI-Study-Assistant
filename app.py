import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(page_title="AI Study Assistant", page_icon="🎓", layout="centered")

# Questions database
questions = [
    "What is Python?",
    "What is SQL?",
    "What is DBMS?",
    "What is Artificial Intelligence?",
    "What is Machine Learning?",
    "What is Data Science?",
    "What is HTML?",
    "What is CSS?",
    "What is JavaScript?",
    "What is an API?",
    "What is a database?",
    "What is a primary key?",
    "What is a foreign key?",
    "What is a loop in Python?",
    "What is a function in Python?",
    "What is a list in Python?",
    "What is a tuple in Python?",
    "What is a variable?",
    "What is an algorithm?",
    "What is an operating system?",
    "What is computer networking?",
    "What is cloud computing?",
    "What is cybersecurity?",
    "What is data structure?",
    "What is an object in Python?",
]

# Answers database
answers = [
    "Python is a high-level programming language. It is easy to learn and used in web development, AI and data science.",
    "SQL stands for Structured Query Language. It is used to store, manage and retrieve data from databases.",
    "DBMS stands for Database Management System. It is software used to create and manage databases.",
    "Artificial Intelligence (AI) is a technology that enables computers to perform tasks that normally require human intelligence.",
    "Machine Learning is a branch of AI that allows computers to learn patterns from data and make predictions.",
    "Data Science is the process of collecting, analyzing and understanding data to find useful information.",
    "HTML stands for HyperText Markup Language. It is used to create the structure of web pages.",
    "CSS stands for Cascading Style Sheets. It is used to design and style web pages.",
    "JavaScript is a programming language used to make websites interactive.",
    "API stands for Application Programming Interface. It allows two software applications to communicate.",
    "A database is an organized collection of data that can be stored and accessed easily.",
    "A primary key uniquely identifies each record in a database table.",
    "A foreign key connects one database table with another table.",
    "A loop is used to repeat a block of code multiple times. Python has for and while loops.",
    "A function is a reusable block of code that performs a specific task. It is defined using def.",
    "A list is an ordered and changeable collection of items in Python.",
    "A tuple is an ordered collection of items that cannot be changed after creation.",
    "A variable is a named storage location used to store data in a program.",
    "An algorithm is a step-by-step procedure used to solve a problem.",
    "An operating system manages computer hardware and software. Examples are Windows and Linux.",
    "Computer networking connects computers to share information and resources.",
    "Cloud computing provides computing services over the internet.",
    "Cybersecurity protects computers, networks and data from cyber attacks.",
    "A data structure is a way of organizing and storing data efficiently.",
    "An object is an instance of a class in Python. It contains data and methods.",
]

# NLP model
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)


# Function to find answers
def get_answer(user_question):

    if not user_question.strip():
        return "Please enter a question."

    greetings = ["hi", "hello", "hey"]

    if user_question.lower().strip() in greetings:
        return "Hello! Welcome to AI Study Assistant. How can I help you?"

    user_vector = vectorizer.transform([user_question])

    similarity = cosine_similarity(user_vector, question_vectors)

    best_match = similarity.argmax()
    confidence = similarity[0][best_match]

    if confidence < 0.18:
        return "Sorry, I could not find an answer. Please ask a computer science related question."

    return answers[best_match]


# Main heading
st.title("🎓 AI Study Assistant")

st.write("Welcome to your smart learning companion!")

st.info(
    "Ask questions about Python, SQL, DBMS, "
    "Artificial Intelligence and Computer Science."
)

# Sidebar
with st.sidebar:

    st.header("📚 About Project")

    st.write(
        "AI Study Assistant helps students "
        "understand basic computer science concepts."
    )

    st.divider()

    st.subheader("Available Topics")

    topics = [
        "Python",
        "SQL",
        "DBMS",
        "Artificial Intelligence",
        "Machine Learning",
        "Web Development",
    ]

    for topic in topics:
        st.write("•", topic)

    st.divider()

    st.caption("Developed using Python, Streamlit and NLP.")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.write(message["content"])


# Suggested questions
st.subheader("💡 Suggested Questions")

col1, col2 = st.columns(2)

if col1.button("What is Python?", use_container_width=True):
    st.session_state.selected_question = "What is Python?"

if col2.button("What is AI?", use_container_width=True):
    st.session_state.selected_question = "What is Artificial Intelligence?"

if col1.button("What is SQL?", use_container_width=True):
    st.session_state.selected_question = "What is SQL?"

if col2.button("What is Machine Learning?", use_container_width=True):
    st.session_state.selected_question = "What is Machine Learning?"


# Chat input
user_input = st.chat_input("Type your question here...")

if "selected_question" in st.session_state:
    user_input = st.session_state.pop("selected_question")


# Generate response
if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})

    response = get_answer(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()


# Clear chat button
if st.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()


# Footer
st.divider()

st.caption("AI Study Assistant | BCA Academic Project")
