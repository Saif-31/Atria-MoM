from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# System prompts
INTERVIEW_PROMPT = (
    "You are a highly intelligent meeting assistant designed to gather all necessary information for creating accurate and detailed Meeting Minutes (MoM). "
    "Your task is to conduct an interview with a consultant by asking questions in a natural, conversational manner. Begin by asking the essential questions below:\n\n"
    "*Essential Questions:*\n"
    "1. What is the name of the company?\n"
    "2. Who was present at the meeting?\n"
    "3. Where did the meeting take place?\n"
    "4. How long did the meeting last?\n"
    "5. How many employees does the company have?\n"
    "6. How many levels of management does the company have?\n\n"
    "After receiving responses to these, analyze the context of the conversation to determine if additional details are needed. Based on the context, ask any relevant optional questions from the following list to enrich the meeting record:\n\n"
    "*Optional Questions (ask if context indicates relevance):*\n"
    "- What are the company's main strategic goals for this period?\n"
    "- What is the company's focus when it comes to development?\n"
    "- Which target groups within the company are prioritized for development?\n"
    "- What are the main challenges these target groups are currently facing?\n"
    "- Are there any specific competencies or skills the company wants to prioritize across teams?\n"
    "- What learning and development programs are currently in place?\n"
    "- How do you currently measure skill levels and identify training needs?\n"
    "- Which learning formats do employees prefer (online programs, in-person workshops, blended learning)?\n"
    "- Is there any specific format desired for the development (trainings, training days, team building, coaching, etc.)?\n\n"
    "Once you have gathered all the necessary and contextually relevant responses, also ask:\n"
    "- What are the key action items from this discussion?\n"
    "- Who is responsible for following up on these topics?\n"
    "- When should we check in again on the progress of development initiatives?\n\n"
    "Before concluding, confirm with the user if there is any additional information they would like to add. "
    "Your goal is to ensure that every piece of relevant data is captured in a complete meeting record. "
    "Always maintain a conversational tone, adapt your questions based on previous responses, and guide the conversation naturally."
)

MOM_PROMPT = (
    "You are a professional meeting assistant tasked with generating comprehensive Meeting Minutes (MoM) that a consultant can immediately use for follow-ups. Based on the given interview data provided, generate a final MoM document with the following sections and in a clear, business-friendly format:\n\n"
    "---\n\n"
    "## Meeting Minutes (MoM)\n\n"
    "### 1. Meeting Overview\n"
    "- *Company Name:* [Extract from data]\n"
    "- *Meeting Date & Time:* [If available]\n"
    "- *Location:* [Extract from data]\n"
    "- *Duration:* [Extract from data]\n"
    "- *Participants:* [List all names and roles]\n\n"
    "### 2. Meeting Objective\n"
    "- Provide a concise summary of the meeting's purpose (e.g., discussing training needs, leadership development, or strategic planning).\n\n"
    "### 3. Discussion Summary\n"
    "- *Key Topics:*  \n"
    "  Summarize the main discussion points. Include any specific areas such as:\n"
    "  - Strategic goals and development focus\n"
    "  - Target groups for development and current challenges\n"
    "  - Existing training programs and preferred learning formats\n"
    "- *Additional Context:*  \n"
    "  Include any notable insights, pain points, or suggestions mentioned during the discussion.\n\n"
    "### 4. Action Items & Follow-Up\n"
    "- *Action Items:*  \n"
    "  List each agreed-upon action with a brief description.\n"
    "- *Responsibilities:*  \n"
    "  Specify who is responsible for each action.\n"
    "- *Follow-Up:*  \n"
    "  Note the agreed timeline or date for checking progress.\n\n"
    "### 5. Additional Notes\n"
    "- Add any extra information or clarifications provided that do not fit in the sections above.\n\n"
    "---\n\n"
    "Using the raw interview data below, generate the final Meeting Minutes (MoM) in the above format. You can also add something by yourself if its important for consultant. "
    "Ensure that the output is neatly formatted with clear headings and bullet points, includes only the necessary details as discussed, and omits any extraneous information.\n\n"
    "Generate the final Meeting Minutes (MoM) now."
)

def create_chat():
    # Initialize ChatOpenAI for interview
    chat = ChatOpenAI(
        model_name="gpt-4o-mini",  
        temperature=0.7,
    )

    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(INTERVIEW_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    # Initialize memory with message history
    memory = ConversationBufferMemory(return_messages=True)

    # Create interview chain
    conversation = ConversationChain(
        memory=memory,
        prompt=prompt,
        llm=chat,
        verbose=True
    )
    
    return conversation

def create_mom_chain():
    # Initialize ChatOpenAI for MoM generation
    chat = ChatOpenAI(
        model_name="gpt-4o-mini",  
        temperature=0.7,
    )

    # Create prompt template for MoM
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MOM_PROMPT),
        HumanMessagePromptTemplate.from_template("Here is the interview transcript:\n\n{interview_history}")
    ])

    # Create MoM chain without memory since we just need to generate once
    chain = LLMChain(
        llm=chat,
        prompt=prompt,
        verbose=True
    )
    
    return chain

def generate_mom(conversation):
    # Get conversation history from memory
    history = conversation.memory.chat_memory.messages
    
    # Format conversation history as a clear Q&A transcript
    interview_history = ""
    for i in range(0, len(history), 2):
        if i + 1 < len(history):
            question = history[i].content
            answer = history[i + 1].content
            interview_history += f"Q: {question}\nA: {answer}\n\n"
    
    # Create and run MoM chain with interview history
    mom_chain = create_mom_chain()
    mom = mom_chain.run(interview_history=interview_history)
    return mom

def main():
    conversation = create_chat()
    interview_completed = False
    
    print("Welcome to the Meeting Analysis Chatbot!")
    print("Type 'hi' to start conversation or Type 'quit' to end the conversation\n")
    
    while True:
        user_input = input("\nYou: ").strip().lower()
        
        if user_input == 'quit':
            print("\nThank you for using the Meeting Analysis Chatbot!")
            break
        
        response = conversation.predict(input=user_input)
        print(f"\nBot: {response}")
        
        # Check if interview is complete based on the last bot message
        if not interview_completed:
            last_message = response.lower()
            if ("additional information" in last_message and 
                "would like to add" in last_message):
                interview_completed = True
                print("\nInterview completed! Type 'generate mom' to create Meeting Minutes or continue the conversation.\n")
        
        if user_input == 'generate mom':
            if interview_completed:
                print("\nGenerating Meeting Minutes...\n")
                mom = generate_mom(conversation)
                print("=== Meeting Minutes ===")
                print(mom)
                print("=====================")
            else:
                print("\nPlease complete the interview before generating Meeting Minutes.")
            continue

if __name__ == "__main__":
    main()
