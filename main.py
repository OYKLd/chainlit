from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
import chainlit as cl
from dotenv import load_dotenv
import google.generativeai as genai
import os 
load_dotenv()


apkikey = os.getenv("GOOGLE_API_KEY")
# Clé API Gemini (ajoute ta clé ici)
genai.configure(api_key=apkikey)




@cl.on_chat_start
async def on_chat_start():
    llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # Spécifiez le modèle Gemini souhaité
    temperature=0.7,         # Ajustez la température selon vos besoins
    streaming=True           # Activez le streaming si nécessaire
)
    prompt = ChatPromptTemplate.from_messages([
    ("system", "Vous êtes un assistant en finance pour faciliter l'accès au financement des femmes."),
    ("human", "{question}")
])
    runnable = prompt | llm | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()