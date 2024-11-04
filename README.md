# AI-Powered HR Assistant & Chatbot

Welcome to the **AI-Powered HR Assistant & Chatbot** repository! This project is a comprehensive Streamlit application crafted to assist HR teams in effectively managing and interacting with document data through AI-driven insights and natural language processing (NLP). By combining the capabilities of OpenAI's GPT-4 and LangChain, the assistant provides precise answers to document-specific questions, enhancing productivity and data accessibility.

## 🎥 Demo Video
[![Watch the video](https://1drv.ms/v/c/A9927BE78AA24F21/EYGrivVWTNBCiM-ZpqDzbBIBMs-DgxsIXlGbu-qHDHEWsw?e=L73ZZy))
> 👉 **Click the image above** to watch the demo video and see how the AI Assistant works

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Examples](#examples)
8. [Future Enhancements](#future-enhancements)
9. [Acknowledgements](#acknowledgements)

## Overview

The **AI-Powered HR Assistant & Chatbot** is tailored to streamline document management for HR professionals. It enables users to upload various document types—such as policies, contracts, and employee handbooks—and ask detailed questions about their content. The assistant retrieves document-specific answers and uses an intelligent chatbot interface for a seamless, interactive experience.

## Features

- **Interactive User Interface**: Built with Streamlit, the application features a responsive and intuitive layout with custom branding for a polished user experience.
- **Multi-Format Document Management**: Supports a wide range of document types, including PDF, DOCX, and TXT, allowing users to upload and process various file formats.
- **AI-Powered Chatbot**: Leverages OpenAI's GPT-4 for intelligent, context-aware responses, allowing users to ask both general and document-specific questions.
- **Retrieval-Augmented Generation (RAG)**: Integrates LangChain to handle large documents by splitting them into manageable chunks, improving retrieval accuracy and response relevance.
- **Session-Based Interaction**: Maintains a history of user queries and responses within a session, providing continuous, reviewable interaction.
- **Parameter Customization**: Adjustable settings for chunk size, overlap, temperature, and top-k retrieval help users control how documents are processed and responses are generated.
- **Embedding Cost Calculation**: Automatically calculates the cost of embedding documents, helping users monitor and manage API usage.
- **Local Data Processing**: Ensures data privacy and confidentiality by processing documents locally, ideal for sensitive HR files.

## Architecture

The application architecture consists of three main components:

1. **Frontend (Streamlit)**: Provides a web interface where users can upload documents, set processing parameters, and interact with the chatbot.
2. **Backend (LangChain + OpenAI GPT-4)**:
   - **Document Embedding and Retrieval**: Uses LangChain to embed documents in chunks and enables efficient retrieval for context-based querying.
   - **LLM Integration**: Integrates OpenAI’s GPT-4 for generating responses based on document data.
3. **Storage**: Stores embeddings and session data to support continuous interaction and efficient data retrieval.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/symeon158/ai-hr-assistant-chatbot.git
   cd ai-hr-assistant-chatbot

Install dependencies: Make sure you have Python installed. Then, install the required packages!

## Usage
Upload a Document: Use the sidebar to upload your PDF, DOCX, or TXT file.
Set Parameters: Adjust chunk size, overlap, temperature, and top-k retrieval settings in the sidebar to tailor the processing.
Enter API Key: Input your OpenAI API key in the sidebar to enable AI functionalities.
Ask Questions: Enter your question in the input field to receive an AI-generated answer.
Review History: The chat history of your session is maintained in the interface, providing a reviewable log of questions and responses.
Configuration
The app provides several configurable options in the sidebar for a tailored experience:

Chunk Size: Controls how large each document chunk is for embedding. Larger chunks preserve more context but may require more tokens.
Chunk Overlap: Adds overlap between chunks to maintain continuity, especially useful for lengthy paragraphs or sentences split across chunks.
Temperature: Adjusts the creativity of responses. Higher values yield more creative answers, while lower values produce more focused, deterministic answers.
Top-k Retrieval: Defines how many top-ranked chunks are retrieved for answering queries, optimizing context relevance.
Examples
Example Queries:

"What are the key benefits mentioned in the employee handbook?"
"Provide an overview of the health and safety policies."
"List the onboarding procedures for new hires."
Sample Response: Based on the query, the assistant will retrieve the most relevant document sections and generate a concise, context-aware response using GPT-4.

## Future Enhancements
Multi-Language Support: Expand capabilities to support multiple languages for broader applicability.
Additional File Formats: Add support for XLSX, CSV, and other file types commonly used in HR.
Enhanced Security Features: Implement encryption options for even greater data protection.
Acknowledgements
OpenAI: For providing the GPT-4 model, which powers the chatbot and document query functionalities.
Streamlit: For the framework that enables a fast and interactive UI.
LangChain: For the tools used in document embedding, chunking, and retrieval.

## Acknowledgements
OpenAI: For providing the GPT-4 model, which powers the chatbot and document query functionalities.
Streamlit: For the framework that enables a fast and interactive UI.
LangChain: For the tools used in document embedding, chunking, and retrieval.
