# AI-Powered HR Assistant & Chatbot

Welcome to the **AI-Powered HR Assistant & Chatbot** repository! This project is a robust Streamlit application designed to help HR teams efficiently manage and interact with their document data using AI-driven insights and natural language processing. The tool allows users to upload documents, ask questions, and receive detailed answers, all powered by OpenAI's GPT-4 and LangChain.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgements](#acknowledgements)

## Overview

The AI-Powered HR Assistant & Chatbot is designed to streamline document management for HR professionals. By leveraging AI, this application enables users to ask specific questions about the content of documents like policies, contracts, and employee handbooks, and get accurate, context-aware answers. The tool also features an intelligent chatbot interface that enhances user interaction and data retrieval.

## Features

- **Interactive User Interface**: A clean, intuitive design built with Streamlit, featuring custom branding and a responsive layout.
- **Document Management**: Supports PDF, DOCX, and TXT files, allowing users to upload and process various document types.
- **AI-Driven Insights**: Uses OpenAI's GPT-4 to answer queries about document content and powers a chatbot for dynamic interactions.
- **Chunk-Based Processing**: Efficiently handles large documents by splitting them into manageable chunks while preserving context.
- **Embedding Cost Calculation**: Automatically calculates and displays the cost of embedding documents, helping users manage their API usage.
- **Session History**: Keeps track of user queries and responses within a session, providing a continuous and reviewable experience.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/symeon158/ai-hr-assistant-chatbot.git
   cd ai-hr-assistant-chatbot

## Usage
Upload a Document: Use the sidebar to upload your PDF, DOCX, or TXT file.
Set Parameters: Adjust chunk size and other settings in the sidebar as needed.
Ask Questions: Enter your query in the input field and receive an AI-generated answer.
Review History: The chat history of your session is automatically maintained for your reference.

## Configuration
Chunk Size: Customize how the document is split into chunks for processing.
API Key: Enter your OpenAI API key in the sidebar to enable the application’s AI functionalities.
Embeddings: The application uses OpenAI Embeddings for document processing.

## Acknowledgements
OpenAI: For providing the GPT-4 model that powers the chatbot and document query functionalities.
Streamlit: For the easy-to-use framework that made building this application a breeze.
LangChain: For the tools used in document processing and chunking.

