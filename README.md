# 🔎 Retrieval Augmented Generation (RAG)

This application is designed to enhance data interaction through natural language processing. By uploading documents in various formats (PDF, CSV, XLSX, TXT), users can leverage the power of OpenAI's language models to query and interact with their data directly. This application is particularly useful for data analysts, researchers, and business professionals who need to extract insights from their documents swiftly and conversationally.

## Features
- File Upload: Supports multiple file formats including PDF, CSV, XLSX, and TXT.
- Data Processing: Transforms raw text into actionable insights by extracting key information and generating relevant questions.
- Text Embedding and Retrieval: Uses OpenAI's embedding models to create dense vector representations of the data which are then indexed for efficient retrieval.
- Interactive Chat: Allows users to ask questions and receive answers directly related to the uploaded content.
- Flexible Configuration: Provides options to select different models and parameters to customize the processing according to the user's needs.

## Future Improvements
- Refactor and improve code quality.
- Use already created vector store(s) to decrease cost.
- Improvement of the whole process for better results: loading, indexing, storing, & querying.
- Enable the use of local models (Hugging Face).
- Add & display evaluation to measure the quality.
- Improve front-end.

**Suggestions??**

## What is RAG?
Large Language Models (LLMs) are developed using vast datasets, but they don't initially include specific user data. Retrieval-Augmented Generation (RAG) addresses this by integrating user-specific data with the existing datasets accessible to LLMs.

In the RAG framework, user data is first indexed, making it searchable. When a user submits a query, the system consults this index to extract the most pertinent information. This selected data, along with the user’s query, is then submitted to the LLM in the form of a prompt, prompting the LLM to generate a relevant response.

Whether you are developing a chatbot or another type of interactive agent, understanding how to incorporate RAG techniques to incorporate relevant data into your application is crucial.

# Getting started
## Prerequisites installation

Create a .env file, add your OpenAI API key & desired directory where index will be stored (default is ./index):
```
OPENAI_API_KEY='your_api_key'
DIR_PATH='your_directory'
```

#### Windows
* [Windows Subsystem for Linux 2](https://docs.microsoft.com/en-us/windows/wsl/install)
* [Docker desktop](https://www.docker.com/products/docker-desktop)

**NOTE:** On latest Windows version, command `wsl --install` will automatically configure and install the WSL 2. If the command is missing on your system, you can also do perform [manual installation](https://docs.microsoft.com/en-us/windows/wsl/install-manual)

### Linux
* [Docker](https://docs.docker.com/engine/install/)

### Common
* [Visual Studio code](https://code.visualstudio.com/)
* [Remote Development extension for Visual studio Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)


## Starting up

You can clone the git repository on windows filesystem or inside WSL 2. WSL 2 is preferred as the file system performance is better.

### Windows
Open Powershell or WSL 2 terminal and issue the following commands (you can choose a different name for "myprojects"):
```
mkdir ~/myprojects
cd ~/myprojects
git clone git@github.com:erikhren/RAG.git
cd RAG
code .
```

Once visual studio code opens, select `Extensions` in the left menu and type `Remote Development` extension pack into search field and install the one at the top (it should be authored by Microsoft).

Once the extension is installed, select `Remote explorer` from the left menu and select `Containers` from the drop-down menu at the top of the navigation pane. Click the + sign and select `Open current folder in container`. The container should be built and started.

Alternatively, you can use `Ctrl-Shift-P` and and select `Dev Containers: Open Folder in Container` and select current folder.

### Python interpreter selection

When starting the container for the first time, the extension might incorrectly determine the python interpreter location. To fix this, open any python source file (e.g. cli.py) and check the interpreter in the lower right corner. It should read: 3.11.0 ('.venv': venv). If it shows other value, click on it and select *3.11.0 ('.venv': venv)* from the drop down.

### Python path
Python path environment variable is set in '.devcontainer/devcontainer.json' in `remoteEnv` section. This sets the module root directory to `app` so any imports of custome modules are relative to that directory.
