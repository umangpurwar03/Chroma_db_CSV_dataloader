# Chroma CSV Loader for LangChain

This repository includes a Python script (`csv_loader.py`) showcasing the integration of LangChain to process CSV files, split text documents, and establish a Chroma vector store. The script employs the LangChain library for embeddings and vector stores and incorporates multithreading for concurrent processing.

## Requirements

- [LangChain](https://github.com/langchain-ai): LangChain is a library for natural language processing tasks, such as document loading, text splitting, and vector stores.
- [Chroma](https://github.com/chroma-core/chroma): Chroma is a library for efficient similarity search and clustering of dense vectors.

## Installation

1. Install LangChain, Chroma, and other prerequisites using the following commands:

   ```bash
   pip install langchain
   pip install chroma
   pip install -r requirements.txt
   ```

2. Clone this repository:

   ```bash
   git clone https://github.com/umangpurwar03/Chroma-CSV-Loader-LLM
   ```

3. Navigate to the repository directory:

   ```bash
   cd Chroma-CSV-Loader-LLM
   ```

## Usage

1. Adjust the `data_dir` variable in `csv_loader.py` to point to the directory containing your CSV files.

2. Execute the script:

   ```bash
   python chroma_csv_loader.py
   ```

This script processes each CSV file concurrently using multithreading. It loads the data, splits the text into chunks, generates embeddings using Hugging Face models, and ultimately stores the vectors in a Chroma vector store.

## Customization

- Modify the `model_name` parameter in the `HuggingFaceEmbeddings` initialization to customize the model used for embeddings.

- Tailor the chunk size and overlap in the `RecursiveCharacterTextSplitter` initialization based on your specific requirements.

- Customize other parameters and configurations to align with your unique use case.

## Multithreading

The script employs multithreading to concurrently process multiple CSV files. The `process_csv_files_in_parallel` function initiates a separate thread for each CSV file, enabling efficient parallel processing. Adjust the number of threads to match your system's capabilities and needs.

## License

This code is distributed under the [MIT License](LICENSE). Feel free to utilize and modify it as per your requirements.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai)
- [Chroma](https://github.com/chroma-core/chroma)

If you find this code beneficial or have suggestions for enhancements, please contribute or open an issue.
