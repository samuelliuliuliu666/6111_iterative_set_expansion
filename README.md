# ISE Algorithm Implementation

## Team members

- Ziyu Liu (zl3220)
- Zelin Wang (zw2852)

## Files submitted

1. `ISE.py`
2. `requirements.txt`
3. `README.md`
4. Same other files from [this repository](https://github.com/zackhuiiiii/SpanBERT)

## How to run the program

1. Set up your Google Cloud VM following the provided instructions.

2. Install required packages and dependencies by running:

```sh
sudo apt update
sudo apt install python3-pip
pip3 install beautifulsoup4
sudo apt-get update
pip3 install -U pip setuptools wheel
pip3 install -U spacy
python3 -m spacy download en_core_web_lg
```

3. Clone the SpanBERT repository and set it up:

```sh
git clone https://github.com/zackhuiiiii/SpanBERT
cd SpanBERT
pip3 install -r requirements.txt
bash download_finetuned.sh
```

4. Install OpenAI in your Python environment:

```sh
pip3 install openai
```

5. Update the `ISE.py` script with your Google Custom Search Engine JSON API Key and Engine ID.

6. Run the program with:

```sh
python3 ISE.py
```

## Internal design

The project is designed with the following high-level components:

1. `ISE.py`: Main script that contains the implementation of the ISE algorithm. It includes the following functionality:
    - Searching the web using the Google Custom Search API
    - Extracting plain text from webpages using Beautiful Soup
    - Processing and annotating text using spaCy
    - Extracting relations using the SpanBERT classifier
    - Extracting relations using the OpenAI GPT-3 API

2. `requirements.txt`: A file containing the required packages for the project.

## External libraries used

1. Google Custom Search API: Used for searching the web.
2. Beautiful Soup: Used for extracting plain text from webpages.
3. spaCy: Used for processing and annotating text through linguistic analysis.
4. SpanBERT: Used for extracting relations from text documents.
5. OpenAI GPT-3 API: Used as an alternative to SpanBERT for extracting relations from text documents.

## Step 3 description

Step 3 of the ISE.py implementation involves processing each URL obtained from the Google Custom Search API that has not been processed before. The following steps are carried out:

1. Retrieve the corresponding webpage for each URL. If the webpage cannot be retrieved (e.g., due to a timeout), it is skipped and the program moves on to the next URL.

2. Extract the actual plain text from the webpage using the Beautiful Soup library.

3. If the resulting plain text is longer than 10,000 characters, truncate the text to its first 10,000 characters for efficiency and discard the rest.

4. Use the spaCy library to split the text into sentences and extract named entities (e.g., PERSON, ORGANIZATION).

5. Depending on whether `-spanbert` or `-gpt3` is specified, the program uses either the SpanBERT classifier or the OpenAI GPT-3 API for relation extraction:

- If `-spanbert` is specified, the sentences and named entity pairs are input to SpanBERT to predict the corresponding relations. All instances of the relation specified by input parameter r are extracted.
- If `-gpt3` is specified, the OpenAI GPT-3 API is used for relation extraction.

6. Based on the specified extraction method, tuples are added to set X:

- If `-spanbert` is specified, tuples with an associated extraction confidence of at least t are added to set X.
- If `-gpt3` is specified, all extracted tuples are added to set X. Since extraction confidence values are not provided by the OpenAI GPT-3 API, you can hard-code a value of 1.0 for the confidence value for all GPT-3-extracted tuples.

By following these steps, the `ISE.py` script processes each URL, extracts the required relations based on the chosen method, and populates the set X with the relevant tuples. This set will later be used for further processing and analysis as required by the overall ISE algorithm.

## Google Custom Search Engine Credentials

To test the project, please use the following Google Custom Search Engine JSON API Key and Engine ID:

- API Key: AIzaSyBe5rpBXOaw4IW8piccWUA0VZ9gNhUMWgY
- Engine ID: 497c7780958f3ec5b

## Additional Information

1. Error handling: The program has been designed with error handling mechanisms to handle common issues such as timeouts and invalid URLs while processing webpages. However, if you encounter any unexpected errors, please feel free to report them.

2. Performance considerations: The program may take a while to run depending on the number of URLs being processed and the extraction method chosen (SpanBERT or GPT-3). For better performance, you can try adjusting the number of URLs being processed or using the provided confidence threshold `t` to filter out less confident extractions.

3. Limitations: The ISE.py program relies on external APIs and libraries like Google Custom Search, Beautiful Soup, spaCy, SpanBERT, and OpenAI GPT-3. As such, the overall performance and accuracy of the program are subject to the limitations and performance of these APIs and libraries.