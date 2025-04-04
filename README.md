# End-to-end-Medical-Chatbot-Generative-AI

# How to run?

### STEPS:

Clone the repository

```bash
Project repo: git@github.com:ashutosh678/Medical-Chat-Bot.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```

### STEP 02- install the requirements

```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GOOGLE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,

```bash
open up localhost:
```

### Techstack Used:

- Python
- LangChain
- Flask
- Gemini
- Pinecone

## Create ECR repo to store/save docker image

- save the URI: 959610928395.dkr.ecr.ap-south-1.amazonaws.com/medicalchatbot
