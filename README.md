# Langchain Custom PDF Document Question Asker

This is an example console question and answer app that loads in a set of PDFs (recursively from PDF_ROOT directory)  and let's you ask questions about them using a semantic search. Semantic search is meaning-based instead of keyword.

It uses langchain (https://github.com/hwchase17/langchain/) with Postgres pgvector plug-in as the datastore and hkunlp/instructor-large (https://huggingface.co/hkunlp/instructor-large) open source embeddings from Hugging face. This allows you to create embeddings for free instead of using OpenAI's and needing to pay. 

It does use OpenAI llm for the question processing. I hope to change this to an open source llm that runs locally as well.


## Installation


### Create a default .env

```bash
cp .env.example .env
```

Modify with your specifics including:

1. The place you would like your postgres data along with connection info
2. Path to your documents
3. Name of your collection (can be anything)
4. Can use `postgres` as `PGVECTOR_DATABASE` or create one (see below for how to create a new database)
5. live openai key (Only used for `query_data.py`)

### Install pgvector
To install pgvector using Docker (make sure you have docker installed)

```bash
git clone https://github.com/pgvector/pgvector
cd pgvector
sudo docker build .
```

Start container with a script similar to `docker_run.sh.example` (make a copy first if desired)

### Configure Python environment

I recommend using a virtualenv with conda or pyenv. You will also need an nvidia video card and cuda drivers installed.

```bash
pip install python-dotenv openai langchain InstructorEmbedding pypdf pgvector psycopg2-binary torch torchvision torchaudio sentence_transformers
```

### Load data

```bash
python load_data.py
```

### Ask Questions about PDFs

```bash
python query_data.py
```


### Create Database via command line

```bash
docker exec -it pg-docker bash
```

Once in shell

```bash
root@2045d767567:/# psql -U postgres
psql (15.3 (Debian 15.3-1.pgdg110+1))
Type "help" for help.

postgres=# CREATE DATABASE {YOUR DATABASE NAME}
```


### `StatementError` Workaround   ( as of 5/13/2023)

If you get 

`StatementError: (builtins.ValueError) expected 1536 dimensions, not 768`

You must change a couple of things

in {your python install}/site-packages/langchain/vectorstores/pgvector.py

change
`ADA_TOKEN_COUNT = 768`
instead of `1536`

in the database

```bash
docker exec -it pg-docker bash
```

Once in shell

```bash
root@2045d767567:/# psql -U postgres
psql (15.3 (Debian 15.3-1.pgdg110+1))
Type "help" for help.

postgres=#  \c {YOUR DATABASE NAME}
postgres=# alter table langchain_pg_embedding alter column embedding type vector(768);
```

ref: https://github.com/hwchase17/langchain/issues/2219
