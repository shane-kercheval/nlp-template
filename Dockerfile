FROM python:3.11

WORKDIR /code
ENV PYTHONPATH "${PYTHONPATH}:/code"

RUN apt-get update -y && apt-get install zsh -y
RUN PATH="$PATH:/usr/bin/zsh"

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

RUN pip install -U spacy
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download en_core_web_md

WORKDIR /
RUN pip install fasttext
RUN wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
RUN mkdir fasttext
RUN mv lid.176.ftz fasttext/lid.176.ftz
WORKDIR /code

RUN pip install nltk
RUN python -m nltk.downloader stopwords

CMD '/bin/zsh'
