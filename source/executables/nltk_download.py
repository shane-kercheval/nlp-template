# https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed
import nltk
import ssl


def main():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('stopwords')


if __name__ == '__main__':
    main()
