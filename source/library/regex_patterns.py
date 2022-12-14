"""
Most of these regex patterns are taken from:
    https://github.com/blueprints-for-text-analytics-python/blueprints-text/blob/master/ch04/Data_Preparation.ipynb
"""
SUSPICIOUS = r'[&#<>:/{}\[\]\\]'
MARKDOWN_URLS = r'\[([^\[\]]*)\]\([^\(\)]*\)'  # markdown URLs like [Some text](https://....)
ANGLE_BRACKETS = r'<([^<>]*)>'  # tags like <tab>
BRACKETS = r'\[([^\[\]]*)\]'  # text or code in brackets like [0]
STANDALONE_SPECIAL_CHARACTERS = r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)'  # standalone sequences of specials, matches &# but not #cool  # noqa
STANDALONE_SEQUENCES = r'(?:^|\s)[\-=\+]{2,}(?:\s|$)'  # standalone sequences of hyphens like --- or ==  # noqa
WHITESPACE = r'\s+'  # sequences of white spaces

TOKENS_SIMPLE = r'[\w-]*\p{L}[\w-]*'
TOKENS = r"""
( [#]?[@\w'’\.\-\:]*\w     # words, hash tags and email adresses
| [:;<]\-?[\)\(3]          # coarse pattern for basic text emojis
| [\U0001F100-\U0001FFFF]  # coarse code range for unicode emojis
)
"""
